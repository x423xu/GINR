import os
import time
import wandb
import numpy as np
import torch
import copy
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import random
import torch.nn as nn

from dataset_zoo import ShapeNetGEM
from utils import plot_single_pcd, make_log, compute_psnr
from model_zoo import meta_modules
from model_zoo.inr_loe import INRLoe
from configs import args



def get_data(args):
    dataset = ShapeNetGEM(split=args.split,
                       sampling=args.sampling,
                       random_scale=args.random_scale,
                       dataset_root=args.dataset_root)
    dataloader = DataLoader(dataset,
                        shuffle=not args.eval,
                        batch_size=args.batch_size,
                        pin_memory=True,
                        num_workers=args.num_workers)
    return dataloader


def image_mse(mask, model_output, gt):
    if mask is None:
        return {'img_loss': ((model_output['model_out'] - gt['img']) ** 2).mean()}
    else:
        return {'img_loss': (mask * (model_output['model_out'] - gt['img']) ** 2).mean()}

def train(args):
    '''
    create log directory and files:
        - log.txt
        - checkpoints
        - imgs
    '''
    logger, log_dir, ckpt_path, img_path, tm_str = make_log(args)
    if args.wandb:
        wandb.init(entity='xxy', project='ginr', dir=args.log_dir, config = args)
        wandb.run.name = tm_str

    train_loader = get_data(args)
    '''
    get model
    '''
    if args.model_type == 'functa':
        model = meta_modules.LatentModulatedSiren(in_channels=args.in_channels,
                                                out_channels=args.out_channels,
                                                width=args.width,
                                                depth=args.depth,
                                                latent_dim=args.hidden_features,
                                                latent_vector_type='instance',
                                                use_meta_sgd=args.use_meta_sgd,
                                                w0=args.w0)
    elif args.model_type == 'mnif':
        model = meta_modules.LinearMixtureINR(width=args.width,
                                              depth=args.depth,
                                              in_channels=args.in_channels,
                                              out_channels=args.out_channels,
                                              k_mixtures=args.k_mixtures,                                             
                                              w0=args.w0,
                                              mixture_type=args.mixture_type,
                                              embedding_type=args.embedding_type,
                                              outermost_linear=args.outermost_linear,
                                              pred_type=args.pred_type,
                                              use_meta_sgd=args.use_meta_sgd,
                                              use_latent_embedding=args.use_latent_embedding,
                                              std_latent=args.std_latent,
                                              latent_channels=args.hidden_features,
                                              norm_latents=args.norm_latents,
                                             )
    elif args.model_type == 'inr_loe':
        model = INRLoe(
                     input_dim=args.input_dim,
                     output_dim=args.output_dim,
                     hidden_dim=args.hidden_dim,
                     num_hidden=args.num_hidden,
                     num_exps=args.num_exps,
                     ks=args.ks,
                     latent_size=args.latent_size,
                     gate_type=args.gate_type,
                     std_latent = args.std_latent,
                     ).cuda()
    else:
        raise NotImplementedError
    model = model.cuda()
    model.train()
    params = sum(p.numel() for p in model.get_parameters())
    logger.info("Total number of parameters is: {}".format(params))
    logger.info("Model size is: {:.2f} MB".format(params * 4 / 1024**2)) 
    param_group = [{'params': model.get_parameters(), 'lr': args.lr_outer}]
    optim = torch.optim.AdamW(param_group, lr=args.lr_outer, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs, eta_min=1e-7)

    keep_params = dict()
    with torch.no_grad():
        for name, param in model.get_named_parameters():
            keep_params[name] = param.clone()
    blend_alphas = [0] * len(model.num_exps)
    meta_grad_init = [0 for _ in model.get_parameters()]
    

    global_steps = 0
    if not args.eval:
        with torch.no_grad():
            for name, param in model.get_named_parameters():
                param = keep_params[name].clone()
    list_context_params, gen_acts = [], []
    for epoch in range(args.epochs):
        all_losses, all_psnr, all_acc, steps = 0.0, 0.0, 0.0, 0
        for step, (model_input_batch, gt_batch) in enumerate(train_loader):
            model_input, gt = model_input_batch, gt_batch
            model_input = {key: value.cuda() for key, value in model_input.items()}
            gt = {key: value.cuda() for key, value in gt.items()}

            if args.gate_type in ['conditional', 'separate']:
                context_params = torch.randn(gt['img'].size(0), len(args.num_exps), args.latent_size).cuda() * args.std_latent
            elif args.gate_type == 'shared':
                context_params = torch.randn(gt['img'].size(0), args.latent_size).cuda() * args.std_latent
            else:
                raise ValueError("Invalid gate type")
            context_params.requires_grad = True

            meta_grad = copy.deepcopy(meta_grad_init)
            
            if args.use_meta_sgd:
                meta_sgd_inner = model.meta_sgd_lrs()
            
            # latents step
            for inner_step in range(args.inner_steps):
                pred_inner = model(model_input, context_params,  args.top_k, 
                                                blend_alphas=blend_alphas)
                loss_inner = ((pred_inner['model_out'] - gt['img']) ** 2).mean()
                grad_inner = torch.autograd.grad(loss_inner,
                                            context_params,
                                            create_graph=not args.eval)[0]
                if args.use_meta_sgd:
                    context_params = context_params - args.lr_inner * (meta_sgd_inner * grad_inner)
                else:
                    context_params = context_params - args.lr_inner * grad_inner
            
            pred_outer= model(model_input,
                              context_params,  
                              args.top_k, 
                              blend_alphas=blend_alphas)
            losses = ((pred_outer['model_out'] - gt['img']) ** 2).mean()
            all_losses += losses.detach().cpu().item()*gt['img'].size(0)
            # PSNR
            for pred_img, gt_img in zip(pred_outer['model_out'].detach().cpu(), gt['img'].cpu()):
                if args.pred_type == 'voxel':
                    psnr = compute_psnr(pred_img * 0.5 + 0.5, gt_img * 0.5 + 0.5) # rescale from [-1, 1] to [0, 1]
                        
                all_psnr += psnr
                steps += 1

            # voxel accuracy
            if args.pred_type == 'voxel':
                pred_voxel = pred_outer['model_out'] >= 0.0 # [non-exist (-1), exists (+1)]
                gt_voxel = gt['img'] >= 0.0
                acc = (pred_voxel == gt_voxel).float().mean()
                all_acc += float(acc) * gt['img'].size(0)

            global_steps += 1
            
            # params step
            losses_all = losses
            task_grad = torch.autograd.grad(losses_all, model.get_parameters())
            for g in range(len(task_grad)):
                meta_grad[g] += task_grad[g].detach()
            optim.zero_grad()
            for c, param in enumerate(model.get_parameters()):
                param.grad = meta_grad[c]
            torch.nn.utils.clip_grad_norm_(model.get_parameters(), max_norm=1.)
            optim.step()


            if step % args.log_every_n_steps == 0:
                description = f'[e{epoch} s{step}/{len(train_loader)}], mse_loss:{all_losses/steps:.4f} PSNR:{all_psnr/steps:.2f} Ctx-mean:{float(context_params.mean()):.8f}'
                if args.pred_type == 'voxel':
                    description += f' acc:{all_acc/steps:.4f}'
                logger.info(description)
                psnr = compute_psnr(pred_outer['model_out'].cpu() * 0.5 + 0.5, gt['img'].cpu() * 0.5 + 0.5)
                if args.wandb:
                    wandb.log({'outer_loss': (all_losses/steps), 'psnr': psnr, 'global_step': global_steps})
                if step % args.save_every_n_steps == 0:
                    ind = pred_outer['model_out'][0]>=0
                    im_show = model_input['coords'][0][ind.squeeze()]
                    plot_single_pcd(im_show.detach().cpu().numpy(), '{}/point_cloud_e{}s{}.png'.format(img_path,epoch, step))
        scheduler.step()
        if epoch % args.ckpt_every_n_epochs == 0:
            torch.save(model.state_dict(), os.path.join(ckpt_path, 'model_{}.pt'.format(epoch)))
                


def main():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    train(args)

main()
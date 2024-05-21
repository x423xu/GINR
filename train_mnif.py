import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
from utils import *
from model_zoo import meta_modules
from model_zoo.vae_backbone import VAEBackbone
from configs import args

torch.autograd.set_detect_anomaly(True)


def get_data(args):
    dataset = ShapeNetGEM(split=args.split,
                       sampling=args.sampling,
                       random_scale=args.random_scale,
                       dataset_root=args.dataset_root,
                       shuffle=not args.eval and args.cache_latents,) # we shuffle only once if cache latents
    dataloader = DataLoader(dataset,
                        shuffle=not args.eval and not args.cache_latents,
                        batch_size=args.batch_size,
                        pin_memory=True,
                        num_workers=args.num_workers)
    return dataloader

def get_model(args):
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
    else:
        raise NotImplementedError
    model = model.cuda()
    model.train()

    if args.vae is not None:
        if args.vae == 'hierarchical_vae':
            vae_model = VAEBackbone(input_channel = args.vae_input_channel, 
                                    latent_channel = args.vae_latent_channel,  
                                    layers = args.vae_layers, 
                                    sampler_dim = args.vae_sampler_dim, 
                                    prior_scale = args.vae_prior_scale, 
                                    num_heads = args.vae_num_heads,
                                    dim_head = args.vae_dim_head,
                                    dropout = args.vae_dropout,
                                    sample_decoder = args.vae_sample_decoder,
                                    latent_dim=args.hidden_features,)
        else:
            raise NotImplementedError
        vae_model = vae_model.cuda()
        vae_model.train()
    else:
        vae_model = None
    return model, vae_model

def make_training_params(args, logger, model, vae_model):
    params = sum(p.numel() for p in model.get_parameters())
    logger.info("Total number of parameters is: {}".format(params))
    logger.info("Model size is: {:.2f} MB".format(params * 4 / 1024**2)) 
    param_group = [{'params': model.get_parameters(), 'lr': args.lr_outer}]
    inr_optim = torch.optim.AdamW(param_group, lr=args.lr_outer, weight_decay=0)
    inr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(inr_optim, T_max=args.epochs, eta_min=1e-7)

    if args.vae is not None:
        vae_optim = torch.optim.AdamW(vae_model.parameters(), lr=args.vae_lr, weight_decay=0, betas=(0.9, 0.999))
        vae_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(vae_optim, T_max=args.epochs, eta_min=1e-7)
    else:
        vae_optim = None
        vae_scheduler = None
    return inr_optim, inr_scheduler, vae_optim, vae_scheduler

def latents_step(args, model, model_input, gt, context_params, meta_sgd_inner, cache_path, step):
    for p in model.get_parameters():
        p.requires_grad = False
    # latents updation step
    for inner_step in range(args.inner_steps):
        pred_inner = model(model_input, context_params)
        loss_inner = image_mse(None,pred_inner, gt)
        grad_inner = torch.autograd.grad(loss_inner['img_loss'],
                                    context_params,
                                    create_graph=not args.eval)[0]
        if args.use_meta_sgd:
            context_params = context_params - args.lr_inner * (meta_sgd_inner * grad_inner)
        else:
            context_params = context_params - args.lr_inner * grad_inner
    '''
    We cache the latents after it is updated.
    '''
    if args.cache_latents:
        torch.save(context_params.detach().cpu(), os.path.join(cache_path, f'd{step}.pt'))
    for p in model.get_parameters():
            p.requires_grad = True
    return context_params

def model_update_step(model, model_input, context_params, inr_optim, meta_grad, gt):
    # model params update step
    '''
    Here the latents are the updated ones, and it is the same as the cached ones.
    '''
    model_output = model(model_input, context_params)
    losses = image_mse(None, model_output, gt)
    losses_all = losses['img_loss']
    task_grad = torch.autograd.grad(losses_all, model.get_parameters())
    for g in range(len(task_grad)):
        meta_grad[g] += task_grad[g].detach()
    inr_optim.zero_grad()
    for c, param in enumerate(model.get_parameters()):
        param.grad = meta_grad[c]
    torch.nn.utils.clip_grad_norm_(model.get_parameters(), max_norm=1.)
    inr_optim.step()
    return model_output, losses

def vae_step(args, model, vae_model, vae_optim, context_params, epoch, step, l_epoch, model_input, gt, vae_loss_avg):
    if args.vae is not None:
        z_dist, kl_all, kl_diag, log_q, log_p = vae_model(context_params.data.unsqueeze(1))
        if args.vae_sample_decoder:
            z,_ = z_dist.sample()
        else:
            z = z_dist
        z = z.squeeze(2)

        # get kld loss
        kl_all = torch.stack(kl_all)
        kl_coeff_i, kl_vals = kl_per_group(kl_all)
        total_kl = torch.sum(kl_coeff_i)

        kl_coeff_i = kl_coeff_i / (1.0 * total_kl)
        kl_coeff_i = kl_coeff_i / torch.mean(kl_coeff_i, dim=0, keepdim=True)
        kld_loss = torch.mean(kl_all * (kl_coeff_i.detach()))

        # get reconstruction loss
        if args.vae_sample_decoder:
            recon_loss = z_dist.log_prob(context_params.data)
        else:
            recon_loss = nn.functional.mse_loss(z, context_params.data,reduction='sum')
        
        # do not calculate the gradient for the model
        with torch.no_grad():
            model_output = model(model_input, z)
        mse_loss = image_mse(None, model_output, gt)['img_loss']
        
        # get kld coefficient
        if epoch%args.annealing_every_n_epochs<args.warmup_epochs:
            kl_ratio = args.kl_r_max*(step+epoch%args.warmup_epochs*l_epoch)/(args.warmup_epochs*l_epoch)                
        elif epoch%args.annealing_every < args.annealing_every:
            kl_ratio = args.kl_r_max
        vae_loss = 0.0
        for l in args.vae_loss_type.split('+'):
            weight, loss_name = l.split('*')
            if loss_name == 'kld':
                vae_loss += float(weight) * kl_ratio * kld_loss
            elif loss_name == 'recon':
                vae_loss += float(weight) * recon_loss
            elif loss_name == 'mse':
                vae_loss += float(weight) * mse_loss
        vae_loss_avg.update(vae_loss.item())
        vae_optim.zero_grad()
        vae_loss.backward()
        nn.utils.clip_grad_norm_(vae_model.parameters(), max_norm=5.)
        vae_optim.step()

        return mse_loss.detach().cpu().item(), recon_loss.detach().cpu().item(), kld_loss.detach().cpu().item(), model_output
    return None, None, None, None

def get_logs(losses, batch_size, model_output, gt, all_losses, all_psnr, all_acc, steps):
    train_loss = 0.
    for loss_name, loss in losses.items():
        single_loss = loss.mean()
        train_loss += single_loss.cpu()
    all_losses += float(train_loss) * batch_size
    # PSNR for (latents step)
    for pred_img, gt_img in zip(model_output['model_out'].cpu(), gt['img'].cpu()):
        if args.pred_type == 'voxel':
            psnr = compute_psnr(pred_img * 0.5 + 0.5, gt_img * 0.5 + 0.5) # rescale from [-1, 1] to [0, 1]
                
        all_psnr += psnr
        steps += 1

    # voxel accuracy for (latents step)
    if args.pred_type == 'voxel':
        pred_voxel = model_output['model_out'] >= 0.0 # [non-exist (-1), exists (+1)]
        gt_voxel = gt['img'] >= 0.0
        acc = (pred_voxel == gt_voxel).float().mean()
        all_acc += float(acc) * batch_size
    return all_losses, all_psnr, all_acc, steps
    

def train(args):
    '''
    create log directory and files:
        - log.txt
        - checkpoints
        - imgs
        - cache (if cache_latents is True)
    '''
    logger, log_dir, ckpt_path, img_path, tm_str = make_log(args)
    cache_path = make_cache(args, log_dir)

    '''training configuration steps'''
    make_wandb(args, tm_str)
    train_loader = get_data(args)
    model, vae_model = get_model(args)    
    inr_optim, inr_scheduler, vae_optim, vae_scheduler = make_training_params(args, logger, model, vae_model)
    keep_params = dict()

    '''
    Not sure how do this affect the training process. Just follow the original mnif code.
    '''
    with torch.no_grad():
        for name, param in model.get_named_parameters():
            keep_params[name] = param.clone()
    meta_grad_init = [0 for _ in model.get_parameters()]
    global_steps = 0
    '''
    Not sure how do this affect the training process. Just follow the original mnif code.
    '''
    if not args.eval:
        with torch.no_grad():
            for name, param in model.get_named_parameters():
                param = keep_params[name].clone()

    for epoch in range(args.epochs):
        all_losses, all_psnr, all_acc, steps = 0.0, 0.0, 0.0, 0
        vae_loss_avg = AverageValueMeter() if args.vae is not None else None
        for step, (model_input_batch, gt_batch) in enumerate(train_loader):
            # prepare data
            model_input, gt = model_input_batch, gt_batch
            model_input = {key: value.cuda() for key, value in model_input.items()}
            gt = {key: value.cuda() for key, value in gt.items()}
            batch_size = gt['img'].size(0)
            meta_grad = copy.deepcopy(meta_grad_init)
            '''
            We use the cached latents if the cache_latents is True. In the first epoch, we will not use the cached latents.
            Another point: if the cache_latents is True, the training data cannot be shuffled.
            '''
            if args.cache_latents and os.path.exists(os.path.join(cache_path, f'd{step}.pt')):
                rand_params = torch.load(os.path.join(cache_path, f'd{step}.pt'), map_location='cpu').cuda()
                context_params = rand_params.detach()
                context_params.requires_grad_()
            else:
                context_params = model.get_context_params(batch_size, args.eval)
            if args.use_meta_sgd:
                meta_sgd_inner = model.meta_sgd_lrs()
            
            context_params = latents_step(args, model, model_input, gt, context_params, meta_sgd_inner, cache_path, step)      
            model_output, losses = model_update_step(model, model_input, context_params, inr_optim, meta_grad, gt)
            vae_mse, recon_loss, kld_loss, vae_out = vae_step(args, model, vae_model, vae_optim, context_params, epoch, step, len(train_loader), model_input, gt, vae_loss_avg)   
            
            # log step
            global_steps += 1       
            all_losses, all_psnr, all_acc, steps = get_logs(losses, batch_size, model_output, gt, all_losses, all_psnr, all_acc, steps) 
            if step % args.log_every_n_steps == 0:
                description = f'[e{epoch} s{step}/{len(train_loader)}], mse_loss:{all_losses/steps:.4f} PSNR:{all_psnr/steps:.2f} Ctx-mean:{float(context_params.mean()):.8f}'
                if args.vae is not None:
                    description += f' VAE-loss:{vae_loss_avg.avg:.4f}, Recon-loss:{recon_loss:.4f}, KLD-loss:{kld_loss:.4f}, VAE_mse_loss:{vae_mse:.4f}'
                logger.info(description)
                psnr = compute_psnr(model_output['model_out'].cpu() * 0.5 + 0.5, gt['img'].cpu() * 0.5 + 0.5)
                if args.wandb:
                    wandb.log({'outer_loss': (all_losses/steps), 'psnr': psnr,'global_step': global_steps})
                    if args.vae is not None:
                        wandb.log({'vae_loss': vae_loss_avg.avg, 'recon_loss': recon_loss, 'kld_loss': kld_loss, 'vae_mse_loss': vae_mse, 'global_step': global_steps})
                if step % args.save_every_n_steps == 0:
                    ind = model_output['model_out'][0]>=0
                    im_show = model_input['coords'][0][ind.squeeze()]
                    plot_single_pcd(im_show.detach().cpu().numpy(), '{}/point_cloud_e{}s{}.png'.format(img_path,epoch, step))
                    if args.vae is not None:
                        vae_ind = vae_out['model_out'][0]>=0
                        vae_show = model_input['coords'][0][vae_ind.squeeze()]
                        plot_single_pcd(vae_show.detach().cpu().numpy(), '{}/vae_point_cloud_e{}s{}.png'.format(img_path,epoch, step))
            
        inr_scheduler.step()
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
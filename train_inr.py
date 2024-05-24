import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
from model_zoo.inr_loe import INRLoe
import model_zoo.vae_backbone as vb
from configs import args

torch.autograd.set_detect_anomaly(True)


def get_data(args):
    def _seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    dataset = ShapeNetGEM(split=args.split,
                       sampling=args.sampling,
                       random_scale=args.random_scale,
                       dataset_root=args.dataset_root,
                       shuffle=not args.eval and args.cache_latents,) # we shuffle only once if cache latents
    dataloader = DataLoader(dataset,
                        shuffle=not args.eval and not args.cache_latents, # since the shuffle is disabled if cache_latents is True, the training log will be a little bit different
                        batch_size=args.batch_size,
                        pin_memory=True,
                        num_workers=args.num_workers,)
                        # worker_init_fn=_seed_worker) # set the seed for reproducibility
    return dataloader

def get_model(args, ckpt=None):
    if args.model_type == 'functa':
        model = meta_modules.LatentModulatedSiren(in_channels=args.in_channels,
                                                out_channels=args.out_channels,
                                                width=args.width,
                                                depth=args.depth,
                                                latent_dim=args.hidden_features,
                                                latent_vector_type='instance',
                                                use_meta_sgd=args.use_meta_sgd,
                                                w0=args.w0,
                                                ffm_map_scale=args.ffm_map_scale,
                                                ffm_map_size=args.ffm_map_size,
                                                pos_emb = args.pos_emb,
                                                batch_norm_init=args.norm_latents)
    elif args.model_type == 'mnif':
        model = meta_modules.LinearMixtureINR(
                                            width=args.width,
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
                                            ffm_map_scale=args.ffm_map_scale,
                                            ffm_map_size=args.ffm_map_size,
                                            pos_emb = args.pos_emb,
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
                     std_latent=args.std_latent,
                     norm_latents=args.norm_latents,
                     ffm_map_scale=args.ffm_map_scale,
                     ffm_map_size=args.ffm_map_size,
                     pos_emb = args.pos_emb,
                     ).cuda()
    else:
        raise NotImplementedError
    model.train()
    keep_params = dict()
    with torch.no_grad():
        for name, param in model.get_named_parameters():
            keep_params[name] = param.clone()
    meta_grad_init = [0 for _ in model.get_parameters()]
    if ckpt is not None:
        model.load_state_dict(ckpt['inr_model'])
    model = model.cuda()

    if args.vae is not None:
        if args.vae == 'hierarchical_vae':
            if args.model_type == 'functa' or args.model_type == 'mnif':
                latent_dim = args.hidden_features
            elif args.model_type == 'inr_loe':
                latent_dim = args.latent_size*len(args.num_exps)
            vae_model = vb.HVAEBackbone(input_channel = args.vae_input_channel, 
                                    latent_channel = args.vae_latent_channel,  
                                    layers = args.vae_layers, 
                                    sampler_dim = args.vae_sampler_dim, 
                                    prior_scale = args.vae_prior_scale, 
                                    num_heads = args.vae_num_heads,
                                    dim_head = args.vae_dim_head,
                                    dropout = args.vae_dropout,
                                    sample_decoder = args.vae_sample_decoder,
                                    latent_dim=latent_dim)
        elif args.vae == 'layer_vae':
            if args.model_type == 'functa' or args.model_type == 'mnif':
                assert args.hidden_features % args.depth == 0, 'The hidden features must be divisible by the depth, so as for vae to work.'
                latent_dim = args.hidden_features//args.depth
                attn_layers = args.depth*[args.vae_attn_depth] # we do not want the prior features
            elif args.model_type == 'inr_loe':
                latent_dim = args.latent_size
                attn_layers = len(args.num_exps)*[args.vae_attn_depth]
            vae_model = vb.LayerVAE(input_channel = args.vae_input_channel, 
                                    latent_channel = args.vae_latent_channel,  
                                    layers = attn_layers, 
                                    sampler_dim = args.vae_sampler_dim, 
                                    prior_scale = args.vae_prior_scale, 
                                    num_heads = args.vae_num_heads,
                                    dim_head = args.vae_dim_head,
                                    dropout = args.vae_dropout,
                                    sample_decoder = args.vae_sample_decoder,
                                    latent_dim=latent_dim,)
        else:
            raise NotImplementedError
        vae_model.train()
        if ckpt is not None:
            vae_model.load_state_dict(ckpt['vae_model'])
        vae_model = vae_model.cuda()
    else:
        vae_model = None
    
    return model, vae_model, keep_params, meta_grad_init

def make_training_params(args, logger, model, vae_model, ckpt=None):
    params = sum(p.numel() for p in model.get_parameters())
    logger.info("Total number of parameters is: {}".format(params))
    logger.info("Model size is: {:.2f} MB".format(params * 4 / 1024**2)) 
    param_group = [{'params': model.get_parameters(), 'lr': args.lr_outer}]
    inr_optim = torch.optim.AdamW(param_group, lr=args.lr_outer, weight_decay=0)
    inr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(inr_optim, T_max=args.epochs, eta_min=1e-7)
    if ckpt is not None:
        inr_optim.load_state_dict(ckpt['inr_optim'])
        inr_scheduler.load_state_dict(ckpt['inr_scheduler'])
    if args.vae is not None:
        vae_optim = torch.optim.AdamW(vae_model.parameters(), lr=args.vae_lr, weight_decay=0, betas=(0.9, 0.999))
        vae_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(vae_optim, T_max=args.epochs, eta_min=1e-7)
        if ckpt is not None:
            vae_optim.load_state_dict(ckpt['vae_optim'])
            vae_scheduler.load_state_dict(ckpt['vae_scheduler'])
    else:
        vae_optim = None
        vae_scheduler = None
    return inr_optim, inr_scheduler, vae_optim, vae_scheduler

def latents_step(args, model, model_input, gt, context_params, meta_sgd_inner, cache_path, step):
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
    '''
    for different inr models and different vae models, the shape of latent is different. Please refer to the paper for details.
    '''            
    if args.vae is not None:
        # get gt latents for vae
        vae_mode, latents, latents_input, b, nl, ne = get_vae_in(args, context_params)
        
        z_dist, kl_all, kl_diag, log_q, log_p = vae_model(latents_input)
        if args.vae_sample_decoder:
            z,_ = z_dist.sample()
        else:
            z = z_dist
        vae_latents = get_vae_out(vae_mode, z, b, nl, ne)

        # get kld loss
        kl_all = torch.stack(kl_all)
        kl_coeff_i, kl_vals = kl_per_group(kl_all)
        total_kl = torch.sum(kl_coeff_i)

        kl_coeff_i = kl_coeff_i / (1.0 * total_kl)
        kl_coeff_i = kl_coeff_i / torch.mean(kl_coeff_i, dim=0, keepdim=True)
        kld_loss = torch.mean(kl_all * (kl_coeff_i.detach()))

        # get reconstruction loss
        if args.vae_sample_decoder:
            logp = z_dist.log_p(latents)
            recon_loss = -torch.sum(logp, dim=(1,2)) if len(logp.shape) == 3 else -torch.sum(logp, dim=1)
            recon_loss = recon_loss.mean()
        else:
            recon_loss = nn.functional.mse_loss(vae_latents, latents,reduction='sum')
        
        # do not calculate the gradient for the model
        with torch.no_grad():
            model_output = model(model_input, vae_latents)
        mse_loss = image_mse(None, model_output, gt)['img_loss']
        
        # get kld coefficient
        if epoch%args.annealing_every_n_epochs<args.warmup_epochs:
            kl_ratio = args.kl_r_max*(step+epoch%args.warmup_epochs*l_epoch)/(args.warmup_epochs*l_epoch)                
        elif epoch%args.annealing_every_n_epochs < args.annealing_every_n_epochs:
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
    ckpt = set_resume(args) # it has to be here, since the random state is also resumed and has to be at very beginning
    logger, log_dir, ckpt_path, img_path, tm_str = make_log(args)
    cache_path = make_cache(args, log_dir)

    '''training configuration steps'''
    make_wandb(args, tm_str)
    train_loader = get_data(args)
    model, vae_model, keep_params, meta_grad_init = get_model(args, ckpt)
    logger.info(model) if args.resume_from is None else logger.info('Resume training from {}'.format(args.resume_from))
    inr_optim, inr_scheduler, vae_optim, vae_scheduler = make_training_params(args, logger, model, vae_model)

    global_steps = 0 if args.resume_from is None else ckpt['global_steps']
    start_epoch = 0 if args.resume_from is None else ckpt['last_epoch']+1
    '''
    Not sure how do this affect the training process. Just follow the original mnif code.
    '''
    if not args.eval:
        with torch.no_grad():
            for name, param in model.get_named_parameters():
                param = keep_params[name].clone()

    if do_intra(args):
        intra_flag = False
    for epoch in range(start_epoch, args.epochs):
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
            We use the cached latents if the cache_latents is True. In the first epoch, latents are initialized as randn.
            Note: if the cache_latents is True, the training data cannot be shuffled.
            '''
            if args.cache_latents and os.path.exists(os.path.join(cache_path, f'd{step}.pt')):
                rand_params = torch.load(os.path.join(cache_path, f'd{step}.pt'), map_location='cpu').cuda()
                context_params = rand_params.detach()
                context_params.requires_grad_()
            else:
                context_params = model.get_context_params(batch_size, args.eval)
            
            if do_intra(args) and intra_flag:
                with torch.no_grad():
                    vae_mode, lts, vin, b, nl, ne = get_vae_in(args, context_params)
                    z = vae_model(vin)[0]
                    vae_ctx_gen = get_vae_out(vae_mode, z, b, nl, ne)
                context_params.data = vae_ctx_gen.data
                intra_flag = False
            
            meta_sgd_inner = model.meta_sgd_lrs() if args.use_meta_sgd else None           
            context_params = latents_step(args, model, model_input, gt, context_params, meta_sgd_inner, cache_path, step)      
            model_output, losses = model_update_step(model, model_input, context_params, inr_optim, meta_grad, gt)
            vae_mse, recon_loss, kld_loss, vae_out = vae_step(args, model, vae_model, vae_optim, context_params, epoch, step, len(train_loader), model_input, gt, vae_loss_avg) 
            if do_intra(args):
                intra_flag = vae_mse < losses['img_loss'].item()
            '''
            We cache the latents after it is updated.
            '''
            if args.cache_latents:
                torch.save(context_params.detach().cpu(), os.path.join(cache_path, f'd{step}.pt'))  
            
            # log step       
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
                plot_dict = {
                    'Inr_out': im_show.detach().cpu().numpy(),
                }
                if args.vae is not None:
                    vae_ind = vae_out['model_out'][0]>=0
                    vae_show = model_input['coords'][0][vae_ind.squeeze()]
                    plot_dict.update(
                        {'VAE_out': vae_show.detach().cpu().numpy()}
                    )
                plot_single_pcd(plot_dict, '{}/point_cloud_e{}s{}.png'.format(img_path,epoch, step))
            vis_vae(args, vae_model, context_params, epoch, step, log_dir, global_steps)
            global_steps += 1
            
        inr_scheduler.step()
        if args.vae is not None:
            vae_scheduler.step()
        if epoch % args.ckpt_every_n_epochs == 0:
            ckpt_dict = {
                'args': args,
                'last_epoch': epoch,
                'global_steps': global_steps,
                'inr_model': model.state_dict(),
                'inr_optim': inr_optim.state_dict(),
                'inr_scheduler': inr_scheduler.state_dict(),               
                'vae_model': vae_model.state_dict() if args.vae is not None else None,
                'vae_optim': vae_optim.state_dict() if args.vae is not None else None,
                'vae_scheduler': vae_scheduler.state_dict() if args.vae is not None else None,
                # 'random_states': get_random_states(),
            }
            torch.save(ckpt_dict, os.path.join(ckpt_path, 'ckpt_{}.pt'.format(epoch)))
    logger.info('Training finished')
    if args.wandb:
        wandb.finish()
                


def main():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    train(args)

main()
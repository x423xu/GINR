'''
backbone options:
    - functa
    - mnif
    - mod
latent updation options:
    - gaussian noise
    - optimizable
vae options:
    - nvae
    - lnvae
'''
import sys,os 
import math
import torch
import torch.nn as nn
import wandb
import numpy as np
import random

import torch.distributed as dist
from torch.utils.data import DataLoader
import multiprocessing
from multiprocessing import Process
import pytorch_lightning as pl


from configs import args
from model_zoo.functa import Siren_decoder
from dataset_zoo import ShapeNetGEM
from utils import plot_single_pcd, make_log, compute_psnr



if args.wandb:
    # wandb.init(project='ginr', entity='xxy', config=args)
    # wandb.config.update(args)
    wandb.init(project='ginr', entity='xxy', dir=args.log_dir, config = args)

def cleanup():
    dist.destroy_process_group()
def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = '6061'
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(args)
    cleanup()

def get_data(args):
    data_meta = {}
    if args.dataset == 'shapenet':
        sampling = None
        dataset = ShapeNetGEM(split=args.split,
                        sampling=sampling,
                        random_scale=args.random_scale,
                        dataset_root=args.dataset_root)
        in_channels, out_channels, pred_type = 3, 1, 'voxel'
        
    else:
        raise NotImplementedError
    data_meta.update(
        {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'pred_type': pred_type
        }   
    )
    dataloader = DataLoader(dataset,
                            shuffle=not args.eval,
                            batch_size=args.batch_size,
                            pin_memory=True,
                            num_workers=args.num_workers)
    return dataloader, data_meta



def train(args):
    current_rank = dist.get_rank()
    print('spawned process with rank {} in {} training' .format(current_rank, 'ddp' if args.ddp else 'normal'))
    '''
    create log directory and files:
        - log.txt
        - checkpoints
        - imgs
    '''
    if current_rank == 0:
        logger, log_dir, ckpt_path, img_path, tm_str = make_log(args)
        if args.wandb:
            wandb.run.name = tm_str
        if args.cache_latents:
            if os.path.exists(os.path.join(log_dir, 'latents')):
                os.system('rm -r {}'.format(os.path.join(log_dir, 'latents')))
            os.makedirs(os.path.join(log_dir, 'latents'), exist_ok=True)

    '''Init dataloader'''
    train_loader, data_meta = get_data(args)

    '''Init model'''
    model = Siren_decoder(args)
    model = model.cuda()
    model.train()
    # make sure all ranks have the same model
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    inner_optimizer = torch.optim.AdamW(model.get_parameters(), lr=0.0001, weight_decay=0, betas=(0.9, 0.999))

    global_step = 0
    for e in range(args.epochs):
        for data_id, (model_input_batch, gt_batch) in enumerate(train_loader):
            model_input, gt = model_input_batch, gt_batch
            model_input = {key: value.cuda() for key, value in model_input.items()}
            gt = {key: value.cuda() for key, value in gt.items()}
            global_step += 1

            meta_grad = [0 for _ in model.get_parameters()]

            '''
            only initialize latents at rank 0, and broadcast to other ranks
            '''
            siren_latent_in = torch.zeros(gt['img'].shape[0], 1024).cuda()
            if current_rank == 0:
                if args.cache_latents:
                    if not os.path.exists(os.path.join(log_dir, 'latents', 'd_{}.pt'.format(data_id))):
                        siren_latent_in = 1e-4*torch.randn(gt['img'].shape[0], 1024).cuda()
                    else:
                        siren_latent_in = torch.load(os.path.join(log_dir, 'latents', 'd_{}.pt'.format(data_id)), map_location='cpu')
                        siren_latent_in = siren_latent_in.cuda()
                else:
                    siren_latent_in = 1e-4*torch.randn(gt['img'].shape[0], 1024).cuda()
            dist.broadcast(siren_latent_in, src=0)
            
            '''
            inner optimization step: updates for 3
            '''
            # s1. calculate grad for latent but not for siren
            siren_latent_in.requires_grad = True
            for p in model.get_parameters():
                p.requires_grad = False
            # s2. zero grad
            inner_optimizer.zero_grad()
            if siren_latent_in.grad is not None:
                siren_latent_in.grad.zero_()
            # s3. calculate grad
            for _ in range(3):
                pred, _ = model(coords = model_input['coords'], z = siren_latent_in)
                # pred = nn.functional.sigmoid(pred)
                outer_loss = (pred-gt['img']).pow(2).mean(dim=(1,2)).mean()
                latent_gradients = torch.autograd.grad(outer_loss, siren_latent_in, create_graph=True)[0]
                # gather gradients from all ranks
                dist.all_reduce(latent_gradients, op=dist.ReduceOp.SUM)
                siren_latent_in = siren_latent_in - 0.5 * latent_gradients.detach()/args.global_size

            '''
            outer optimization step: updates for 1
            '''
            inner_optimizer.zero_grad()
            for p in model.get_parameters():
                p.requires_grad = True
            pred, _ = model(coords = model_input['coords'], z = siren_latent_in)
            outer_loss = (pred-gt['img']).pow(2).mean(dim=(1,2)).mean()
            siren_grad = torch.autograd.grad(outer_loss, model.get_parameters())
            nn.utils.clip_grad_norm_(model.get_parameters(), max_norm = 5.0, error_if_nonfinite=True)
            # gather gradients     
            for g in range(len(siren_grad)):
                dist.all_reduce(siren_grad[g], op=dist.ReduceOp.SUM)
                meta_grad[g] += siren_grad[g].detach()
            for n,p in enumerate(model.get_parameters()):
                p.grad = meta_grad[n]
            
            inner_optimizer.step()
            siren_latent_in = siren_latent_in.detach()
            if current_rank == 0:
                # get psnr

                psnr = compute_psnr(pred.detach().cpu() * 0.5 + 0.5, gt['img'].cpu() * 0.5 + 0.5) # rescale from [-1, 1] to [0, 1]
                            
                # log
                if data_id % args.log_every_n_steps == 0:
                    logger.info('***** e:{} s:{}/{} outer mse_loss {:.6f}, psnr {:.6f}'.format(e, data_id, len(train_loader),outer_loss.item(), psnr))
                    if args.wandb:
                        wandb.log({'outer_loss': outer_loss.cpu().item(), 'psnr': psnr,'global_step': global_step})
                # plot
                if data_id % args.save_every_n_steps == 0:
                    ind = pred[0]>=0
                    im_show = model_input['coords'][0][ind.squeeze()]
                    plot_single_pcd(im_show.detach().cpu().numpy(), '{}/outer_e{}s{}.png'.format(img_path,e, data_id))
                if args.cache_latents:
                    torch.save(siren_latent_in.data,os.path.join(log_dir, 'latents', 'd_{}.pt'.format(data_id)))
        if e % args.ckpt_every_n_epochs == 0 and current_rank == 0:
            torch.save(model.state_dict(), os.path.join(ckpt_path, 'model_{}.pt'.format(e)))
   

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    '''set seed'''
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    '''
    We do ddp training for multi gpu setting, otherwise normal training.
    '''
    if args.ddp:
        size = args.num_process_per_node
        assert size >1, 'ddp training requires more than 1 process per node'
        processes = []
        try:
            for rank in range(size):
                args.local_rank = rank
                global_rank = rank + args.node_rank * args.num_process_per_node
                global_size = args.num_proc_node * args.num_process_per_node
                args.global_rank = global_rank
                args.global_size = global_size
                print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
                p = Process(target=init_processes, args=(global_rank, global_size, train, args))
                p.start()
                processes.append(p)
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
            for p in processes:
                p.terminate()
        except Exception as e:
            print(e)
            for p in processes:
                p.terminate()
        finally:
            for p in processes:
                p.join()
    else:
        args.global_rank = 0
        args.global_size = 1
        init_processes(args.global_rank, args.global_size, train, args)
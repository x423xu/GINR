import argparse
import yaml
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--config', type=str, default='cfgs/train_functa_shapenet.yml')

# data params
parser.add_argument('--dataset', type=str, default='shapenet')
parser.add_argument('--dataset_root', type=str, default='/home/xxy/Documents/data/')
parser.add_argument('--split', type=str, default='train')
parser.add_argument('--random_scale', action='store_true', default=False)
parser.add_argument('--num_workers', type=int, default=4)

# training params
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--eval', action='store_true', default=False)
parser.add_argument('--lr_outer', type=float, default=0.0001, help='learning rate')
parser.add_argument('--inner_steps', type=int, default=3, help='number of inner steps for each coords')
parser.add_argument('--lr_inner', type=float, default=1e-4, help='learning rate for inner loop')
parser.add_argument('--cache_latents', action='store_true', default=False, help = 'training with cached latents')
parser.add_argument('--resume_from', type=str, default=None, help='The path where the resumed training starts from')

# ddp training params
parser.add_argument('--ddp', action='store_true', default=False)
parser.add_argument('--num_process_per_node', type=int, default=1, 
                    help='number of processes per node, 1 for normal training, 4 for ddp training')
parser.add_argument('--local_rank', type=int, default=0, help='rank of process inside a node')
parser.add_argument('--master_address', type=str, default='127.0.0.1',
                    help='address for master')
parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
# parser.add_argument('--global_size', type=int, default=1,help = 'The number of processes in total.')

# log params
parser.add_argument('--log_dir', type=str, default='logs')
parser.add_argument('--wandb', action='store_true', default=False, help='use wandb for logging')
parser.add_argument('--log_every_n_steps', type=int, default=10)
parser.add_argument('--ckpt_every_n_epochs', type=int, default=1)
parser.add_argument('--save_every_n_steps', type=int, default=100)

# vae params
parser.add_argument('--vae', type=str, default=None, choices=[None, 'simple_vae', 'hierarchical_vae', 'layer_vae'], help = 'Which vae to use')
parser.add_argument('--warmup_epochs', type=int, default=1, help='warmup epochs for annealing kl loss')
parser.add_argument('--annealing_every_n_epochs', type=int, default=2, help='annealing kl loss every n epochs')
parser.add_argument('--kl_r_max', type=float, default=1e-5, help='maximum kl loss weight')
parser.add_argument('--vae_lr', type=float, default=1e-4, help='learning rate for vae')
parser.add_argument('--vae_loss_type', type=str, default='10*recon+1*kld+1*mse', help='loss type for vae')
args = parser.parse_args()

'''
Read the config file, merge the config into the args. !!! DO NOT OVERIDE THE DEFAULT ARGS !!!
'''
with open(args.config, 'r') as file:
    config =  yaml.safe_load(file)
for key, value in config.items():
    if getattr(args, key, None) == parser.get_default(key):
        setattr(args, key, value)

'''
If vae is not None, merge the vae configs into the args. !!! DO NOT OVERIDE THE DEFAULT ARGS !!!
'''
if args.vae is not None:
    with open(f'cfgs/{args.vae}.yml', 'r') as file:
        vae_config =  yaml.safe_load(file)
    for key, value in vae_config.items():
        if getattr(args, key, None) == parser.get_default(key):
            setattr(args, key, value)
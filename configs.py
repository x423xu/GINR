import argparse
import yaml
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--config', type=str, default='cfgs/train_functa_shapenet.yml')

# data params
parser.add_argument('--dataset_root', type=str, default='/home/xxy/Documents/data/')
parser.add_argument('--split', type=str, default='train')
parser.add_argument('--random_scale', action='store_true', default=False)
parser.add_argument('--num_workers', type=int, default=4)

# training params
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--eval', action='store_true', default=False)

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

args = parser.parse_args()

with open(args.config, 'r') as file:
    config =  yaml.safe_load(file)
for key, value in config.items():
    if getattr(args, key, None) == parser.get_default(key):
        setattr(args, key, value)
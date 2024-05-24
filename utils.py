import os
import torch
import math
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d %b %Y %H:%M:%S')
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import wandb
import random
matplotlib.use('Agg')

def get_pts(pcd):
    points = np.asarray(pcd.points)
    X = []
    Y = []
    Z = []
    for pt in range(points.shape[0]):
        X.append(points[pt][0])
        Y.append(points[pt][1])
        Z.append(points[pt][2])
    return np.asarray(X), np.asarray(Y), np.asarray(Z)

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.5*max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    ax.set_axis_off()

def plot_single_pcd(plot_dict,save_path, wandb = None):
    num_imgs = len(plot_dict.keys())
    fig, axs = plt.subplots(1, num_imgs, figsize=(num_imgs*3, 3),subplot_kw={'projection': '3d'})
    fig.set_facecolor('lightgray') # set background color
    for i, (title, points) in enumerate(plot_dict.items()):
        if num_imgs>1:
            ax = axs[i]
        else:
            ax = axs
        ax.set_aspect('equal')
        if points.shape[0]>4000:
            skip = points.shape[0]//4000
            points = points[::skip]
        ax.set_aspect('equal')
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        rotation_matrix = np.asarray([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        pcd = pcd.transform(rotation_matrix)
        X, Y, Z = get_pts(pcd)
        t = Z
        ax.scatter(X, Y, Z, c=t, cmap='jet', marker='o', s=20, depthshade=True, edgecolors='k')
        ax.grid(False)
        ax.set_facecolor('lightgray') # set background color
        set_axes_equal(ax)
        ax.set_title(title)
    fig.savefig(save_path, format='png', dpi=600)
    if wandb is not None:
        wandb.log({save_path.split('/')[-1].split('_')[0]: wandb.Image(fig)})
    plt.close()

def compute_psnr(signal, gt):
    mse = max(float(torch.mean((signal-gt)**2)), 1e-8)
    psnr = float(-10 * math.log10(mse))
    return psnr  

def get_timestr():
    import pytz
    from datetime import datetime
    toronto_tz = pytz.timezone('America/Toronto')
    utc_now = datetime.utcnow()
    toronto_now = utc_now.astimezone(toronto_tz)
    timestr = toronto_now.strftime("%Y%m%d-%H%M%S")
    return timestr

def make_log(args):
    if args.resume_from is not None:
        assert os.path.exists(args.resume_from), 'The resume_from path does not exist.'
        log_dir = args.resume_from
        tm_str = log_dir.split('/')[-1]
    else:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        # make path
        tm_str = get_timestr()
        tm_str = tm_str + '+' + args.model_type.upper()+ '+' +args.dataset.upper()
        if args.vae is not None:
            tm_str = tm_str + '+' + args.vae.upper()
        if args.model_type == 'functa':
            tm_str = tm_str + '+' + 'D{}W{}'.format(args.depth, args.width)
        elif args.model_type == 'mnif':
            tm_str = tm_str + '+' + 'D{}W{}E{}'.format(args.depth, args.width,args.k_mixtures)
        elif args.model_type == 'inr_loe':
            tm_str = tm_str + '+' + 'D{}W{}E{}'.format(len(args.num_exps), args.hidden_dim, args.num_exps[0])
        log_dir = os.path.join(args.log_dir, tm_str)
    ckpt_path = os.path.join(log_dir, 'checkpoints')
    img_path = os.path.join(log_dir, 'imgs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    # set up logging
    logger = logging.getLogger()
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(os.path.join(log_dir,'log.txt'), mode = 'a')
    c_handler.setLevel(logging.WARNING)
    f_handler.setLevel(logging.DEBUG)

    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s', datefmt='%d %b %Y %H:%M:%S')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%d %b %Y %H:%M:%S')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    logger.info(args)
    return logger, log_dir, ckpt_path, img_path, tm_str

def make_cache(args, log_dir):
    if args.cache_latents:
        if args.resume_from is not None:
            assert os.listdir(log_dir), f'The resumed path {log_dir} has no cached latents.'
        cache_path = os.path.join(log_dir, 'cache')
        os.makedirs(cache_path, exist_ok=True)
        return cache_path
    else:
        return None
    
def make_wandb(args, tm_str):
    if args.wandb:
        
        # set wandb tags and descriptions
        tags = []
        tags.append(args.model_type)
        if args.tag is not None:
            tags.append(args.tag)
        descriptions = f'This is a training record of **{args.model_type}**. '
        if args.vae is not None:
            descriptions += f'The vae **{args.vae}** is enabled. '
        if args.cache_latents:
            descriptions += 'The latents are **cached**. '
        if args.resume_from is not None:
            wandb.init(entity='xxy', project='ginr', dir=args.log_dir, config = args, tags=tags, notes=descriptions, id = tm_str, name = tm_str, resume='must')
        else:
            wandb.init(entity='xxy', project='ginr', dir=args.log_dir, config = args, tags=tags, notes=descriptions, id = tm_str, name = tm_str)

def image_mse(mask, model_output, gt):
    if mask is None:
        return {'img_loss': ((model_output['model_out'] - gt['img']) ** 2).mean()}
    else:
        return {'img_loss': (mask * (model_output['model_out'] - gt['img']) ** 2).mean()}
    
def kl_per_group(kl_all):
        kl_vals = torch.mean(kl_all, dim=1)
        kl_coeff_i = torch.abs(kl_all)
        kl_coeff_i = torch.mean(kl_coeff_i, dim=1, keepdim=True) + 0.01
        return kl_coeff_i, kl_vals

class AverageValueMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_resume(args):
    if args.resume_from is not None:
        model_name = os.listdir(os.path.join(args.resume_from, 'checkpoints'))
        latest_model = sorted(model_name, key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)[0]
        ckpt = torch.load(os.path.join(args.resume_from,'checkpoints',latest_model), map_location='cpu')
        # load_random_states(ckpt['random_states'])
    else:
        ckpt = None
    '''
    check if some key args are legal with the resumed ckpt
    '''
    if ckpt is not None:
        # check if the model type is the same
        assert args.model_type == ckpt['args'].model_type, 'The model type is different from the resumed model.'
        # check if the dataset is the same
        assert args.dataset == ckpt['args'].dataset, 'The dataset is different from the resumed model.'
        # check if the vae is the same
        assert args.vae == ckpt['args'].vae, 'The vae is different from the resumed model.'
        # check if the cache_latents is the same
        assert args.cache_latents == ckpt['args'].cache_latents, 'The cache_latents is different from the resumed model.'

    return ckpt

def get_random_states():
    state_dict = {}
    state_dict['py_random_state'] = random.getstate()
    state_dict['np_random_state'] = np.random.get_state()
    state_dict['torch_random_state'] = torch.get_rng_state()
    
    if torch.cuda.is_available():
        state_dict['torch_cuda_random_state'] = torch.cuda.get_rng_state()
    
    return state_dict

def load_random_states(state_dict):
    random.setstate(state_dict['py_random_state'])
    np.random.set_state(state_dict['np_random_state'])
    torch.set_rng_state(state_dict['torch_random_state'])
    
    if torch.cuda.is_available() and 'torch_cuda_random_state' in state_dict:
        torch.cuda.set_rng_state(state_dict['torch_cuda_random_state'])

def do_intra(args):
    if args.intra_latent and args.vae is not None:
        return True
    else:
        return False

def get_vae_in(args, context_params):
        mode_mapping= {
            'functa': '0',
            'mnif': '1',
            'inr_loe': '2',
            'hierarchical_vae': '0',
            'layer_vae': '1',     
        }
        b, nl, ne = None, None, None
        vae_mode = mode_mapping[args.model_type]+mode_mapping[args.vae]
        if vae_mode == '00' or vae_mode =='10':
            latents = context_params.data.clone()
            latents_input = latents.unsqueeze(1)
        elif vae_mode == '01' or vae_mode == '11':
            latents = context_params.data.clone()
            b, _ = latents.size()
            latents_input = latents.contiguous().view(b, args.depth, args.hidden_features//args.depth)
        elif vae_mode == '20':
            latents = context_params.data.clone()
            b, nl, ne = latents.size()
            latents_input = latents.contiguous().view(b, 1, -1)
        elif vae_mode == '21':
            latents = context_params.data.clone()
            b, nl, ne = latents.size()
            latents_input = latents
        else:
            raise NotImplementedError
        if args.vae_norm_in_out:
            dims_to_reduce = tuple(range(1, latents_input.dim()))
            lmu = latents_input.mean(dim=dims_to_reduce, keepdim=True)
            lstd = latents_input.std(dim=dims_to_reduce, keepdim=True)+1e-7
            latents_input = (latents_input - lmu) / lstd
            in_dict = {
                'lin': latents_input,
                'lmu': lmu,
                'lstd': lstd
            }
        else:
            in_dict = latents_input
        return vae_mode, latents, in_dict, b, nl, ne

def get_vae_out(vae_mode, z, b, nl, ne):
    vae_latents = z.squeeze(-1)
    if vae_mode == '01' or vae_mode=='11':
        vae_latents = vae_latents.contiguous().view(b, -1)
    elif vae_mode == '20':
        vae_latents = vae_latents.contiguous().view(b, nl, ne)
    return vae_latents

@torch.no_grad()
def vis_vae(args, vae_model, context_params, epoch, step, log_dir, global_steps):
    if args.vis_vae_every_n_steps and args.vae is not None:
        vae_path = os.path.join(log_dir, 'vis_vae')
        if not os.path.exists(vae_path):
            os.makedirs(vae_path)     
        if step % args.vis_vae_every_n_steps == 0:
            vae_mode, latents, latents_input, _,_,_ = get_vae_in(args, context_params)
            if args.vae_norm_in_out:
                latents_input = latents_input['lin']
            out_dist, kl_all, all_q, all_p, all_log_q, all_log_p = vae_model(latents_input, return_meta = True)
            if args.vae_sample_decoder:
                z,_ = out_dist.sample()
            else:
                z = out_dist
            lin = latents_input.squeeze().cpu().numpy()
            lout = z.squeeze().cpu().numpy()
            num_axs = lin.shape[1]
            fig, axs = plt.subplots(1, num_axs, figsize=(num_axs*3, 3))
            for i in range(num_axs):
                if num_axs == 1:
                    ax = axs
                else:
                    ax = axs[i]
                ax.hist(lin[:,i].reshape(-1), bins=100, alpha=0.5, label=f'latent_in_l{i+1}', color='r')
                ax.hist(lout[:,i].reshape(-1), bins=100, alpha=0.5, label=f'latent_out_l{i+1}', color='b')
                ax.legend()    
            fig.savefig(os.path.join(vae_path, f'latent_hist_{epoch}_{step}.png'))
            if args.wandb:
                wandb.log({f'latent_his': wandb.Image(fig)},step = global_steps)
            plt.close(fig)
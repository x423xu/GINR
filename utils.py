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

def plot_single_pcd(points, save_path, wandb = None):
    if points.shape[0]>4000:
        skip = points.shape[0]//4000
        points = points[::skip]
    fig = plt.figure(figsize=(3, 3))
    fig.set_facecolor('lightgray') # set background color
    ax = fig.add_subplot(111, projection='3d')
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
    plt.axis('off')
    plt.savefig(save_path, format='png', dpi=600)
    if wandb is not None:
        wandb.log({save_path.split('/')[-1].split('_')[0]: wandb.Image(save_path)})
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
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    # clean empty dir in log_dir
    for root, dirs, files in os.walk(args.log_dir):
        if not dirs and not files:
            os.rmdir(root)
    # make path
    tm_str = get_timestr()
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
    f_handler = logging.FileHandler(os.path.join(log_dir,'log.txt'))
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
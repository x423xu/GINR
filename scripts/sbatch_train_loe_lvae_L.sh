#!/bin/bash
#SBATCH --account=rrg-zhouwang
#SBATCH --time=1-8:00:00
#SBATCH --mail-user=x423xu@uwaterloo.ca
#SBATCH --mail-type=ALL
#SBATCH --job-name=GINR
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24000M
#SBATCH --output=/project/def-zhouwang/xiaoyu/scripts/output/%x-%A-%a.out

module load StdEnv/2020
module load gcc opencv python/3.9 scipy-stack cuda/11.7 openmpi
cd $SLURM_TMPDIR
virtualenv --no-download inr
source inr/bin/activate

pip3 install --no-index torch==1.13.0
pip3 install --no-index torchvision==0.14.0
pip3 install --no-index matplotlib
pip3 install --no-index imageio
pip3 install --no-index imageio-ffmpeg
pip3 install --no-index numpy
pip3 install --no-index scikit-image
pip3 install --no-index jupyter
pip3 install --no-index glfw
pip3 install --no-index open3d
pip3 install --no-index torch_geometric
pip3 install --no-index PyYAML
pip3 install --no-index wandb
pip3 install --no-index einops

cp -r /scratch/xiaoyu/code/sampyl $SLURM_TMPDIR
cd $SLURM_TMPDIR/sampyl
python setup.py install
cd /scratch/xiaoyu/code/GINR
wandb offline

python train_inr.py --config cfgs/train_loe_shapenet_L.yml --epochs 100 --batch_size 24 --save_every_n_steps 1000 --dataset_root /scratch/xiaoyu/data/ --wandb --log_dir logs --cache_latents --vae layer_vae
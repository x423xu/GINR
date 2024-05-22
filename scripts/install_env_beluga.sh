module load StdEnv/2020
module load gcc opencv python/3.9 scipy-stack cuda/11.7 openmpi
cd $SLURM_TMPDIR
virtualenv --no-download inr
source inr/bin/activate

pip install --no-index torch==1.13.0
pip install --no-index torchvision==0.14.0
pip install --no-index matplotlib
pip install --no-index imageio
pip install --no-index imageio-ffmpeg
pip install --no-index numpy
pip install --no-index scikit-image
pip install --no-index jupyter
pip install --no-index glfw
pip install --no-index open3d
pip install --no-index torch_geometric
pip install --no-index PyYAML
pip install --no-index wandb
pip install --no-index einops

cp /scratch/xiaoyu/code/sampyl $SLURM_TMPDIR
cd $SLURM_TMPDIR/sampyl
python setup.py install
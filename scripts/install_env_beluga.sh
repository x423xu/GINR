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

cd /scratch/xiaoyu/code/sampyl
python setup.py install
cd /scratch/xiaoyu/code/GINR
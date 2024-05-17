from torch.utils.data import Dataset
import torch
import os
import numpy as np
import random

# ShapeNet DataLoader is modified from the original code
# https://github.com/yilundu/gem/blob/2778026ac4508f44c7af160e4e157c6fb039f4ce/dataio.py#L718
class ShapeNetGEM(Dataset):
    def __init__(self, split='train', sampling=None, dataset_root='datasets', simple_output=False, random_scale=False):
        self.dataset_root = dataset_root
        self.sampling = sampling
        self.init_model_bool = False
        self.split = split
        self.simple_output = simple_output
        self.random_scale = random_scale
        self.init_model()
        self.data_type = 'voxel'

    def __len__(self):
        if self.split == "train":
            return 35019
            # return 5000
        else:
            return 8762

    def init_model(self):
        split = self.split
        points_path = os.path.join(self.dataset_root, 'shapenet', 'all_vox256_img', 'data_points_int_' + split + '.pth')
        values_path = os.path.join(self.dataset_root, 'shapenet', 'all_vox256_img', 'data_values_' + split + '.pth')

        self.data_points_int = torch.load(points_path).byte()
        self.data_values = torch.load(values_path).byte()
    
    def __getitem__(self, idx):
        points = (self.data_points_int[idx].float() + 1) / 128 - 1
        occs = self.data_values[idx].float() * 2 -1

        if self.sampling is not None:
            idcs = np.random.randint(0, len(points), size=self.sampling)
            points = points[idcs]
            occs = occs[idcs]

        if self.random_scale:
            points = random.uniform(0.75, 1.25) * points

        if self.simple_output:
            return occs

        else:
            in_dict = {'idx': idx, 'coords': points}
            gt_dict = {'img': occs}

            return in_dict, gt_dict

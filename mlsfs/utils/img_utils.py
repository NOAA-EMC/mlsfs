import logging
import glob
from types import new_class
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch import Tensor
import math
import torchvision.transforms.functional as TF

def reshape_fields(img, inp_or_tar, normalize=True):
    #Takes in np array of size (n_history+1, c, h, w) and returns torch tensor of size ((n_channels*(n_history+1), crop_size_x, crop_size_y)
    in_channels = np.shape(img)[0] #this will either be N_in_channels or N_out_channels

    #if len(np.shape(img)) ==3:
    #      img = np.expand_dims(img, 0)

    # n_history = np.shape(img)[0] - 1
    # img_shape_x = np.shape(img)[-2]
    # img_shape_y = np.shape(img)[-1]
    # channels = params.in_channels if inp_or_tar =='inp' else params.out_channels
    # if crop_size_x == None: #     crop_size_x = img_shape_x
    # if crop_size_y == None:
    #     crop_size_y = img_shape_y

    if normalize:
        #logging.info(f'Normalize the data')
        #means = np.load(params.global_means_path)[:in_channels]
        #stds = np.load(params.global_stds_path)[:in_channels]
        means = np.load('data/global_mean_1979-2016.npy')[0, :in_channels]
        #means_exp = np.expand_dims(means, axis=(0, 2, 3))
        stds = np.load('data/global_std_1979-2016.npy')[0, :in_channels]
        #stds_exp = np.expand_dims(stds, axis=(0, 2, 3))
        #if params.normalization == 'minmax':
        #  raise Exception("minmax not supported. Use zscore")
        #elif params.normalization == 'zscore':
        #img = means
        #img /= stds

    return torch.as_tensor((img - means) / stds)

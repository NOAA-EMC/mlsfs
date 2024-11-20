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

def reshape_fields(img, inp_or_tar, params, normalize=True, orog=None, lsm=None, lake=None):
    in_channels = np.shape(img)[0] #this will either be N_in_channels or N_out_channels
    img_shape_x = np.shape(img)[-2]
    img_shape_y = np.shape(img)[-1]

    #if len(np.shape(img)) ==3:
    #      img = np.expand_dims(img, 0)

    if normalize:
        means = np.load(params.global_means_path)[0, :in_channels]
        stds = np.load(params.global_stds_path)[0, :in_channels]
        if params.multi_step_training and inp_or_tar == 'tar':
            img -= np.expand_dims(means, axis=1)
            img /= np.expand_dims(stds, axis=1)
        else:
            img -= means
            img /= stds

    if params.multi_step_training and inp_or_tar == 'tar':
        img = np.swapaxes(img, 0, 1)
        img = np.reshape(img, (in_channels*params.nfutures, img_shape_x, img_shape_y))

    if params.orography and inp_or_tar == 'inp':
        img = np.concatenate((img, np.expand_dims(orog, axis=0)), axis=0)

    if params.lsmask and inp_or_tar == 'inp':
        img = np.concatenate((img, np.expand_dims(lsm, axis=0)), axis=0)

    if params.lakemask and inp_or_tar == 'inp':
        img = np.concatenate((img, np.expand_dims(lake, axis=0)), axis=0)

    return torch.as_tensor(img)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torchdiffeq._impl.conv import StaticODEfunc, DenseODEfunc


class SRTrunk(nn.Module):
    def __init__(self, n_filters, n_blocks, ODEBlock_=None):
        super(SRTrunk, self).__init__()
        self.ODEBlock = ODEBlock_
        self.odefunc = DenseODEfunc(dim=n_filters, growth=n_filters//2, nb=5, bias=True, normalization=False)

    def forward(self, x):
        return self.odefunc(x)


def lr_schedule(lr, epoch):
    optim_factor = 0
    if epoch > 250:
        optim_factor = 2
    elif epoch > 150:
        optim_factor = 1

    return lr / math.pow(10, (optim_factor))

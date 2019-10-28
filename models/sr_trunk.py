import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torchdiffeq._impl.conv import ODEfunc, DenseODEfunc


class SRTrunk(nn.Module):
    def __init__(self, n_filters, n_blocks, ODEBlock_=None):
        super(SRTrunk, self).__init__()
        # self.odefunc = ODEfunc(n_filters, n_blocks)
        # self.odefunc = DenseODEfunc(dim=n_filters, growth=n_filters//2, nb=n_blocks, bias=True, normalization=False)
        self.odefunc = ODEfunc(dim=n_filters, nb=n_blocks, normalization=False, time_dependent=False)
        self.ODEBlock = ODEBlock_(self.odefunc)

    def forward(self, x):
        return self.ODEBlock(x)

    @property
    def nfe(self):
        return self.ODEBlock.nfe

    @nfe.setter
    def nfe(self, value):
        self.ODEBlock.nfe = value


def lr_schedule(lr, epoch):
    optim_factor = 0
    if epoch > 250:
        optim_factor = 2
    elif epoch > 150:
        optim_factor = 1

    return lr / math.pow(10, (optim_factor))

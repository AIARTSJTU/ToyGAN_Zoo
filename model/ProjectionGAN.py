#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: Projection_GAN.py
# Created Date: Monday February 17th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Tuesday, 18th February 2020 1:04:09 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################


import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transf
import torchvision.models as models
import torchvision.utils as vutils
import torch.nn.utils.spectral_norm as spectral_norm
from model.CategoricalConditionalBatchNorm1d import CategoricalConditionalBatchNorm1d

import sys
import datetime
import time
from torch.nn import utils

def print_now(cmd, file=None):
    time_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if file is None:
        print('%s %s' % (time_now, cmd))
    else:
        print_str = '%s %s' % (time_now, cmd)
        print(print_str, file=file)
    sys.stdout.flush()

class GAN_G(nn.Module):
    def __init__(self, param):
        super(GAN_G, self).__init__()
        class_num        = param.n_class
        self.layer1      = torch.nn.Linear(param.z_size, param.G_h_size)
        self.layer1cbn   = CategoricalConditionalBatchNorm1d(class_num, param.G_h_size)
        self.relu        = torch.nn.ReLU()

        self.layer2      = torch.nn.Linear(param.G_h_size, param.G_h_size)
        self.layer2cbn   = CategoricalConditionalBatchNorm1d(class_num, param.G_h_size)

        self.layer3      = torch.nn.Linear(param.G_h_size, param.G_h_size)
        self.layer3cbn   = CategoricalConditionalBatchNorm1d(class_num, param.G_h_size)

        self.layer4      = torch.nn.Linear(param.G_h_size, param.G_h_size)
        self.layer4cbn   = CategoricalConditionalBatchNorm1d(class_num, param.G_h_size)

        self.fc          = torch.nn.Linear(param.G_h_size, 2)
    
    def forward(self, input, a):
        input = input.squeeze()
        output = self.layer1(input)
        output = self.layer1cbn(output,a)
        output = self.relu(output)

        output = self.layer2(output)
        output = self.layer2cbn(output,a)
        output = self.relu(output)

        output = self.layer3(output)
        output = self.layer3cbn(output,a)
        output = self.relu(output)

        output = self.layer4(output)
        output = self.layer4cbn(output,a)
        output = self.relu(output)

        output = self.fc(output)

        return output


class Maxout(nn.Module):
    def __init__(self, d_in, d_out, pool_size):
        super().__init__()
        self.d_in, self.d_out, self.pool_size = d_in, d_out, pool_size
        # self.lin = utils.spectral_norm(nn.Linear(d_in, d_out * pool_size))
        self.lin = nn.Linear(d_in, d_out * pool_size)

    def forward(self, inputs):
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.lin(inputs)
        m, i = out.view(*shape).max(max_dim)
        return m

class GAN_D(nn.Module):
    def __init__(self, param):
        super(GAN_D, self).__init__()
        model = []

        model.append(Maxout(2, param.D_h_size, 5))
        model.append(Maxout(param.D_h_size, param.D_h_size, 5))
        model.append(Maxout(param.D_h_size, param.D_h_size, 5))
        self.fc  = nn.Linear(param.D_h_size, param.num_outcomes)
        self.l_y = nn.Embedding(param.n_class, param.D_h_size)

        # self.fc  = utils.spectral_norm(nn.Linear(param.D_h_size, param.num_outcomes))
        # self.l_y = utils.spectral_norm(nn.Embedding(param.n_class, param.D_h_size))

        model = torch.nn.Sequential(*model)
        self.model = model
        self.param = param
    
    def forward(self, input, a):
        h = self.model(input)
        out = self.fc(h) + torch.sum(self.l_y(a)*h,dim=1,keepdim=True)
        return out



#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: learn_projection.py
# Created Date: Monday February 17th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Tuesday, 18th February 2020 2:36:57 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
import math
import sys
import datetime
import time

from collections import namedtuple

def print_now(cmd, file=None):
    time_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if file is None:
        print('%s %s' % (time_now, cmd))
    else:
        print_str = '%s %s' % (time_now, cmd)
        print(print_str, file=file)
    sys.stdout.flush()

def learnG_Realness(param, D, G, optimizerG, data_loader, z_sampler, label_sampler, label_sampler2):
    G.train()
    for p in D.parameters():
        p.requires_grad = False

    for t in range(param.G_updates):
        G.zero_grad()
        optimizerG.zero_grad()

        # gradients are accumulated through subiters
        for _ in range(param.effective_batch_size // param.batch_size):
            # fake images
            z_sampler.sample_()
            label_sampler.sample_()
            imgs_fake = G(z_sampler, label_sampler)
            feat_fake = D(imgs_fake, label_sampler)
            label_sampler2.sample_()
            g_out_fake_neg  = D(imgs_fake,label_sampler)
            # compute loss
            lossG = -feat_fake.mean() + 0.5*g_out_fake_neg.mean()
            lossG.backward()

        optimizerG.step()
    return lossG

def learnD_Realness(param, D, G, optimizerD, data_loader, z_sampler, label_sampler, label_sampler2):
    for p in D.parameters():
        p.requires_grad = True

    for t in range(param.D_updates):
        D.zero_grad()
        optimizerD.zero_grad()

        # gradients are accumulated through subiters
        for _ in range(param.effective_batch_size // param.batch_size):
            images,label = next(data_loader)
            # real images
            feat_real = D(images.cuda(),label.cuda().long())
            # fake images
            z_sampler.sample_()
            label_sampler.sample_()
            imgs_fake   = G(z_sampler, label_sampler)
            feat_fake   = D(imgs_fake.detach(), label_sampler)
            d_loss_real = torch.nn.ReLU()(1.0 - feat_real).mean()
            d_loss_fake = torch.nn.ReLU()(1.0 + feat_fake).mean()
            label_sampler2.sample_()
            d_out_real_neg  = D(images.cuda(),label_sampler2)
            d_out_real_neg  = torch.nn.ReLU()(d_out_real_neg)
            d_out_real_neg  = -0.5*torch.nn.ReLU()(1.0 - d_out_real_neg).mean()
            lossD           = d_loss_real + d_loss_fake + d_out_real_neg
            lossD.backward()

        optimizerD.step()

    return lossD
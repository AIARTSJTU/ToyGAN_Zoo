#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: train_toy_projection.py
# Created Date: Monday February 17th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Tuesday, 18th February 2020 4:52:57 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################


import os
import numpy as np
import random
import math
import gc
import pickle
import time
import sys
import datetime

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
from MixtureGaussianData import get_dataset

from training.learn_projection import learnD_Realness,learnG_Realness
import training.Sampler as Sampler
from training.loss import CategoricalLoss
from Config_from_yaml import get_config_from_yaml
import matplotlib.pyplot as plt
import shutil

def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def print_now(cmd, file=None):
	time_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
	if file is None:
		print('%s %s' % (time_now, cmd))
	else:
		print_str = '%s %s' % (time_now, cmd)
		print(print_str, file=file)
	sys.stdout.flush()

configFile = "./train_projection.yaml"

param = get_config_from_yaml(configFile)

start = time.time()

if param.load_ckpt is None:
    if param.gen_extra_images > 0 and not os.path.exists(f"{param.extra_folder}"):
        os.mkdir(f"{param.extra_folder}")
    print_now(param)

if param.cuda:
    import torch.backends.cudnn as cudnn
    cudnn.deterministic = True
    device = 'cuda'

random.seed(param.seed)
np.random.seed(param.seed)
torch.manual_seed(param.seed)
if param.cuda:
    torch.cuda.manual_seed_all(param.seed)



random_sample = get_dataset("./data/MixtureGaussianCXH.pk",param.batch_size,0)

from model.ProjectionGAN import GAN_G, GAN_D
G = GAN_G(param)
D = GAN_D(param)
print_now('Using feature size of {}'.format(param.num_outcomes))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    # elif classname.find('BatchNorm') != -1:
    #     m.weight.data.normal_(1.0, 0.02)
    #     m.bias.data.fill_(0)



runingZ,runingLabel = Sampler.prepare_z_c(param.batch_size, param.z_size, param.n_class)

print_now("Initialized weights")
G.apply(weights_init)
D.apply(weights_init)

# to cuda
G = G.to(device)
D = D.to(device)
optimizerD = torch.optim.Adam(D.parameters(), lr=param.lr_D, betas=(param.beta1, param.beta2), weight_decay=param.weight_decay, eps=float(param.adam_eps))
optimizerG = torch.optim.Adam(G.parameters(), lr=param.lr_G, betas=(param.beta1, param.beta2), weight_decay=param.weight_decay)
decayD = torch.optim.lr_scheduler.ExponentialLR(optimizerD, gamma=1-param.decay)
decayG = torch.optim.lr_scheduler.ExponentialLR(optimizerG, gamma=1-param.decay)

if param.load_ckpt != "None":
    checkpoint = torch.load(param.load_ckpt)
    current_set_images = checkpoint['current_set_images']
    iter_offset = checkpoint['i']
    G.load_state_dict(checkpoint['G_state'])
    D.load_state_dict(checkpoint['D_state'], strict=False)
    optimizerG.load_state_dict(checkpoint['G_optimizer'])
    optimizerD.load_state_dict(checkpoint['D_optimizer'])
    decayG.load_state_dict(checkpoint['G_scheduler'])
    decayD.load_state_dict(checkpoint['D_scheduler'])
    del checkpoint
    print_now(f'Resumed from iteration {current_set_images * param.gen_every}.')
else:
    current_set_images = 0
    iter_offset = 0
if not os.path.exists(param.extra_folder):
    os.makedirs(param.extra_folder)
sample_path = os.path.join(param.extra_folder, "samples")
if not os.path.exists(sample_path):
    os.makedirs(sample_path)

shutil.copyfile(configFile,os.path.join(param.extra_folder,"train_projection.yaml"))

sampleBatch = param.sample_batch_size
fixed_z = torch.randn(param.n_class*sampleBatch, param.z_size)
fixed_z = fixed_z.to("cuda")
fixed_c = Sampler.sampleFixedLabels(param.n_class,sampleBatch,"cuda")

print_now(G)
print_now(D)
wocao = iter(cycle(random_sample))
for i in range(iter_offset, param.total_iters):
    print('***** start training iter %d *******'%i)
    D.train()
    G.train()

    lossD = learnD_Realness(param, D, G, optimizerD, wocao, runingZ, runingLabel)
    lossG = learnG_Realness(param, D, G, optimizerG, wocao, runingZ, runingLabel)

    decayD.step()
    decayG.step()

    if i < 1000 or (i+1) % 100 == 0:
        end = time.time()
        fmt = '[%d / %d] SD: %d Diff: %.4f loss_D: %.4f loss_G: %.4f time:%.2f'
        s = fmt % (i+1, param.total_iters, param.seed,
                    -lossD.data.item() + lossG.data.item() if (lossD is not None) and (lossG is not None) else -1.0,
                    lossD.data.item()                      if lossD is not None else -1.0,
                    lossG.data.item()                      if lossG is not None else -1.0,
                    end - start)
        print_now(s)

    if (i+1) % param.gen_every == 0:
        current_set_images += 1
        if not os.path.exists('%s/models/' % (param.extra_folder)):
            os.mkdir('%s/models/' % (param.extra_folder))
        torch.save({
            'i': i + 1,
            'current_set_images': current_set_images,
            'G_state': G.state_dict(),
            'D_state': D.state_dict(),
            'G_optimizer': optimizerG.state_dict(),
            'D_optimizer': optimizerD.state_dict(),
            'G_scheduler': decayG.state_dict(),
            'D_scheduler': decayD.state_dict(),
        }, '%s/models/state_%02d.pth' % (param.extra_folder, current_set_images))
        print_now('Model saved.')

        if os.path.exists('%s/%01d/' % (param.extra_folder, current_set_images)):
            for root, dirs, files in os.walk('%s/%01d/' % (param.extra_folder, current_set_images)):
                for f in files:
                    os.unlink(os.path.join(root, f))
        else:
            os.mkdir('%s/%01d/' % (param.extra_folder, current_set_images))

        G.eval()
        with torch.no_grad():
            print("Sampling")
            fake_test_list = []
            fake_test = G(fixed_z,fixed_c).squeeze()
            fake_test = fake_test.cpu().clone().numpy()
            fake_test_list.extend(fake_test)
            x=[]
            y=[]
            for dot in fake_test_list:
                x.append(dot[0])
                y.append(dot[1])
            fig = plt.figure(figsize=(10,10))
            ax1 = fig.add_subplot(111)
            ax1.set_title('Points distribution')
            plt.xlabel('X')
            plt.ylabel('Y')
            ax1.scatter(x,y,c = 'r',marker = '.')
            plt.savefig(sample_path +'/test%d.png'%(i+1),bbox_inches = 'tight')
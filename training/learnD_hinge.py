import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
import math
import sys
import datetime
import time


def print_now(cmd, file=None):
    time_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if file is None:
        print('%s %s' % (time_now, cmd))
    else:
        print_str = '%s %s' % (time_now, cmd)
        print(print_str, file=file)
    sys.stdout.flush()

def learnD_Realness(param, D, G, optimizerD, random_sample, Triplet_Loss, x, anchor1, anchor0):
    device = 'cuda' if param.cuda else 'cpu'
    z = torch.FloatTensor(param.batch_size, param.z_size, 1, 1)
    z = z.to(device)

    for p in D.parameters():
        p.requires_grad = True

    for t in range(param.D_updates):
        D.zero_grad()
        optimizerD.zero_grad()

        # gradients are accumulated through subiters
        for _ in range(param.effective_batch_size // param.batch_size):
            images, _ = random_sample.__next__()
            x.copy_(images)
            del images

            # real images
            feat_real = D(x)
            # fake images
            z.normal_(0, 1)
            imgs_fake   = G(z)
            feat_fake   = D(imgs_fake.detach())
            d_loss_real = torch.nn.ReLU()(1.0 - feat_real).mean()
            d_loss_fake = torch.nn.ReLU()(1.0 + feat_fake).mean()

            lossD = d_loss_real + d_loss_fake
            lossD.backward()

        optimizerD.step()

    return lossD




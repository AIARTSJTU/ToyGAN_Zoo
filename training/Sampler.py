#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: Sampler.py
# Created Date: Monday February 17th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Monday, 17th February 2020 10:18:55 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################


import torch
import torch.nn as nn
import numpy as np

class Distribution(torch.Tensor):
  # Init the params of the distribution
  def init_distribution(self, dist_type, **kwargs):    
    self.dist_type = dist_type
    self.dist_kwargs = kwargs
    if self.dist_type == 'normal':
      self.mean, self.var = kwargs['mean'], kwargs['var']
    elif self.dist_type == 'categorical':
      self.num_categories = kwargs['num_categories']

  def sample_(self):
    if self.dist_type == 'normal':
      self.normal_(self.mean, self.var)
    elif self.dist_type == 'categorical':
      self.random_(0, self.num_categories)    
    # return self.variable
    
  # Silly hack: overwrite the to() method to wrap the new object
  # in a distribution as well
  def to(self, *args, **kwargs):
    new_obj = Distribution(self)
    new_obj.init_distribution(self.dist_type, **self.dist_kwargs)
    new_obj.data = super().to(*args, **kwargs)    
    return new_obj

def prepare_z_c(G_batch_size, dim_z, nclasses, device='cuda', z_var=1.0):

    z_ = Distribution(torch.randn(G_batch_size, dim_z, requires_grad=False))
    z_.init_distribution('normal', mean=0, var=z_var)
    z_ = z_.to(device, torch.float32)
    c_ = Distribution(torch.zeros(G_batch_size, requires_grad=False))
    c_.init_distribution('categorical',num_categories=nclasses)
    c_ = c_.to(device, torch.int64)
    return z_,c_

def prepareSampleZ(G_batch_size, dim_z, device='cuda', z_var=1.0):
  z_ = Distribution(torch.randn(G_batch_size, dim_z, requires_grad=False))
  z_.init_distribution('normal', mean=0, var=z_var)
  z_ = z_.to(device, torch.float32)
  return z_

def sampleG(G, z_, c_, parallel=False):
  with torch.no_grad():
    z_.sample_()
    c_.sample_()
    if parallel:
      G_z =  nn.parallel.data_parallel(G,[z_,c_])
    else:
      G_z = G(z_,c_)
    return G_z

def sampleFixedLabels(numClasses,batchSize,device):
  a = [1]*batchSize
  res = []
  for i in range(numClasses):
    res += [t*i for t in a]
  pseudo_labels = torch.tensor(res).long().to(device)
  return pseudo_labels
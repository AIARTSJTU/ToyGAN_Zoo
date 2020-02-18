#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: CategoricalConditionalBatchNorm1d.py
# Created Date: Monday February 17th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Monday, 17th February 2020 11:36:48 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################


import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from model.ConditionalBatchNorm1d import ConditionalBatchNorm1d

class CategoricalConditionalBatchNorm1d(ConditionalBatchNorm1d):

    def __init__(self, num_classes, num_features=1, eps=1e-5, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(CategoricalConditionalBatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.weights = nn.Embedding(num_classes, num_features)
        self.biases = nn.Embedding(num_classes, num_features)

        self._initialize()

    def _initialize(self):
        init.ones_(self.weights.weight.data)
        init.zeros_(self.biases.weight.data)

    def forward(self, input, c, **kwargs):
        weight = self.weights(c)
        bias = self.biases(c)

        return super(CategoricalConditionalBatchNorm1d, self).forward(input, weight, bias)
#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: MixtureGaussianData.py
# Created Date: Monday February 17th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Monday, 17th February 2020 7:02:52 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################
from torch.utils import data
import torch
import os
import random
import pickle

label_table = {
    "2,-2":0,
    "2,0":1,
    "2,2":2,
    "0,-2":3,
    "0,0":4,
    "0,2":5,
    "-2,-2":6,
    "-2,0":7,
    "-2,2":8,
}

class MixtureGaussianData(data.Dataset):
    """Dataset class for the Mixture Gaussian dataset."""

    def __init__(self, dataset_path):
        """Initialize and preprocess the Mixture Gaussian dataset."""
        self.dataset_path   = dataset_path
        self.train_dataset  = []
        self.preprocess()
        self.num_dataset    = len(self.train_dataset)

    def preprocess(self):
        """Preprocess the Mixture Guassian data and its label."""
        with open(self.dataset_path, 'rb') as f:
            self.train_dataset = pickle.load(f)
        random.shuffle(self.train_dataset)
        print('Finished preprocessing the Mixture Guassian dataset...')

    def __getitem__(self, index):
        """Return one points and its corresponding attribute label."""
        point = self.train_dataset[index]
        label = point[1]
        label = torch.FloatTensor([label])
        value = torch.FloatTensor(point[0])
        return value, label

    def __len__(self):
        """Return the number of points."""
        return self.num_dataset

def get_dataset(path,batch_size,num_workers):
    wocao = MixtureGaussianData(path)
    data_loader = data.DataLoader(dataset=wocao,batch_size=batch_size,drop_last=True,shuffle=True,num_workers=num_workers,pin_memory=True)
    return data_loader

if __name__ == "__main__":
    # data_path = "./data/MixtureGaussian3By3.pk"
    # data_set  = []
    # with open(data_path, 'rb') as f:
    #     train_dataset = pickle.load(f)
    
    # for item in train_dataset:
    #     coor_str = ""
    #     if item[0] < -1:
    #         coor_str += "-2"
    #     elif item[0] > 1:
    #         coor_str += "2"
    #     else:
    #         coor_str += "0"
        
    #     if item[1] < -1:
    #         coor_str += ",-2"
    #     elif item[1] > 1:
    #         coor_str += ",2"
    #     else:
    #         coor_str += ",0"
    #     label = label_table[coor_str]
    #     data_set.append([item,label])
    # with open('./data/MixtureGaussianCXH.pk', 'wb') as f:
    #     pickle.dump(data_set, f, protocol=pickle.HIGHEST_PROTOCOL)
    path = './data/MixtureGaussianCXH.pk'
    wocao = get_dataset(path,64,8)
    wocao = iter(wocao)
    print(len(wocao))
    for i in range(100):
        a,label = next(wocao)
        print(a)
        print(label)
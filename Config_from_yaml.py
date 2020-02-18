#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: Config_from_yaml.py
# Created Date: Monday February 17th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Monday, 17th February 2020 11:13:36 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################


import yaml
from easydict import EasyDict

def get_config_from_yaml(yaml_file):
    with open(yaml_file, 'r') as config_file:
        try:
            config_dict = yaml.load(config_file)
            config = EasyDict(config_dict)
            return config
        except ValueError:
            print('INVALID YAML file format.. Please provide a good yaml file')
            exit(-1)

if __name__ == "__main__":
    a = get_config_from_yaml("./train_projection.yaml")
    print(a.batch_size)
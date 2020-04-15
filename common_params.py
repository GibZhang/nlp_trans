#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @Time    : 2020-04-13 21:55
# @Author  : zhangjingbo
# @mail    : jingbo.zhang@yooli.com
# @File    : common_params.py
# @Description: 
# @Software: PyCharm
import torch

device = "gpu" if torch.cuda.is_available() else "cpu"
PAD_token = 0
SOS_token = 1
EOS_token = 2

MAX_LENGTH = 10
hidden_size = 256
batch_size = 64
input_n_layers = 1
output_n_layers = 2
dropout = 0.1

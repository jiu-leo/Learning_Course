# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/8 11:12
@Auth ： zmmmmm
@File ：09 CNN00.py
@IDE ：PyCharm
"""
import torch

###### test1
# in_channels, out_channels = 5,10
# width, height = 100,100
# kernel_size = 3 # 卷积核大小，若为常数3，即3×3；若为元组(5,3)，即5×3
# batch_size = 1
#
# input_ = torch.randn(batch_size,in_channels,width,height) # randn为正态分布采样随机数
# conv_layer = torch.nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size )
# output = conv_layer(input_)
# print(input_.shape)
# print(output.shape)
# print(conv_layer.weight.shape)

##### test2

# import torch
#
input = [3,4,6,5,7,
         2,4,6,8,2,
         1,6,7,8,4,
         9,7,4,6,2,
         3,7,5,4,1]

inputs = torch.Tensor(input).view(1,1,5,5) # B,C,W,H  其中batch_size=1意味着一次送入一张照片

conv_layer = torch.nn.Conv2d(1,1,kernel_size=3,padding=1,bias=False)
kernel = torch.Tensor([1,2,3,4,5,6,7,8,9]).view(1,1,3,3)
conv_layer.weight.data = kernel.data

output = conv_layer(inputs)
print(output)


####### test3

import torch

inputs = [3, 4, 6, 5, 7,
         2, 4, 6, 8, 2,
         1, 6, 7, 8, 4,
         9, 7, 4, 6, 2,
         3, 7, 5, 4, 1]
inputs = torch.Tensor(inputs).view(1, 1, 5, 5)
conv_layer = torch.nn.Conv2d(1, 1, kernel_size=3, stride=2, bias=False)
kernel = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]).view(1, 1, 3, 3)
conv_layer.weight.data = kernel.data
output = conv_layer(inputs)
print(output)

inputs = [[3, 4, 6, 5, 7],
         [2, 4, 6, 8, 2],
         [1, 6, 7, 8, 4],
         [9, 7, 4, 6, 2],
         [3, 7, 5, 4, 1]]
inputs = torch.Tensor(inputs).view(1, -1)
print(inputs)
print(inputs.shape)
print(inputs.size())

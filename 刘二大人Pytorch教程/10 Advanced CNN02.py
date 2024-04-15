# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/10 16:54
@Auth ： zmmmmm
@File ：10 Advanced CNN02.py
@IDE ：PyCharm
"""

import torch.nn as nn
import torch.nn.functional as F

'''
ResidualBlock 的核心思想是通过引入一个“残差连接”（residual connection）或“短路连接”（shortcut connection），
使得网络可以学习输入和输出之间的残差，而不是直接学习输入到输出的映射。
这种设计有助于缓解深度神经网络在训练过程中的优化难题
'''
### 残差网络、残差块（见刘二大人B站视频-Advanced CNN）
#1.出发点：当网络的层数过多时，离输入最近的层会出现梯度消失的情况，（梯度小于1，多层累加，会使最初层的梯度接近0）
#2.避免梯度消失的情况发生，在一部分层之后（隔一段时间）加上最开始的输入向量，会使梯度大于1
##必须保证当时层的输出向量维度与最开始的输入向量维度相同

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels,channels,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(channels,channels,kernel_size=3,padding=1)

    def forward(self,x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x+y)
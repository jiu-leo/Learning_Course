# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/8 10:17
@Auth ： zmmmmm
@File ：test3.py
@IDE ：PyCharm
"""
import torch

correct = 0
predicted = torch.Tensor([1,2,3,4,5])
labels = torch.Tensor([1,2,3,4,6])
correct += (predicted == labels).sum().item()
print(correct)
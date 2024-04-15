# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/9 17:20
@Auth ： zmmmmm
@File ：10 Advanced CNN00.py
@IDE ：PyCharm
"""
import torch

'''
concatenate的作用
'''
# batch_size = 1
# width = 2
# height = 2
#
# branch1 = torch.randn(batch_size,3,width,height)
# branch2 = torch.randn(batch_size,2,width,height)
# branch3 = torch.randn(batch_size,1,width,height)
# print(branch1)
# print(branch2)
# print(branch3)
# outputs = [branch1, branch2, branch3]
# # print(outputs)
# concatenate = torch.cat(outputs, dim=1)
# print(concatenate)

batch_size = 32

outputs = torch.randn(batch_size,10)
res = torch.max(outputs.data,dim=1)
# torch.max 函数用于找到张量中指定维度上的最大值。
# 你正在寻找 outputs 张量中每一行（即每个样本）的最大值，并返回这些最大值以及它们对应的索引
# res 是一个元组，包含两个张量：
# 最大值张量：一个形状为 (batch_size,) 的张量，包含 outputs 中每行的最大值。
# 索引张量：一个形状也为 (batch_size,) 的张量，包含 outputs 中每行最大值的索引。
_, predicted = res
print(res)
print(type(res))
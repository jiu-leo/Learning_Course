# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/10 17:28
@Auth ： zmmmmm
@File ：11 Basic RNN00.py
@IDE ：PyCharm
"""

import torch

############# torch.nn.RNNCell，使用RNNCell，需要自己构建循环，输入是序列中的一个词或一个部分
# batch_size = 1
# seq_len = 3
# input_size = 4
# hidden_size = 2
#
# cell = torch.nn.RNNCell(input_size=input_size, hidden_size = hidden_size)
# # 输入维度：(输入序列的长度，batch_size的大小，每一个输入x的维度)
# dataset = torch.randn(seq_len, batch_size, input_size)
# hidden = torch.zeros(batch_size, hidden_size)
#
# for idx, inputs in enumerate(dataset):
#     print('='*20, idx,'='*20 )
#     print(f'Inputsize:{inputs.shape}')  # [1, 4]
#
#     hidden = cell(inputs, hidden)
#
#     print(f'hidden size:', hidden.shape)
#
#     print((hidden))

############# torch.nn.RNN，使用RNN，不需要自己构建循环，输入是一个序列，封装了处理单元

# import torch
# batch_size = 1
# seq_len = 3
# input_size = 4
# hidden_size = 2
# num_layers = 1
#
# cell = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size,num_layers=num_layers)
#
# # inputs = torch.randn(seq_len, batch_size, input_size) # [3,1,4]
# # batch_first=True,将batch_size放在序列长度seq_len的前面
# inputs = torch.randn(batch_size, seq_len, input_size, batch_first=True) # [3,1,4]
# hidden = torch.zero_(num_layers,batch_size,hidden_size)
#
# out, hidden = cell(inputs,hidden)
#
# print('output size:',out.shape)
# print(out)
# print('Hidden size:',hidden.shape)
# print((hidden))
inputs = torch.Tensor([[[1],[2]],
          [[3],[4]],
          [[5],[6]]])

print(inputs.shape)
print(inputs)

res = torch.Tensor([[1],
                    [2],
                    [3],
                    [4],
                    [5],
                    [6],])

print(res.shape)
print(res)
res1 = res.view(2,3)
print(res1.shape)
print(res1)
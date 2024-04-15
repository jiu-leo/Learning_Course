# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/3 14:04
@Auth ： zmmmmm
@File ：03 反向传播.py
@IDE ：PyCharm
"""
import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.Tensor([1.0])
w.requires_grad = True  # 创建张量默认不需要计算梯度，需要指定

# print(w.item())
# print(type(w.data))
# print(w.grad)
# print(type(w.grad))
def forward(x):
    return x*w

def loss(x,y):
    y_pred = forward(x)
    return (y_pred-y)**2

print('predict before training',4,forward(4).item())


# 训练过程(随机梯度下降)
for epoch in range(1):
    for x,y in zip(x_data, y_data):
        l = loss(x,y)
        l.backward()
        print('grad:',x,y,w.grad.item())
        w.data -= 0.01*w.grad.data

        w.grad.data.zero_() #

    print('epoch:',epoch,l.item())

print('predict before training',4,forward(4).item())


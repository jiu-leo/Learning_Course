# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/3 14:04
@Auth ： zmmmmm
@File ：03 反向传播.py
@IDE ：PyCharm
"""
import torch
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w1 = torch.Tensor([1.0])
w1.requires_grad = True  # 创建张量默认不需要计算梯度，需要指定

w2 = torch.Tensor([1.0])
w2.requires_grad = True

b = torch.Tensor([1.0])
b.requires_grad = True

# print(w.item())
# print(type(w.data))
# print(w.grad)
# print(type(w.grad))
def forward(x):
    return w1*x*x + w2*x + b

def loss(x,y):
    y_pred = forward(x)
    return (y_pred-y)**2

print('predict before training',4,forward(4).item())

epoch_list = []
loss_list = []

# 训练过程(随机梯度下降)
for epoch in range(100):
    for x,y in zip(x_data, y_data):
        l = loss(x,y)
        l.backward()
        print('grad:',x,y,w1.grad.item(),w2.grad.item(),b.grad.item())
        w1.data -= 0.01*w1.grad.data
        w2.data -= 0.01 * w2.grad.data
        b.data -= 0.01 * w2.grad.data

        w1.grad.data.zero_() # 梯度清零
        w2.grad.data.zero_()
        b.grad.data.zero_()

    epoch_list.append(epoch)
    loss_list.append(l.item())

    print('epoch:',epoch,l.item())

print('predict before training',4,forward(4).item())

plt.plot(epoch_list, loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
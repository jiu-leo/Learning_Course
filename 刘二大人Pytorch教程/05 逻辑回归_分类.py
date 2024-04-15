# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/3 19:28
@Auth ： zmmmmm
@File ：05 逻辑回归_分类.py
@IDE ：PyCharm
"""


# import torchvision
# train_set = torchvision.datasets.MNIST(root='./dataset/mnist',train=True,download=True) # 训练集
# test_set = torchvision.datasets.MNIST(root='./dataset/mnist',train=False,download=True) # 测试集

import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt



# 准备数据集
x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[0],[0],[1]])

# 模型
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self): # 有参数，需要在构造函数中进行初始化
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x): # 没有参数，不需要在构造函数中进行初始化，可以直接写到forward中
        y_pred = F.sigmoid(self.linear(x)) # 函数, F.sigmoid没有参数，可以直接写到forward中
        return y_pred


model = LogisticRegressionModel()

# 构造损失函数和优化器（使用Pytorch API）
criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练周期：前馈、反馈、更新
for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 测试
x = np.linspace(0, 10, 200)  # 0-10h采样200个点
x_t = torch.Tensor(x).view((200, 1))  # 变为200行1列的矩阵，类似于numpy里的reshape
y_t = model(x_t)
y = y_t.data.numpy()  # 拿到数组
plt.plot(x, y)
plt.plot([0, 10], [0.5, 0.5], c='r')
plt.xlabel('Hours')
plt.ylabel('Probability of Pass')
plt.grid()  # 网格
plt.show()

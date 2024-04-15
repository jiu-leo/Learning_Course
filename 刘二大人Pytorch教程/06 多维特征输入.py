# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/7 10:07
@Auth ： zmmmmm
@File ：06 多维特征输入.py
@IDE ：PyCharm
"""

import torch
import numpy as np

import matplotlib.pyplot as plt

xy = np.loadtxt('./dataset/diabetes/diabetes.csv.gz', delimiter=',', dtype=np.float32)  # delimiter为分隔符
# torch.from_numpy()创建Tensor
x_data = torch.from_numpy(xy[:, :-1])  # 所有行，第1列开始，最后1列不要
y_data = torch.from_numpy(xy[:, [-1]])  # -1外面加[]是为了保证拿出来是一个矩阵，不加的话拿出来是一个向量

# 定义模型（输入8维—>线性层1—>6维—>线性层2—>4维—>线性层3—>输出1维）
class Model(torch.nn.Module):
    def __init__(self): # 有参数，需要在构造函数中进行初始化
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        # Linear有参数，所以构造不同的linear层，需要定义多个
        # 不直接将8->1，是为了更好地拟合非线性变换，多步转化多次调用sigmoid函数
        self.sigmoid = torch.nn.Sigmoid()
        # Sigmoid模块，用于构建module层，作为运算模块，构造计算图
        # 只需要定义一个，因为它没有参数

    def forward(self, x): # 没有参数，不需要在构造函数中进行初始化，可以直接写到forward中
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

model = Model()

# 构建损失和优化器
criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 训练周期（训练时尚未使用Mini-Batch）
epoch_list = []
loss_list = []
for epoch in range(100):
    # 前馈
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)

    epoch_list.append(epoch)
    loss_list.append(loss.item())

    print(epoch, loss.item())

    # 反馈
    optimizer.zero_grad()
    loss.backward()

    # 更新
    optimizer.step()


plt.plot(epoch_list, loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
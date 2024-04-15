# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/2 16:06
@Auth ： zmmmmm
@File ：02 梯度下降.py
@IDE ：PyCharm
"""

import matplotlib.pyplot as plt


# 准备训练集
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 设置一个初始权重（随机）
w = 3.0

# 定义模型
def forward(x):
    return x*w

# 定义损失函数
def cost(x_list, y_list):
    costs = 0
    for x,y in zip(x_list,y_list):
        y_pred = forward(x)
        costs +=(y_pred-y)**2
        return costs/len(x_list)

# 定义梯度函数
def gradient(x_list, y_list):
    grad = 0
    for x, y in zip(x_list, y_list):
        grad += 2*x*(x*w-y)

    return grad/len(x_list)


# 训练
epoch_list = []
cost_list = []

print('Predict(after training)', 4, forward(4))

for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= 0.01*grad_val # 学习率设置为0.01
    print(f'Epoch:{epoch}, w={w}, cost={cost_val}')
    epoch_list.append(epoch)
    cost_list.append(cost_val)

print('Predict(after training)', 4, forward(4))

plt.plot(epoch_list, cost_list)
plt.ylabel('cost')
plt.xlabel('epoch')
plt.show()
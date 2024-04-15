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
def loss(x, y):
    y_pred = forward(x)
    return (y_pred-y)**2


# 定义梯度函数
def gradient(x, y):
    return 2*x*(x*w-y)

# 训练
epoch_list = []
cost_list = []

print('Predict(after training)', 4, forward(4))

for epoch in range(100):
    for x, y in zip (x_data, y_data):
        # 对每一个样本的梯度进行更新，效率低
        grad = gradient(x,y)
        w = w - 0.01 * grad
        losss = loss(x, y)
        print(x, y, grad)
    print(f'Epoch:{epoch}, w={w}, cost={losss}')

print('Predict(after training)', 4, forward(4))

plt.plot(epoch_list, cost_list)
plt.ylabel('cost')
plt.xlabel('epoch')
plt.show()
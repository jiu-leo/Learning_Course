# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/2 9:53
@Auth ： zmmmmm
@File ：01 线性模型1.py
@IDE ：PyCharm
"""

"""
y = w*x
"""
import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 模型
def forward(x):
    return x*w

# 损失函数
def loss(x, y):
    y_pred = forward(x)
    return (y_pred-y)*(y_pred-y)

w_list = []
mse_list = []

for w in np.arange(0.0, 4.1, 0.1):
    print('w=', w)
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        # 使用zip函数将x_data和y_data中的对应元素组合成一个元组，并遍历这些元组。
        # 在每次迭代中，x_val和y_val分别表示x_data和y_data中的一个元素
        y_pred_val = forward(x_val) # 只是为了打印
        loss_val = loss(x_val, y_val) # loss()已经计算了损失
        l_sum += loss_val
        print('\t', x_val, y_val, y_pred_val, loss_val)
    print('MSE=', l_sum/3)
    w_list.append(w)
    mse_list.append(l_sum/3)

plt.plot(w_list, mse_list)
# 调用plot函数来绘制一个线图,w_list是x轴的数据,mse_list是y轴的数据
plt.ylabel('Loss')
plt.xlabel('w')
# 设置x轴和y轴的标签
plt.show()
# 显示绘制的图形














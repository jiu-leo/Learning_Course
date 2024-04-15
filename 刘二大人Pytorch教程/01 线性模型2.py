# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/2 11:13
@Auth ： zmmmmm
@File ：01 线性模型2.py
@IDE ：PyCharm
"""
"""
y = w*x + b
"""

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

# Make data.
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


# 定义模型
def forward(x):
    return x * w + b


# 定义损失函数
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


# 定义空列表
w_list = []
b_list = []
mse_list = []

for w in np.arange(0.0, 4.1, 0.1):
    for b in np.arange(-2.0, 2.1, 0.1):
        print('w=', w, 'b=', b)
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            y_pred_val = forward(x_val)
            loss_val = loss(x_val, y_val)
            l_sum += loss_val
            print('\t', x_val, y_val, y_pred_val, loss_val)
        print("MSE=", l_sum / 3)
        print('--------------------------------')
        w_list.append(w)
        b_list.append(b)
        mse_list.append(l_sum / 3)

'''
此时得到的mse_list是一个列表
 1、将其转化成矩阵（1681*1）
 2、将其转化成41*41的矩阵（此时其与w和b的值并不对应，即x axis与y axis反了）
 3、故转置矩阵
 具体示例见test.py
'''
mse_list = np.array(mse_list)  # 将列表转化成矩阵
print(mse_list)
mse_list = mse_list.reshape(41, 41)  # 将矩阵从1681*1变为41*41
print(mse_list)
mse_list = mse_list.transpose()  # 转置矩阵
print(mse_list)


# w和b由于嵌套的for循环，每个值都出现了41次，故要去重，接下来使用meshgrid将w和b转化成41*41的矩阵
w, b = np.meshgrid(np.unique(w_list), np.unique(b_list))

####### 创建子图的方式 ########
# 方式一
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# 添加一个3D子图到图形中
# 这里的参数111是一个三位数，代表子图的布局和位置
# 第一个数字表示行号，第二个数字表示列号，第三个数字表示子图序号
# 在这个例子中，111表示第一个（也是唯一一个）子图，占据一行一列的位置

# 方式二
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# fig 是一个 Figure 对象，它代表整个图形窗口。
# ax 是一个 Axes3D 对象，它代表一个3D子图，是fig子图中的一个
# subplot_kw={"projection": "3d"}，设置 projection 参数为 "3d"，
# matplotlib 会知道它应该使用 Axes3D 而不是标准的 Axes 对象来创建子图。

# Plot the surface.
surf = ax.plot_surface(w, b, mse_list, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(0, 35)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')  # . 02f效果是z轴上的数字尺度保留两位小数

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

# 给每个轴标上含义
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.text2D(0.4, 0.92, "Cost Values", transform=ax.transAxes)

plt.show()
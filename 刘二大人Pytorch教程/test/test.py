import numpy as np


# 定义空列表
w_list = []
b_list = []
mse_list = []
count = 0

for w in np.arange(0.0, 4.1, 0.1):
    for b in np.arange(-2.0, 2.1, 0.1):
        count +=1
        w_list.append(w)
        b_list.append(b)
        mse_list.append(count)

print(w_list)
print('*'*50)

print(b_list)
print('*'*50)

print(mse_list)
print('*'*50)
mse_list = np.array(mse_list)

mse_list = mse_list.reshape(41, 41)

mse_list = mse_list.transpose()  # 转置矩阵
print(mse_list)
print('*'*50)

w, b = np.meshgrid(np.unique(w_list), np.unique(b_list))

print(w)
print('*'*50)
print(b)
# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/7 14:21
@Auth ： zmmmmm
@File ：07 dataset and data loader.py
@IDE ：PyCharm
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',',dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:,:-1]) # 变为张量维度为（N，8）---类似于矩阵
        self.y_data = torch.from_numpy(xy[:,[-1]]) # 加[],将一维张量变为二维

    def __getitem__(self, item):
        return self.x_data[item],self.y_data[item]

    def __len__(self):
        return self.len

dataset = DiabetesDataset('./dataset/diabetes/diabetes.csv.gz')

train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=0)

# train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

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



if __name__ == '__main__':

    for epoch in range(100):
        for i, data in enumerate(train_loader, 0):
            # dataset每次返回一个样本，dataloader接受数据集返回的样本，自动根据batch_size的大小，封装为一个batch，进行迭代
            # i表示第i个batchsize
            # enumerate 函数用于遍历 train_loader 中的数据，并同时返回每个数据项的索引 i 和数据本身 data。
            # train_loader 通常是一个数据加载器，用于批量地提供训练数据。
            # enumerate 函数的第二个参数 0 是可选的，用于指定索引的起始值，这里设置为 0
            inputs, labels = data
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()


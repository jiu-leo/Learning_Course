import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# 1、准备数据（不再是加载全部数据）
class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]  # xy为N行9列（N行：数据样本个数；9列：8个特征列，1个目标列）
        self.x_data = torch.from_numpy(xy[:, :-1])  # 前8列
        self.y_data = torch.from_numpy(xy[:, [-1]])  # 最后1列

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]  # 返回的是一个元组

    def __len__(self):
        return self.len


dataset = DiabetesDataset('../dataset/diabetes/diabetes.csv.gz')
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=0)


# 2、设计模型
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()

# 3、构建损失和优化器
criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 4、训练周期（变为了嵌套循环，以便使用Mini-Batch）
# 嵌套循环
for epoch in range(100):
    # enumerate()获得当前是第几次迭代
    for i, data in enumerate(train_loader, 0):  # data为元组(x,y)
        # 1. 准备数据
        inputs, labels = data
        # 2. 前馈
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        print(epoch, i, loss.item())
        # 3. 反馈
        optimizer.zero_grad()
        loss.backward()
        # 4. 更新
        optimizer.step()
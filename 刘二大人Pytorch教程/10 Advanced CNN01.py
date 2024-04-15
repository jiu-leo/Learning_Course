# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/10 11:16
@Auth ： zmmmmm
@File ：10 Advanced CNN01.py
@IDE ：PyCharm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms   # 对图像进行原始处理的工具
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim   # 为了构建优化器


batch_size = 128

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307, ),(0.3081, ))])

train_dataset = datasets.MNIST(root='./dataset/mnist', train=True, download=False, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

test_dataset = datasets.MNIST(root='./dataset/mnist', train=False, download=False, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)



# 不用关注W,H是多少，只要保证在模型中W,H尺寸不变，只关注输入通道和输出通道的变化
class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionA,self).__init__()

        # 分支一
        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size= 1)

        # 分支二
        self.branch_1x1 = nn.Conv2d(in_channels,16,kernel_size = 1)

        # 分支三
        self.branch5x5_1 = nn.Conv2d(in_channels,16,kernel_size = 1)
        self.branch5x5_2 = nn.Conv2d(16,24,kernel_size = 5,padding =2)

        # 分支四

        self.branch3x3_1 = nn.Conv2d(in_channels,16,kernel_size = 1)
        self.branch3x3_2 = nn.Conv2d(16,24,kernel_size = 3,padding =1)
        self.branch3x3_3 = nn.Conv2d(24,24,kernel_size = 3,padding =1)


    def forward(self, x):

        # 分支一
        branch_pool = self.branch_pool(F.avg_pool2d(x,kernel_size=3,stride=1,padding=1))

        # 分支二
        branch_1x1 = self.branch_1x1(x)

        # 分支三
        branch_5x5 = self.branch5x5_2(self.branch5x5_1(x))

        # 分支四
        branch_3x3 = self.branch3x3_3(self.branch3x3_2(self.branch3x3_1(x)))

        outputs = [branch_pool,branch_1x1,branch_3x3,branch_5x5]

        return torch.cat(outputs, dim=1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size=5)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)
        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)

        self.mp = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(1408,1000)
        self.fc2 = nn.Linear(1000, 600)
        self.fc3 = nn.Linear(600, 10)

    def forward(self,x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        x =  F.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)

        x = x.view(in_size,-1)

        x = self.fc3(self.fc2(self.fc1(x)))
        return x

model = Net()

model.to(device)


criterion = torch.nn.CrossEntropyLoss()
criterion.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    running_loss = 0
    for batch_idx, data in enumerate(train_loader,0):
        inputs, target = data
        inputs = inputs.to(device)
        target = target.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs,target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 300 == 299:
            print(f'{epoch + 1:d} {batch_idx + 1:5d} loss:{running_loss / 300:.3f}')
            running_loss = 0

epoch_list = []
accuracy_list = []

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, label = data
            images = images.to(device)
            label = label.to(device)
            outputs = model(images)
            _,predicted = torch.max(outputs.data,dim=1)
            total += label.size(0)
            correct += (predicted==label).sum().item()
        print(f'Accuracy on test set: {100 * correct / total}')

    accuracy_list.append(correct / total)








if __name__ == '__main__':
    # 找到线性层的输入特征维度
    # batch_size = 32
    # inputs = torch.randn(batch_size, 1, 28, 28)
    # model = Net()
    # res = model(inputs)
    # print(res.shape)

    # inputs1 = torch.randn(batch_size, 1, 28, 28)
    # conv1 = nn.Conv2d(1, 10, kernel_size=5)
    # pool2d = nn.MaxPool2d(2)
    # res = conv1(inputs1)
    # print(res.shape)
    # res = pool2d(conv1(inputs1))
    # print(res.shape)

    for epoch in range(10):
        train(epoch)
        epoch_list.append(epoch)
        test()

    import matplotlib.pyplot as plt



    plt.plot(epoch_list,accuracy_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()
    # model = Net()
    # print(model)

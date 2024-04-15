# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/7 16:49
@Auth ： zmmmmm
@File ：08 多分类问题.py
@IDE ：PyCharm
"""
# 0.导包
import torch
from torchvision import transforms  # 对图像进行原始处理的工具
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F  # 为了使用函数 relu()
import torch.optim as optim  # 为了构建优化器

batch_size = 64
transform = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.1307,),(0.3081,))])

train_dataset = datasets.MNIST(root='./dataset/mnist', train=True, download=False, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

test_dataset = datasets.MNIST(root='./dataset/mnist', train=False, download=False, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

class Net(torch.nn.Module): # 全连接神经网络，网络全都是由线性层串接起来的网络
    def __init__(self): # 需要参数的函数，和模块定义在init中，
        super(Net,self).__init__()
        self.l1 = torch.nn.Linear(784,512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self, x): # 模型的步骤定义在forward中，当调用模型对象时，会自动执行forward
        x = x.view(-1, 784)
        # 神经网络的输入需要是矩阵，view()将x转化为矩阵，
        # view() 包含两个参数，即代表二阶张量（矩阵），第二个参数为矩阵列数，-1表示自动计算行数
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        # relu()，非线性激活层不改变x的维度，
        # 划分多个维度变换在每一个变换之后加上非线性激活层是为了更好的拟合非线性变换（所以可能的变换状态）
        return self.l5(x)  # 最后一层不做激活，是因为CrossEntropyLoss中包含了softmax，在计算损失时会自动计算

model = Net()

criterion = torch.nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.01,momentum=0.5)

def train(epoch):
    running_loss = 0
    for batch_idx, data in enumerate(train_loader,0):
        inputs, target = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        # nn.CrossEntropyLoss() 会自动处理target的索引，并与 outputs 中的对应类别得分匹配，计算损失。
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx%300 == 299:
            # '[%d,%5d]loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300)
            print(f'{epoch + 1:d} {batch_idx + 1:5d} loss:{running_loss / 300:.3f}')
            running_loss = 0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, label = data
            outputs = model(images)
            _,predicted = torch.max(outputs.data,dim=1)
            total += label.size(0)
            correct += (predicted==label).sum().item()
        print(f'Accuracy on test set: {100*correct/total}')






if __name__ == '__main__':
    # x = torch.Tensor([[-1,2,4,-3,9],[-1,2,4,-3,9]])
    # x = F.relu(x) 将负数变为0
    # print(x)
    # tensor([[0., 2., 4., 0., 9.],
    #         [0., 2., 4., 0., 9.]])
    for epoch in range(15):
        train(epoch)
        test()

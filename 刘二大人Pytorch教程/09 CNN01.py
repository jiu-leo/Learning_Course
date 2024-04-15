# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/9 10:44
@Auth ： zmmmmm
@File ：09 CNN01.py
@IDE ：PyCharm
"""
import torch
from torchvision import transforms  # 对图像进行原始处理的工具
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F  # 为了使用函数 relu()
import torch.optim as optim  # 为了构建优化器
import timeit

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 64
transform = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.1307,),(0.3081,))])

train_dataset = datasets.MNIST(root='./dataset/mnist', train=True, download=False, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

test_dataset = datasets.MNIST(root='./dataset/mnist', train=False, download=False, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = torch.nn.Conv2d(1,10,kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.linear= torch.nn.Linear(320,10)

    def forward(self,x): # x:(batch_size,C,W,H) 卷积神经网络的输入是四维张量
        batch_size = x.size(0)
        x = self.pooling(F.relu(self.conv1(x)))
        x = self.pooling(F.relu(self.conv2(x)))
        x = x.view(batch_size,-1) # 全连接神经网络的输入是二维张量
        # 神经网络的输入必须包含batch_size
        x = self.linear(x)
        return x   # 最后一层不做激活，因为要算交叉熵损失,交叉熵损失中包含了softmax层

model = Net()

model.to(device)

criterion = torch.nn.CrossEntropyLoss()
criterion.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01,momentum=0.5)

def train(epoch):
    running_loss = 0
    for batch_idx, data in enumerate(train_loader,0):
        inputs, target = data
        inputs = inputs.to(device)
        target = target.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx%300 == 299:
            # '[%d,%5d]loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300)
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
        print(f'Accuracy on test set: {100*correct/total}')

    accuracy_list.append(correct / total)



if __name__ == '__main__':
    start_time = timeit.default_timer()
    for epoch in range(10):
        train(epoch)
        epoch_list.append(epoch)
        test()
    end_time = timeit.default_timer()
    elapsed_time = end_time - start_time
    print(f"程序运行时间: {elapsed_time:.6f} 秒")

    import matplotlib.pyplot as plt

    plt.plot(epoch_list, accuracy_list)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.grid()
    plt.show()

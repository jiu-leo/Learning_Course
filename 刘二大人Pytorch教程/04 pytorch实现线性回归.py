# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/3 15:48
@Auth ： zmmmmm
@File ：04 pytorch实现线性回归.py
@IDE ：PyCharm
"""
# 更有弹性

# 1.准备数据集
# 使用torch，x、y必须是矩阵
import torch

x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[2.0],[4.0],[6.0]])

# 2.定义模型，计算y_pred
class LinearModel(torch.nn.Module): # 用继承nn.Module的模块构造出来的对象，会自动的根据计算图进行backward过程
    def __init__(self): # 构造函数，初始化对象的时候默认调用的函数
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1,1) # linear是一个Linear对象，也继承了Module
    """
    Module自带的__call__方法，定义了forward()函数，当调用类实例化对象时，会触发__call__方法，执行forward()函数
    def __call__(): 
        forward()
    继承Module，调用对象的时候，会自动执行forward()方法
    """

    def forward(self, x): # 进行前馈的时候，所要执行的计算
        y_pred = self.linear(x) # 调用linear对象，执行Linear类中的forward()，计算y_pred
        return y_pred

model = LinearModel() # 可以被callable（唤醒的）：model()，通过对象调用的方式，执行forward()


# 3.构造损失函数和优化器
criterion = torch.nn.MSELoss(size_average=False) # size_average指定是否求均值

optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # model.parameters()获取需要计算的参数
# 优化器知道要对哪些权重做优化

# 4.定义训练周期，forward、backward、update

for epoch in range(1000):
    y_pred = model(x_data) # 对象调用，直接执行forward()
    loss = criterion(y_pred, y_data)
    print(epoch, loss)

    optimizer.zero_grad() # .backward()计算的梯度会被累加，因此计算前先清零
    loss.backward()  # 随机挑一个样本计算梯度（x，y）？？？
    # 有疑惑，这里的优化采用随机梯度下降，那么损失是怎么更新的呢？
    # 如果是一个一个样本更新的，w的更新影响下一次的预测，但y_pred已经计算了，所以到底是怎么进行的？
    # 还是随机挑选一个样本，这个看起来比较合理，只进行一次更新，目前是这么理解的---批量随机梯度下降，n=3，随机挑一个样本进行梯度下降
    optimizer.step() # 更新参数

# 输出权重和偏置
print('w=', model.linear.weight.item())
print('b=', model.linear.bias.item())

# 测试模型
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred=', y_test.data)

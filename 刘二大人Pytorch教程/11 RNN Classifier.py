# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/15 17:00
@Auth ： zmmmmm
@File ：11 RNN Classifier.py
@IDE ：PyCharm
"""

# 导入第三方库
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
import torch
import gzip
import csv
import matplotlib.pyplot as plt
import numpy as np
import time
import math

# 参数设置
HIDDEN_SIZE = 100
BATCH_SIZE = 256
N_LAYER = 2
N_EPOCHS = 100
N_CHARS = 128
USE_GPU = True


# 数据类
class NameDataset(Dataset):
    def __init__(self, is_train_set=True):  # 构造函数
        filename = 'dataset/name/names_train.csv.gz' if is_train_set else 'dataset/name/names_test.csv.gz'
        # 此处与刘二老师的源码不同
        with gzip.open(filename, 'rt') as f:
            reader = csv.reader(f)
            rows = list(reader)
        self.names = [row[0] for row in rows]  # names
        self.len = len(self.names)  # 样本数
        self.countries = [row[1] for row in rows]  # countries
        self.country_list = list(sorted(set(self.countries)))  # set()去重，删除重复的数据； sorted()排序
        self.country_dict = self.getCountryDict()
        self.country_num = len(self.country_list)  # 国家数

    def __getitem__(self, index):  # 取数据
        return self.names[index], self.country_dict[self.countries[index]]

    def __len__(self):  # 取样本数
        return self.len

    def getCountryDict(self):
        country_dict = dict()
        for idx, country_name in enumerate(self.country_list, 0):  # 从0开始遍历
            country_dict[country_name] = idx  # 构造键字对，为国家编码；如：{'china': 1, 'japan': 2}
        return country_dict

    def idx2country(self, index):
        return self.country_list[index]  # 根据索引取出相应的国家名

    def getCountriesNum(self):
        return self.country_num  # 返回国家数


# 导入数据集
trainset = NameDataset(is_train_set=True)  # 训练集
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testset = NameDataset(is_train_set=False)  # 测试集
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

N_COUNTRY = trainset.getCountriesNum()  # 国家数


#
class RNNClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):  # bidirectional：单双向循环
        super(RNNClassifier, self).__init__()  # 构造函数
        self.hidden_size = hidden_size  # 网络输出维度
        self.n_layers = n_layers  # 层
        self.n_directions = 2 if bidirectional else 1  # 双向循环，输出的hidden是正向和反向hidden的拼接，所以要 *2

        self.embedding = torch.nn.Embedding(input_size, hidden_size)  # 嵌入层
        self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers, bidirectional=bidirectional)  # GRU循环神经网络
        self.fc = torch.nn.Linear(hidden_size * self.n_directions, output_size)  # 全连接层

    def _init_hidden(self, batch_size):  # 初始化h_0
        hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)  # 双向： *2
        return create_tensor(hidden)

    def forward(self, input, seq_lengths):
        # input shape : B x S -> S x B
        input = input.t()  # 转置
        batch_size = input.size(1)  # 计算batch_size

        hidden = self._init_hidden(batch_size)  # 获得h_0
        embedding = self.embedding(input)

        # pack them up
        gru_input = pack_padded_sequence(embedding, seq_lengths)  # 打包
        output, hidden = self.gru(gru_input, hidden)
        if self.n_directions == 2:  # 双向循环
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)  # 拼接hidden
        else:
            hidden_cat = hidden[-1]
        fc_output = self.fc(hidden_cat)  # 全连接层
        return fc_output


def name2list(name):
    arr = [ord(c) for c in name]  # 函数ord()返回每一个字母的ascii值
    return arr, len(arr)  # 返回元组


def make_tensors(names, countries):
    sequences_and_lengths = [name2list(name) for name in names]  # 元组
    name_sequences = [sl[0] for sl in sequences_and_lengths]  # 取名字，实为一组ascii码
    seq_lengths = torch.LongTensor([sl[1] for sl in sequences_and_lengths])  # LongTensor型，取长度
    countries = countries.long()

    # make tensor of name, BatchSize x SeqLen
    seq_tensor = torch.zeros(len(name_sequences), seq_lengths.max()).long()  # 初始化一个全零的tensor，行：名字数，列：最长的ascii名字
    for idx, (seq, seq_len) in enumerate(zip(name_sequences, seq_lengths), 0):  # 遍历
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)  # 将ascii码 依次输入到全零的tensor中（对应位置覆盖相应的ascii值，替代相应长度）

    # sort by length to use pack_padded_sequence
    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)  # 排序，依据序列长度降序
    seq_tensor = seq_tensor[perm_idx]
    countries = countries[perm_idx]

    return create_tensor(seq_tensor), create_tensor(seq_lengths), create_tensor(countries)


def create_tensor(tensor):  # 是否使用GPU
    if USE_GPU:
        device = torch.device("cuda:0")
        tensor = tensor.to(device)
    return tensor


def time_since(since):  # 计算程序运行的时间
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def trainModel(epoch):
    total_loss = 0
    for i, (names, countries) in enumerate(trainloader, 1):
        inputs, seq_lengths, target = make_tensors(names, countries)  # 生成符合尺寸大小的Tensor数据
        output = classifier(inputs, seq_lengths)  # 输入至网络训练
        loss = criterion(output, target)  # 计算损失
        optimizer.zero_grad()  # 梯度置0
        loss.backward()  # 反向传播
        optimizer.step()  # 优化参数

        total_loss += loss.item()
        if i % 10 == 0:
            print(f'[{time_since(start)}] Epoch {epoch}', end='')
            print(f'[{i * len(inputs)}/{len(trainset)}]', end='')
            print(f'loss={total_loss / (i * len(inputs))}')
    return total_loss


def testModel():
    correct = 0
    total = len(testset)
    print("evaluating trained model ...")
    with torch.no_grad():
        for i, (names, countries) in enumerate(testloader, 1):
            inputs, seq_lengths, target = make_tensors(names, countries)
            output = classifier(inputs, seq_lengths)
            pred = output.max(dim=1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        percent = '%.2f' % (100 * correct / total)
        print(f'Test set: Accuracy {correct}/{total} {percent}%')

    return correct / total


def main():
    if USE_GPU:
        device = torch.device("cuda:0")
        classifier.to(device)

    print("Training for %d epochs..." % N_EPOCHS)
    acc_list = []
    for epoch in range(1, N_EPOCHS + 1):
        # Train cycle
        trainModel(epoch)
        acc = testModel()
        acc_list.append(acc)

    epoch = np.arange(1, len(acc_list) + 1, 1)
    acc_list = np.array(acc_list)
    plt.plot(epoch, acc_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()


classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_COUNTRY, N_LAYER)  # 生成模型对象
criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失计算器
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)  # 优化器
start = time.time()  # 开始时间

main()
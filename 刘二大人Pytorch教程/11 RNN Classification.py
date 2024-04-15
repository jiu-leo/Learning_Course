# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/15 9:58
@Auth ： zmmmmm
@File ：11 RNN Classification.py
@IDE ：PyCharm
"""
import csv
import gzip
import time

import math

import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 参数设置
HIDDEN_SIZE = 100
BATCH_SIZE = 256
N_LAYER = 2
N_EPOCH = 100
N_CHARS = 128
USE_GPU = True

class NameDataset(Dataset):
    def __init__(self, is_train_set=True):
        filename = 'dataset/name/names_train.csv.gz' if is_train_set else 'dataset/name/names_test.csv.gz'
        with gzip.open(filename, 'rt') as f:
            reader = csv.reader(f)
            rows = list(reader)
        self.name = [row[0] for row in rows]
        self.len = len(self.name)
        self.countries = [row[1] for row in rows]
        self.country_list = list(sorted(set(self.countries)))
        self.country_dict = self.getCountryDict()
        self.country_num = len(self.country_list)


    def __getitem__(self, item):
        return self.name[item], self.country_dict[self.countries[item]]
    # self.country_dict[self.countries[item]]返回的是国家的索引，它是一个整数。
    # 当DataLoader迭代NameDataset时，它会收集这些整数，并最终将它们打包成一个张量。

    def __len__(self):
        return self.len

    def getCountryDict(self):
        country_dict = dict()
        for idx, country_name in enumerate(self.country_list,0):
            country_dict[country_name] = idx
        return country_dict

    def idx2country(self, item):
        return self.country_list[item]

    def getCountriesNum(self):
        return self.country_num


train_set = NameDataset(is_train_set=True)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
# 当你使用DataLoader加载数据时，它会按照指定的batch_size将数据集分成多个批次（batches），
# 并且每个批次的数据,整数数据默认会被转换成Tensor（张量）对象。

test_set = NameDataset(is_train_set=False)
test_loader = DataLoader(test_set,batch_size=BATCH_SIZE, shuffle=False)

N_COUNTRY = train_set.getCountriesNum()

class RNNClassifier(torch.nn.Module):
    def __init__(self,input_size,hidden_size,output_size,n_layers,bidirectional=True):
        super(RNNClassifier,self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1

        self.embeddings = torch.nn.Embedding(input_size,hidden_size)
        self.gru = torch.nn.GRU(hidden_size,hidden_size,n_layers,bidirectional=bidirectional)

        self.fc = torch.nn.Linear(hidden_size*self.n_directions*self.n_layers, output_size)

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers*self.n_directions,batch_size,self.hidden_size)
        return create_tensors(hidden)

    def forward(self, inputs, seq_length):
        inputs = inputs.t()
        batch_size = inputs.size(1)

        hidden = self._init_hidden(batch_size)
        embedding = self.embeddings(inputs)

        gru_input = pack_padded_sequence(embedding,seq_length)

        output, hidden = self.gru(gru_input, hidden)

        if self.n_directions == 2 and self.n_layers>=2:
            hidden_cat = torch.cat([h for h in hidden],dim=1)
        else:
            hidden_cat = hidden[-1]

        fc_output = self.fc(hidden_cat)

        return fc_output



classifier = RNNClassifier(N_CHARS,HIDDEN_SIZE,N_COUNTRY,N_LAYER)
classifier.to(device)

def name2list(name):
    arr = [ord(c) for c in name]
    return arr, len(arr)

def create_tensors(tensor):
    if USE_GPU:
        device = torch.device('cuda:0')
        tensor = tensor.to(device)

    return tensor



def make_tensors(names, countries):
    sequences_and_lengths = [name2list(name) for name in names]
    name_sequences = [s1[0] for s1 in sequences_and_lengths]
    seq_lengths = torch.LongTensor([s1[1] for s1 in sequences_and_lengths])
    countries = countries.long()

    #  batchsize*seqLen
    seq_tensor = torch.zeros(len(name_sequences),seq_lengths.max()).long()

    for idx,(seq,seq_len) in enumerate(zip(name_sequences,seq_lengths),0):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    seq_lengths, perm_idx = seq_lengths.sort(dim=0,descending=True)
    seq_tensor = seq_tensor[perm_idx]
    countries = countries[perm_idx]

    return create_tensors(seq_tensor),create_tensors(seq_lengths),create_tensors(countries)


def time_since(since):
    s =time.time() - since
    m = math.floor(s/60)
    s -= m*60
    return '%dm %ds' % (m, s)




criterion = torch.nn.CrossEntropyLoss()
criterion.to(device)
optimizer = torch.optim.Adam(classifier.parameters(),lr =0.001)


start = time.time()  # 开始时间

def trainModel(epoch):
    total_loss = 0
    for i, (names, countries) in enumerate(train_loader,1):
        inputs, seq_lengths, target = make_tensors(names,countries)

        output = classifier(inputs,seq_lengths)
        optimizer.zero_grad()
        loss = criterion(output,target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 10 == 0:
            print(f'[{time_since(start)}] Epoch {epoch}', end='')
            print(f'[{i * len(inputs)}/{len(train_set)}]', end='')
            print(f'loss={total_loss / (i * len(inputs))}')

    return total_loss


def testModel():
    correct = 0
    total = len(test_set)
    print("evaluating trained model ...")
    with torch.no_grad():
        for i, (names,contries) in enumerate(test_loader,1):
            inputs, seq_lengths, target = make_tensors(names,contries)
            output = classifier(inputs,seq_lengths)
            pred = output.max(dim=1,keepdim = True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
        percent = '%.2f' % (100 * correct / total)
        print(f'Test set: Accuracy {correct}/{total} {percent}%')

    return correct / total


def main():

    print("Training for %d epochs..." % N_EPOCH)
    acc_list = []
    for epoch in range(1, N_EPOCH+1):
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





if __name__ == '__main__':
    # classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_COUNTRY, N_LAYER)  # 生成模型对象
    # classifier.to(device)
    # criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失计算器
    # criterion.to(device)
    # optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)  # 优化器
    main()



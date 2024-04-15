# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/14 19:00
@Auth ： zmmmmm
@File ：11 Basic RNN02.py
@IDE ：PyCharm
"""

import torch

input_size = 4
hidden_size = 4
batch_size = 1
num_layers = 1
seq_len = 5

idx2char = ['e', 'h', 'l', 'o']

x_data = [1,0,2,2,3]
y_data = [3,1,2,3,2]

one_hot_lookup = [[1,0,0,0],
                  [0,1,0,0],
                  [0,0,1,0],
                  [0,0,0,1]]

x_one_hot = [one_hot_lookup[x] for x in x_data]

inputs = torch.Tensor(x_one_hot).view(-1,batch_size,input_size)
# print(inputs)

labels = torch.LongTensor(y_data) # (seqLen*batchsize,1)
# print(labels)

class Model2(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size,num_layers = 1):
        super(Model2,self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = torch.nn.RNN(input_size=self.input_size,hidden_size=self.hidden_size, num_layers=self.num_layers)

    def forward(self,inputs):
        hidden = torch.zeros(self.num_layers,self.batch_size,self.hidden_size)

        out,_ = self.rnn(inputs,hidden)
        return out.view(-1,self.hidden_size) # (seqLen*batchsize,hiddensize)

net = Model2(input_size,hidden_size,batch_size,num_layers)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

for epoch in range(15):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    _,idx = outputs.max(dim=1)
    print(f"Predicted: {''.join([idx2char[x] for x in idx])}", end='')
    print(f', Epoch{epoch+1}, loss = {loss.item():.3f}')
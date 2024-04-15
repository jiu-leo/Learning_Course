# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/14 20:21
@Auth ： zmmmmm
@File ：11 Basic RNN03.py
@IDE ：PyCharm
"""
import torch

num_class = 4
input_size = 4
hidden_size = 8
embedding_size = 10
batch_size = 1
num_layers = 2
seq_len = 5

idx2char = ['e', 'h', 'l', 'o']

x_data = [[1,0,2,2,3]]
y_data = [3,1,2,3,2]

inputs = torch.LongTensor(x_data) # (batch_size,seqLen)
labels = torch.LongTensor(y_data) # (batch_size*seqLen)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.emb = torch.nn.Embedding(input_size,embedding_size)
        self.rnn = torch.nn.RNN(input_size=embedding_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True)
        self.fc = torch.nn.Linear(hidden_size,num_class)

    def forward(self,x):
        hidden = torch.zeros(num_layers,x.size(0),hidden_size)
        x = self.emb(x)
        x, _=self.rnn(x,hidden)
        x = self.fc(x)
        return x.view(-1,num_class)

net = Model()

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
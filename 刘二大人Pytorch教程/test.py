# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/15 10:13
@Auth ： zmmmmm
@File ：test.py
@IDE ：PyCharm
"""
import csv
import gzip

import torch
from torch.utils.data import Dataset, DataLoader


# def read(is_train_set=True):
#     filename = 'dataset/name/names_train.csv.gz' if is_train_set else 'dataset/name/names_train.csv.gz'
#     with gzip.open(filename, 'rt') as f:
#         reader = csv.reader(f)
#         rows = list(reader)
#
#     return rows
#
# res = read()

BATCH_SIZE = 256

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

test_set = NameDataset(is_train_set=False)
test_loader = DataLoader(test_set,batch_size=BATCH_SIZE, shuffle=False)

def name2list(name):
    arr = [ord(c) for c in name]
    return arr, len(arr)


def make_tensors(names, countries):
    sequences_and_lengths = [name2list(name) for name in names]
    name_sequences = [s1[0] for s1 in sequences_and_lengths]
    seq_lengths = torch.LongTensor(s1[1] for s1 in sequences_and_lengths)
    countries = countries.long()
    return name_sequences, countries

if __name__ == '__main__':
    for i, (names, countries) in enumerate(train_loader, 1):
        name_sequence, countries = make_tensors(names, countries)
        print(name_sequence)
        print(type(name_sequence))
        print(countries)
        print(type(countries))
        break

# import torch
#
# tensor1 =torch.Tensor([[[1,2],[3,4],[5,6]],
#                       [[1,2],[3,4],[5,6]],
#                       [[1,2],[3,4],[5,6]]])
# print(tensor1.shape)
# print(tensor1)
#
# tensor2 =torch.Tensor([[[1,2],[3,4],[5,6]],
#                       [[1,2],[3,4],[5,6]],
#                       [[1,2],[3,4],[5,6]]])
#
# cat = torch.cat((tensor1,tensor2),dim=1)
# print(cat.shape)
# print(cat)
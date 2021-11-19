# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
from collections import defaultdict


class DNADataset(Dataset):
    
    def __init__(self, data, size):
        super(DNADataset, self).__init__()
        self.size = size
        self.x = np.zeros((len(data),size, 4))
        self.y = []

        for I in range(len(data)):
            self.y.append(data[I][1])
            if type(data[0][0])==str:
                for i in range(size):
                    seq = data[I][0].lower()
                    # one hot encoding
                    pos = "acgt".find(seq[i])
                    if pos >= 0:
                        self.x[I][i][pos] = 1
            else:
                self.x[I] = data[I][0]
        self.x = torch.FloatTensor(self.x)
        self.y = torch.FloatTensor(self.y)

        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
        
    # return a subset of dataset of given range
    def get_subset(self, start, end):
        
        return DNADataset([(self.x[i],self.y[i]) for i in range(start, end)], self.size)



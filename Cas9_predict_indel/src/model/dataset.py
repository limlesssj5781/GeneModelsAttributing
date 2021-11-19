# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
from collections import defaultdict


# build a simple class of text dataset
class DNADataset(Dataset):
    
    def __init__(self, data, size):
        super(TextDataset, self).__init__()
        self.size = size
        self.x = np.zeros((size, 4))
        self.y = []
        if data:
            for i in range(size):
                self.y.append(data[i][1])
                pos = "acgt".find(data[i][j])
                self.x[i, pos] = 1

        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
        
    # return a subset of dataset of given range
    def get_subset(self, start, end):
        subset = TextDataset(None, None, None, self.size)
        subset.x = self.x[start:end]
        subset.y = self.y[start:end]
        return subset





# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn as nn
from .BasicModule import BasicModule
from torch.autograd import Variable
from torch.nn import functional as F

class SimpleCNN(BasicModule):
    def __init__(self,args):
        super(ShallowCNN, self).__init__()
        max_len = args['max_len']
        dropout = args['dropout']
        out1 = args['out1']
        out2 = args['out2']
        self.params = args
        
        # self.k = args['k'] # for k max pooling
        self.kernel_num = args['kernel_num']
        self.dropout = nn.Dropout(dropout)
 
        
        # convolution layer with 6 filters: (3,4,5) * embedding_dimention
        self.conv1 = nn.Conv2d(1, self.kernel_num, (3, 4))
        self.conv2 = nn.Conv2d(1, self.kernel_num, (4, 4))
        self.conv3 = nn.Conv2d(1, self.kernel_num, (5, 4))
        self.conv4 = nn.Conv2d(1, self.kernel_num, (6, 4))

        self.fc1 = nn.Linear(4 * self.kernel_num, out1)
        self.fc2 = nn.Linear(out1, out2)
        self.fc3 = nn.Linear(out2, 1)

    def _conv_and_pool(self, x, conv):
        # x: (batch, 1, sentence_length, dim)
        x = conv(x)
        # x: (batch, kernel_num, H_out, 1)
        x = F.relu(x.squeeze(3))
        # x: (batch, kernel_num, H_out)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
#         x = self.kmax_pooling(x, 2, k=self.k)
#         x = x.view(x.size(0), x.size(1) * x.size(2))
        #  (batch, kernel_num * k)
        return x
    
    def setDropout(self, dropout):
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        # x: (batch, size)
        # x: (batch, sentence_length, embed_dim)
        # x = x.view(x.size(0),1, self.param['max_len'], self.param['dim'])
        x1 = self._conv_and_pool(x, self.conv1)  # (batch, kernel_num * k)
        x2 = self._conv_and_pool(x, self.conv2)  # (batch, kernel_num * k)
        x3 = self._conv_and_pool(x, self.conv3)  # (batch, kernel_num * k)
        x4 = self._conv_and_pool(x, self.conv4)  # (batch, kernel_num * k)

        
        x = torch.cat((x1, x2, x3, x4), 1)  # (batch, 6 * kernel_num * k)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        logit = F.log_softmax(x, dim=1)
        return logit
    
#     def kmax_pooling(self, x, dim, k):
#         index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
#         return x.gather(dim, index)

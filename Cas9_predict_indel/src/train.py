import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from collections import defaultdict
import pandas as pd

#
def test_scores(model, test_set, batch_size):
    tscores = []
    for inp, scores in DataLoader(test_set, batch_size=batch_size):
        inp = inp.unsqueeze(1)
        inp = Variable(inp)
        out = model(inp).view((-1))
        for v in out.detach().numpy():
            tscores.append(v)
    return tscores



def train(model, train_dataset, epochs, batch_size, val_set, cv=0.1, learning_rate=0.001, start = 0, end = 0):
    subset = train_dataset.get_subset(start, end)
    n_training_samples = len(subset) * (1-cv)
    n_val_samples = len(subset) * cv
    train_loader =torch.utils.data.DataLoader(subset, batch_size=batch_size,
                                              sampler=SubsetRandomSampler(
                                                  np.arange(n_training_samples, dtype=np.int64)
                                              ),
                                              num_workers=0)
    val_loader =torch.utils.data.DataLoader(subset, batch_size=100,
                                              sampler=SubsetRandomSampler(
                                                  np.arange(n_training_samples, n_training_samples + n_val_samples, dtype=np.int64)
                                              ), num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()


    print("Train %s samples."%n_training_samples)
    
    for e in range(epochs):
        # train loss
        epoch_train_loss = 0.0
        model.setDropout(model.params['dropout'])
        for inp, scores in train_loader:
            inp = inp.unsqueeze(1)
            inp = Variable(inp)
            out = model(inp).view((-1))

            scores = Variable(torch.FloatTensor(scores))
            loss = criterion(out, scores)
            epoch_train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
        # validation loss
#         epoch_val_loss = 0.0
#         for inp, labels in val_loader:
#             inp = inp.unsqueeze(1)
#             out = model(inp)        
#             loss = criterion(out, labels)
#             epoch_val_loss += loss


#         print(str(epoch_train_loss) + ' ')
#         print(str(epoch_val_loss.tolist()) + '\n')
        model.setDropout(0.0)
        print(e)
        tscores = test_scores(model, val_set, 400)
        corr = pd.DataFrame([[val_set[i][0], float(val_set[i][1]), tscores[i]] for i in range(len(tscores))], columns=['seq', 'endogenous', 'integrated'])\
    [['endogenous', 'integrated']].corr(method='spearman')['endogenous']['integrated']
        print(corr)
        if corr >= 0.68:
            print("Saving.")
            model.save("../model/cnn_3579_90-60-40-110_80-60_90_%s.model"%(round(corr, 3)))
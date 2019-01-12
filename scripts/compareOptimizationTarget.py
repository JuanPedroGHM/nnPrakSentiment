# coding: utf-8

# # RNTN using Dynamic Batching

# ## Import Statements

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim

from torchfold import Fold

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

import numpy as np

import pytreebank

import re
import time

from RNTN_v2 import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Training on {}'.format(device))

batchSize = 256
epochs = 3000

print_every = 10
devSet_every = 10
save_every = 50

saveFineAllPath = '../savedModels/d30/fineAll'
saveFineRootPath = '../savedModels/d30/fineRoot'

totalLoss = []
accuracyAll = []
accuracyRoot = []

net = RNTN(len(word2idx), d=30)
net.to(device)

loss_f = nn.NLLLoss() # Negative log likelihood loss
optimizer = optim.Adagrad(net.parameters(), weight_decay=0.001) # Adagrad with L2 regularization

## First optimize FineAll

checkpoint5000 = torch.load('../savedModels/d30/net_5000.pth')



net.load_state_dict(checkpoint5000)
net.eval()

start = time.time()
for e in range(1, epochs+1):

    train_iter = chunks(shuffle(bank['train']), batchSize)
    totalLoss.append(0)
    for batchIdx, batch in enumerate(train_iter):

        optimizer.zero_grad()

        res = allOutput(batch, net, device)
        error = loss_f(res[0], res[1])
        error.backward(); optimizer.step()
        totalLoss[-1] += error.item()


    if e % save_every == 0:
        torch.save(net.state_dict(), '{}/net_{}.pth'.format(saveFineAllPath, e))
    if e % print_every == 0:
        print('FineAll: Epoch {}: Total Loss = {}, Avg. Time/Epoch = {}'.format(e, totalLoss[-1],(time.time() - start) / print_every))
        start = time.time()
    if e % devSet_every == 0:
        with torch.no_grad():
            res = rootOutput(bank['dev'], net, device)
            accuracyAll.append(accuracy_score(torch.argmax(res[0], dim=1), res[1]))
            print('FineAll: Epoch {}: Root accuracy on the dev set = {}'.format(e, accuracy[-1]))

## Optimize FineRoot
net.load_state_dict(checkpoint5000)
net.eval()

start = time.time()
for e in range(1, epochs+1):

    train_iter = chunks(shuffle(bank['train']), batchSize)
    totalLoss.append(0)
    for batchIdx, batch in enumerate(train_iter):

        optimizer.zero_grad()

        res = rootOutput(batch, net, device)
        error = loss_f(res[0], res[1])
        error.backward(); optimizer.step()
        totalLoss[-1] += error.item()


    if e % save_every == 0:
        torch.save(net.state_dict(), '{}/net_{}.pth'.format(saveFineRootPath, e))
    if e % print_every == 0:
        print('FineRoot: Epoch {}: Total Loss = {}, Avg. Time/Epoch = {}'.format(e, totalLoss[-1],(time.time() - start) / print_every))
        start = time.time()
    if e % devSet_every == 0:
        with torch.no_grad():
            res = rootOutput(bank['dev'], net, device)
            accuracyRoot.append(accuracy_score(torch.argmax(res[0], dim=1), res[1]))
            print('FineRoot: Epoch {}: Root accuracy on the dev set = {}'.format(e, accuracy[-1]))


## Save date for plotting
accuracyData = np.cat([np.arange(10, 3001, 10), accuracyAll, accuracyRoot], dim = 0)
np.savetxt("../data/accuracyComp.csv", accuracyData, delimiter=",")


## Test on the test Set

with torch.no_grad():

    ## FineAll
    checkpointFineAll = torch.load('../savedModels/d30/fineAll/net_3000.pth')
    net.load_state_dict(checkpointFineAll)

    res = allOutput(bank['test'], net, device)
    accuracyRoot.append(accuracy_score(torch.argmax(res[0], dim=1), res[1]))
    print('FineAll: Phrase Accuracy on test set = {}'.format(e, accuracy[-1]))

    res = rootOutput(bank['test'], net, device)
    accuracyRoot.append(accuracy_score(torch.argmax(res[0], dim=1), res[1]))
    print('FineAll: Sentence Accuracy on test set = {}'.format(e, accuracy[-1]))

    ## FineRoot
    checkpointFineAll = torch.load('../savedModels/d30/fineRoot/net_3000.pth')
    net.load_state_dict(checkpointFineAll)

    res = allOutput(bank['test'], net, device)
    accuracyRoot.append(accuracy_score(torch.argmax(res[0], dim=1), res[1]))
    print('FineRoot: Phrase Accuracy on test set = {}'.format(e, accuracy[-1]))

    res = rootOutput(bank['test'], net, device)
    accuracyRoot.append(accuracy_score(torch.argmax(res[0], dim=1), res[1]))
    print('FineAll: Sentence Accuracy on test set = {}'.format(e, accuracy[-1]))

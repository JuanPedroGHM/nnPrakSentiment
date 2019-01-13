
# coding: utf-8

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
import pandas as pd


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Training on {}'.format(device))

# in[ ]:

net = RNTN(len(word2idx), d=30)
net.to(device)
loss_f = nn.NLLLoss() # Negative log likelihood loss
optimizer = optim.Adagrad(net.parameters(), weight_decay=0.001) # Adagrad with L2 regularization


# In[23]:


## Test on the test Set

with torch.no_grad():
    
    
    checkpoint = torch.load('../savedModels/d30/net_5000.pth')
    net.load_state_dict(checkpoint)
    net.eval()

    res = allOutput(bank['test'], net, device)
    accuracy_Phrase = accuracy_score(torch.argmax(res[0], dim=1), res[1])
    print('test set: Phrase Accuracy on test set = {}'.format(accuracy_Phrase))

    res = rootOutput(bank['test'], net, device)
    accuracy_Root = accuracy_score(torch.argmax(res[0], dim=1), res[1])
    print("test set: Sentence Accuracy on test set = {}'.format(accuracy_Root))


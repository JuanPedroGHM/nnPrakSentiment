
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


# In[ ]:


dictFile = open("../stanfordSentimentTreebank/dictionary.txt")
lines = dictFile.readlines()
exp = r'^(\S+)\|\d+$'
words = list(map(lambda x: x.replace('\\', ''),
            map(lambda x: x[0],
            filter(lambda x: len(x) > 0, 
            map(lambda line: re.findall(exp, line), 
            lines)))))
words.append('8 1/2')
words.append('2 1/2')
words.append('9 1/2')

word2idx = dict((word, number) for number, word in enumerate(words))


# In[ ]:


class RNTN(nn.Module): 
    def __init__(self, vocabularySize, classes = 5, d = 30):
        super(RNTN, self).__init__()
        self.d = d
        self.L = nn.Embedding(vocabularySize, d)
        self.W = nn.Linear(d * 2, d)
        self.Ws = nn.Linear(d,  classes)
        self.register_parameter('V', nn.Parameter(torch.rand(2 * d, 2 * d, d)))
        self.lSoftmax = nn.LogSoftmax(dim=1)
    
    def tensorProduct(self, phrase):
        result = torch.empty(phrase.shape[0], self.d)
        for i in range(self.d):
            result[:,i] = torch.sum(phrase * torch.mm(phrase, self.V[:,:,i]), dim = 1)
        return result
    
    def embed(self, inpt):
        return self.L(inpt)
    
    def sentiment(self, inpt):
        return self.lSoftmax(self.Ws(inpt))
    
    def node(self, leftPhrase, rightPhrase):
        phraseVec = torch.cat([leftPhrase, rightPhrase], dim=1)
        return torch.tanh(self.tensorProduct(phraseVec) + self.W(phraseVec))


# In[ ]:


def encodeTree(fold, tree):
    def encodeNode(node):
        if len(node.children) == 0:
            return fold.add('embed', word2idx[node.to_lines()[0]])
        else:
            return fold.add('node', encodeNode(node.children[0]), encodeNode(node.children[1]))
            
    encodedTree = encodeNode(tree)
    return fold.add('sentiment', encodedTree)


# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Training on {}'.format(device))


# ## Plots with d=30 on the devSet

# In[ ]:


path = '../savedModels/d30'
accuracy = []

net = RNTN(len(word2idx), d = 30)

for epoch in np.arange(100, 5001, 100):
    checkpoint = torch.load('{}/net_{}.pth'.format(path, epoch))
    net.load_state_dict(checkpoint)
    net.eval()
    with torch.no_grad():
        fold = Fold(cuda= device.type != 'cpu')
        allOutputs, allLabels = [], []
        for sentenceTree in bank['dev']:
            allOutputs.append(encodeTree(fold, sentenceTree))
            allLabels.append(sentenceTree.label)

        res = fold.apply(net, [allOutputs, allLabels])
        accuracy.append(accuracy_score(torch.argmax(res[0], dim=1), res[1]))
        print('Epoch {}: Accuracy on the dev set = {}'.format(e, accuracy[-1]))


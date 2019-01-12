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


# ## Loading and Pre-processing

# In[ ]:


dictFile = open("../stanfordSentimentTreebank/dictionary.txt", encoding='utf-8')
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


bank = pytreebank.load_sst("../stanfordSentimentTreebank/trees")

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


# ## Define RNTN

# In[ ]:


class RNTN(nn.Module):
    def __init__(self, vocabularySize, classes = 5, d = 30):
        super(RNTN, self).__init__()
        self.d = d
        self.L = nn.Embedding(vocabularySize, d)
        self.W = nn.Linear(d * 2, d)
        self.Ws = nn.Linear(d,  classes)
        self.register_parameter('V', nn.Parameter(torch.rand(2 * d, 2 * d, d).cuda()))
        self.lSoftmax = nn.LogSoftmax(dim=1)

    def tensorProduct(self, phrase):
        result = torch.empty(phrase.shape[0], self.d).cuda()
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


# ## Dynamic Batching with torchfold
def encodeTree4Training(fold, tree):
    allOutputs, allLabels = [], []

    def encodeNodeTraining(node):
        if len(node.children) == 0:
            wordVector = fold.add('embed', word2idx[node.to_lines()[0]])
            allOutputs.append(fold.add('sentiment', wordVector))
            allLabels.append(node.label)
            return wordVector
        else:
            phraseVector = fold.add('node', encodeNodeTraining(node.children[0]), encodeNodeTraining(node.children[1]))
            allOutputs.append(fold.add('sentiment', phraseVector))
            allLabels.append(node.label)
            return phraseVector

    encodedTree = encodeNodeTraining(tree)
    return allOutputs, allLabels

def encodeTree(fold, tree):
    def encodeNode(node):
        if len(node.children) == 0:
            return fold.add('embed', word2idx[node.to_lines()[0]])
        else:
            return fold.add('node', encodeNode(node.children[0]), encodeNode(node.children[1]))

    encodedTree = encodeNode(tree)
    return fold.add('sentiment', encodedTree)

def fineGrainedOutput(batch, device):
    fold = Fold(cuda= device.type != 'cpu')
    allOutputs, allLabels = [], []
    for sentenceTree in batch:
        sentenceOutputs, sentenceLabels = encodeTree4Training(fold, sentenceTree)
        allOutputs.extend(sentenceOutputs)
        allLabels.extend(sentenceLabels)

    return fold.apply(net, [allOutputs, allLabels])

def rootOutput(batch, device):
    fold = Fold(cuda= device.type != 'cpu')
    allOutputs, allLabels = [], []
    for sentenceTree in bank['dev']:
        allOutputs.append(encodeTree(fold, sentenceTree))
        allLabels.append(sentenceTree.label)

    return fold.apply(net, [allOutputs, allLabels])
# ## Training

# In[ ]:

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Training on {}'.format(device))

    batchSize = 256
    epochs = 4000

    print_every = 10
    devSet_every = 10
    save_every = 50

    savePath = '../savedModels/v2/'

    totalLoss = []
    accuracy = []

    net = RNTN(len(word2idx))
    net.to(device)

    loss_f = nn.NLLLoss() # Negative log likelihood loss
    optimizer = optim.Adagrad(net.parameters(), weight_decay=0.001) # Adagrad with L2 regularization

    start = time.time()
    for e in range(1, epochs+1):

        train_iter = chunks(shuffle(bank['train']), batchSize)
        totalLoss.append(0)
        for batchIdx, batch in enumerate(train_iter):

            optimizer.zero_grad()

            res = fineGrainedOutput(batch, device)
            error = loss_f(res[0], res[1])
            error.backward(); optimizer.step()
            totalLoss[-1] += error.item()


        if e % save_every == 0:
            torch.save(net.state_dict(), '{}/{}/net_{}.pth'.format(savePath, d, e))
        if e % print_every == 0:
            print('Epoch {}: Total Loss = {}, Avg. Time/Epoch = {}'.format(e, totalLoss[-1],(time.time() - start) / print_every))
            start = time.time()
        if e % devSet_every == 0:
            with torch.no_grad():
                res = rootOutput(bank['dev'], device)
                accuracy.append(accuracy_score(torch.argmax(res[0], dim=1), res[1]))
                print('Epoch {}: Accuracy on the dev set = {}'.format(e, accuracy[-1]))

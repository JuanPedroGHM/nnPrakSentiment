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
def encodeTree(fold, tree, allPhrases = True):
    allOutputs, allLabels = [], []

    def encodeNode(node):
        if len(node.children) == 0:
            wordVector = fold.add('embed', word2idx[node.to_lines()[0]])

            if allPhrases:
                allOutputs.append(fold.add('sentiment', wordVector))
                allLabels.append(node.label)

            return wordVector
        else:
            phraseVector = fold.add('node', encodeNode(node.children[0]), encodeNode(node.children[1]))
            if allPhrases:
                allOutputs.append(fold.add('sentiment', phraseVector))
                allLabels.append(node.label)
            return phraseVector

    encodedTree = encodeNode(tree)
    if not allPhrases:
        allOutputs.append(fold.add('sentiment', encodedTree))
        allLabels.append(tree.label)
    return allOutputs, allLabels

def foldForward(batch, net, device, allPhrases = True):
    fold = Fold(cuda= device.type != 'cpu')
    allOutputs, allLabels = [], []
    for sentenceTree in batch:
        sentenceOutputs, sentenceLabels = encodeTree(fold, sentenceTree, allPhrases)
        allOutputs.extend(sentenceOutputs)
        allLabels.extend(sentenceLabels)

    return fold.apply(net, [allOutputs, allLabels])


# ## Helper functions
def getAccuracyScores(net, dataset, device, modelFile = None):

    if modelFile:
        checkpoint = torch.load(modelFile)
        net.load_state_dict(checkpoint)
        net.eval()

    with torch.no_grad():
        [outputs, labels] = foldForward(dataset, net, device, allPhrases = True)
        fineAllAcc = accuracy_score(torch.argmax(outputs, dim=1), labels)

        [outputs, labels] = foldForward(dataset, net, device, allPhrases = False)
        fineRootAcc = accuracy_score(torch.argmax(outputs, dim=1), labels)
        return fineAllAcc, fineRootAcc

# ## Training

# In[ ]:

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Training on {}'.format(device))

    net = RNTN(len(word2idx), d=52)
    net.to(device)

    loss_f = nn.NLLLoss() # Negative log likelihood loss
    optimizer = optim.Adagrad(net.parameters(), weight_decay=0.001) # Adagrad with L2 regularization

    savePath = '../savedModels/d{}'.format(net.d)

    batchSize = 512 
    epochs = 5000

    print_every = 50 
    devSet_every = 50 
    save_every = 500 

    totalLoss = []
    allAccArr = []
    rootAccArr = []

    start = time.time()
    for e in range(1, epochs+1):

        train_iter = chunks(shuffle(bank['train']), batchSize)
        totalLoss.append(0)
        for batchIdx, batch in enumerate(train_iter):

            optimizer.zero_grad()

            [outputs, labels] = foldForward(batch, net, device, allPhrases = True)
            error = loss_f(outputs, labels)
            error.backward(); optimizer.step()
            totalLoss[-1] += error.item()

        if e % save_every == 0:
            torch.save(net.state_dict(), '{}/net_{}.pth'.format(savePath, e))
        if e % print_every == 0:
            print('Epoch {}: Total Loss = {}, Avg. Time/Epoch = {}'.format(e, totalLoss[-1],(time.time() - start) / print_every))
            start = time.time()
        if e % devSet_every == 0:
            with torch.no_grad():
                allAcc, rootAcc = getAccuracyScores(net, bank['dev'], device)
                print('Epoch {}: FineAll Accuracy on the dev set = {}'.format(e, allAcc))
                print('Epoch {}: FineRoot Accuracy on the dev set = {}'.format(e, rootAcc))
                allAccArr.append(allAcc); rootAccArr.append(rootAcc)

    accuracyData = np.vstack([np.arange(devSet_every, epochs + 1, devSet_every), allAccArr, rootAccArr])
    np.savetxt("../data/trainingD{}DevAcc.csv".format(net.d), accuracyData, delimiter=",")
    np.savetxt("../data/trainingD{}Loss.csv".format(net.d), totalLoss, delimiter=",")

from RNTN_v2 import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Training on {}'.format(device))

batchSize = 512
epochs = 1000

print_every = 10
devSet_every = 10
save_every = 50

saveFineAllPath = '../savedModels/d30/fineAll'
saveFineRootPath = '../savedModels/d30/fineRoot'

totalLoss = []
accuracyAll = []
accuracyRoot = []

net = RNTN(device, len(word2idx), d=30)
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

        [outputs, labels] = foldForward(batch, net, device, allPhrases = True)
        error = loss_f(outputs, labels)
        error.backward(); optimizer.step()
        totalLoss[-1] += error.item()


    if e % save_every == 0:
        torch.save(net.state_dict(), '{}/net_{}.pth'.format(saveFineAllPath, e))
    if e % print_every == 0:
        print('FineAll: Epoch {}: Total Loss = {}, Avg. Time/Epoch = {}'.format(e, totalLoss[-1],(time.time() - start) / print_every))
        start = time.time()
    if e % devSet_every == 0:
        with torch.no_grad():
            allAcc, rootAcc = getAccuracyScores(net, bank['dev'], device)
            accuracyAll.append(rootAcc)
            print('FineAll: Epoch {}: Root accuracy on the dev set = {}'.format(e, accuracyAll[-1]))

## Optimize FineRoot
print('Starting to optimize FineRoot')
net.load_state_dict(checkpoint5000)
net.eval()

start = time.time()
for e in range(1, epochs+1):

    train_iter = chunks(shuffle(bank['train']), batchSize)
    totalLoss.append(0)
    for batchIdx, batch in enumerate(train_iter):

        optimizer.zero_grad()

        [outputs, labels] = foldForward(batch, net, device, allPhrases = False)
        error = loss_f(outputs, labels) 
        error.backward(); optimizer.step()
        totalLoss[-1] += error.item()


    if e % save_every == 0:
        torch.save(net.state_dict(), '{}/net_{}.pth'.format(saveFineRootPath, e))
    if e % print_every == 0:
        print('FineRoot: Epoch {}: Total Loss = {}, Avg. Time/Epoch = {}'.format(e, totalLoss[-1],(time.time() - start) / print_every))
        start = time.time()
    if e % devSet_every == 0:
        with torch.no_grad():
            allAcc, rootAcc = getAccuracyScores(net, bank['dev'], device)
            accuracyRoot.append(rootAcc)
            print('FineRoot: Epoch {}: Root accuracy on the dev set = {}'.format(e, accuracyRoot[-1]))


## Save date for plotting
accuracyData = np.vstack([np.arange(10, 1001, 10), accuracyAll, accuracyRoot])
np.savetxt("../data/accuracyComp.csv", accuracyData, delimiter=",")

with torch.no_grad():

    ## FineAll
    allAcc, rootAcc = getAccuracyScores(net, bank['test'], device, '../savedModels/d30/fineAll/net_1000.pth')
    print('FineAll: Phrase Accuracy on test set = {}'.format(e, allAcc))
    print('FineAll: Sentence Accuracy on test set = {}'.format(e, rootAcc))

    ## FineRoot
    allAcc, rootAcc = getAccuracyScores(net, bank['test'], device, '../savedModels/d30/fineRoot/net_1000.pth')
    print('FineRoot: Phrase Accuracy on test set = {}'.format(e, allAcc))
    print('FineRoot: Sentence Accuracy on test set = {}'.format(e, rootAcc))

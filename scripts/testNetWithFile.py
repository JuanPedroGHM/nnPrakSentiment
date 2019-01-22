from RNTN_v2 import *
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test the model using the weights on the selected files')
    parser.add_argument('file', help='file path')

    args = parser.parse_args()
    filePath = args['file']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using {}'.format(device))

    # in[ ]:
    net = RNTN(device, len(word2idx), d=30)
    net.to(device)

    print('Testing accuracy with saved model: {}'.format(filePath))
    fineAllAcc, fineRootAcc = getAccuracyScores(net, bank['test'], device, filePath)
    print('Phrase Accuracy on test set = {}'.format(e, fineAllAcc))
    print('Sentence Accuracy on test set = {}'.format(e, fineRootAcc))

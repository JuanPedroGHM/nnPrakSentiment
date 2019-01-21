# Prak NN: Sentiment Analysis

Our model is a direct implementation of the Recurrent Neural Tensor Network (RNTN) from the Stanford [paper](https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf) using PyTorch, but using the [torchfold](https://github.com/nearai/torchfold) library to use [dynamic batching with computational graphs](https://arxiv.org/abs/1702.02181) to obtain faster results when training. We also rely on the pytreebank library to load the Stanford Sentiment Treebank. 

The project is structure like this:

- stanfordSentimentTreebank: contains the original dataset and the files for pytreebank
- scripts
    - RNTN_v2.py contains the implementation of the model and some helper functions. The other scripts and some notebooks rely on this file to load the model. This version uses torchfold to feed the sentence trees to the pytorch model in an efficient manner. 
    - testNetWithFile.py is a quick cmd-line script to test saved weights
    - compareOtimizationTarget trains the network from a checkpoint with two different optimization targets. FineAll tries to predict correctly all the n-grams on a sentence. FineRoot tries to optimize the sentiment prediction of just the whole sentence, and only the error on the root node of the sentence tree is backpropagated on the network.
- notebooks: 
    - RNTN-v1.0.ipynb: It contains our first implementation of the RNTN. This implementation doesn't use batching and each sentence is forwarded sequentially. It is very slow.
    - Plots.ipynb: Notebook where the results are plotted. Uses RNTN_v2.py
- savedModels:
    - All the models we saved with different embedding sizes and optimization targets
- data: 
    - Some csv files where the plotting data is saved
- torchfold: Contains the edited torchfold library, because of some bugs we had to edit the main file. 


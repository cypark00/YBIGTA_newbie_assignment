from typing import Literal


device = "cpu"
d_model = 256

# Word2Vec
window_size = 5
method: Literal["cbow", "skipgram"] = "cbow"
lr_word2vec = 1e-03
num_epochs_word2vec = 15

# GRU 
hidden_size = 256
num_classes = 4
lr = 5e-03
num_epochs = 200
batch_size = 16

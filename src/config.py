import torch

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Date
DATA_PATH = "./data/tiny_corpus.txt"
BLOCK_SIZE = 64 # how many characters of context the model sees
BATCH_SIZE = 64

# Model hyperparameters 
VOCAB_SIZE = None   # we'll fill this dynamically once we build the tokenizer
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2
FFN_DIM = 4 * EMBED_DIM
DROPOUT = 0.1

# TRAINING
LEARNING_RATE = 3E-4
NUM_EPOCHS = 10
RANDOM_SEED = 42
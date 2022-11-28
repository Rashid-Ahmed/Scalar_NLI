# %%
import torch
import os
from utils.train_nn import train_model

BATCH_SIZE = 32
EPOCHS = 10
STEP_SIZE = 0.0001
MODEL = 'microsoft/deberta-v3-large'
DATA_LEN = 940340
DATA_DIR = 'data'
DATA_FILE_NAME = 'nli_dataset.csv'
CHECKPOINT_DIR = 'checkpoints'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_model(MODEL , DATA_LEN , CHECKPOINT_DIR, device, os.path.join(DATA_DIR, DATA_FILE_NAME) ,EPOCHS, STEP_SIZE, BATCH_SIZE, load_model = False)



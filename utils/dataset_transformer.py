import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class NLIDataset(Dataset):
  
  def __init__(self, MODEL_TYPE, csv_file, chunk_size, len_dataset, transform = None):
    """
    Args:
        csv_file (string): Path to the csv file.
        chunk_size: Batch size
        len_dataset: total number of rows in the dataset
        vector_dim: Dimension of our word embeddings
        train(optional): Do we train our embedding Model
        transform (optional): Optional transform to be applied
            on a sample.
    """
    self.csv_file = csv_file
    self.chunk_size = chunk_size
    self.len_dataset = len_dataset//chunk_size
    self.reader = pd.read_csv(self.csv_file, chunksize = self.chunk_size)
    self.tokenizer =  AutoTokenizer.from_pretrained(MODEL_TYPE)
    
  def __len__(self):
    return self.len_dataset
  
  def __getitem__(self, index):
    offset = index * self.chunk_size
    batch = next(self.reader)
    sentences1 = batch['sentence1'].tolist()
    sentences2 = batch['sentence2'].tolist()
    targets = batch['gold_label']
    
    #getting targets from 0 - 4 because classifier predicts outputs starting from 0 not 1
    targets[targets == 'contradiction'] = 0
    targets[targets == 'neutral'] = 2
    targets[targets == 'entailment'] = 1
    #Converting sentences into embedding vectors
    embeddings = self.tokenizer(sentences1, sentences2, padding=True, truncation=True, return_tensors='pt')


    return embeddings, targets.astype(int)

  

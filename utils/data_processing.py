import pandas as pd
import os
from utils.load_NLI import load_nli_data, get_dataframe

def interesting_index(dataframe_interesting):
    index_mnli = dataframe_interesting['index'][dataframe_interesting['index'].str[-1] == 'm'].str[:-1]
    index_snli = dataframe_interesting['index'][dataframe_interesting['index'].str[-1] == 's'].str[:-1]
    return index_mnli, index_snli

def remove_controls(DATA_DIR, mnli_data, snli_data, mnli_index, snli_index):
    mnli_data = mnli_data.drop(index = mnli_index)
    snli_data = snli_data.drop(index = snli_index)
    mnli_data.to_csv(os.path.join(DATA_DIR, 'mnli_removed.csv'), index = False)
    snli_data.to_csv(os.path.join(DATA_DIR, 'snli_removed.csv'), index = False)

def nli_dataframe(DATA_DIR):

    mnli_data = load_nli_data(os.path.join(DATA_DIR, 'multinli.jsonl'))
    mnli_data = get_dataframe(mnli_data)
    mnli_data['index'] = mnli_data.index.astype('str') +'m'

    snli_data = load_nli_data(os.path.join(DATA_DIR, 'snli_1.0_train.jsonl'))
    snli_data = get_dataframe(snli_data)
    snli_data['index'] = snli_data.index.astype('str') +'s'
    
    return mnli_data, snli_data

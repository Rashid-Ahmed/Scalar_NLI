import json
import random
import pandas as pd

LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
    "hidden": 0
}

def load_gradables(path):
  adjectives = []
  with open(path) as f:
    for line in f:
      adjectives.append(line)
  return adjectives

def load_nli_data(path, snli=False):
    """
    Load MultiNLI or SNLI data.
    If the "snli" parameter is set to True, a genre label of snli will be assigned to the data. 
    """
    data = []
    with open(path) as f:
        for line in f:
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in LABEL_MAP:
                continue
            #loaded_example["label"] = LABEL_MAP[loaded_example["gold_label"]]
            data.append(loaded_example)
        random.seed(1)
        random.shuffle(data)
    return data  

def get_dataframe(nli_data):
    data = pd.DataFrame(nli_data)
    data = data.drop(columns = ['annotator_labels', 'genre', 'pairID', 'promptID',
            'sentence1_binary_parse', 'sentence1_parse', 
        'sentence2_binary_parse', 'sentence2_parse', 'captionID'], errors = 'ignore')
    return data
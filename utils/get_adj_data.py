import pandas as pd
import string
import os
def get_adj_sentences(nli_data, synonyms): 

    sentence1 = []
    sentence2 = []
    label = []
    premise_word = []
    hypothesis_word = []
    index = []
    i = 0
    for list in synonyms:
        i = i + 1
        for j in range(len(list)):
            temp_df = nli_data[nli_data['sentence1'].str.contains('\\b' + list[j] + '\\b')]
            for k in range(len(list)):
                if j!=k:
                    x = temp_df[temp_df['sentence2'].str.contains('\\b' + list[k] + '\\b')]
                    sentence1.extend(x['sentence1'])
                    sentence2.extend(x['sentence2'])
                    label.extend(x['gold_label'])
                    premise_word.extend([list[j]] * x.shape[0])
                    hypothesis_word.extend([list[k]] * x.shape[0])
                    index.extend(x['index'])

    adjective_pairs = pd.DataFrame(zip(sentence1, sentence2, premise_word, hypothesis_word, index, label), columns=['sentence1', 'sentence2', 'adj_premise', 'adj_hypothesis', 'index', 'label'])
    adjective_pairs = clean_data(adjective_pairs)
    adjective_pairs.to_csv(os.path.join('data','adjective_pairs_cleaned.csv'))
    return adjective_pairs




def clean_data(adjective_pairs):
    
    translator = str.maketrans('', '', string.punctuation)
    
    adjective_pairs['sentence1'] = adjective_pairs['sentence1'].str.split().str.join(' ')
    adjective_pairs['sentence1'] = adjective_pairs['sentence1'].str.translate(translator)
    adjective_pairs['sentence1'] = adjective_pairs['sentence1'].str.lower()

    adjective_pairs['sentence2'] = adjective_pairs['sentence2'].str.split().str.join(' ')
    adjective_pairs['sentence2'] = adjective_pairs['sentence2'].str.translate(translator)
    adjective_pairs['sentence2'] = adjective_pairs['sentence2'].str.lower()
    
    return adjective_pairs

def adj_list(adjective_pairs):
    
    premise_adj_index = []
    hyp_adj_index = []
    str_list_premise = adjective_pairs['sentence1'].str.split()
    str_list_hypothesis= adjective_pairs['sentence2'].str.split()
    for i in range(len(str_list_premise)):
        try:
            premise_adj_index.append(str_list_premise[i].index(adjective_pairs['adj_premise'][i]))
        except:
            premise_adj_index.append(-1)
            
    for i in range(len(str_list_hypothesis)):
        try:
            hyp_adj_index.append(str_list_hypothesis[i].index(adjective_pairs['adj_hypothesis'][i]))
        except:
            hyp_adj_index.append(-1)
            
    return premise_adj_index, hyp_adj_index

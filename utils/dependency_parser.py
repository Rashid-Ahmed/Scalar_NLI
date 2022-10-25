import stanza
import fasttext
from utils.math_custom import cos_similarity
import os 

def process_adj_dependencies(adjective_pairs, premise_adj_index, hyp_adj_index):

    EMBEDDING_MODEL_PATH = 'D:\work\DFKI\Sentiment Analyzer\Data'
    THRESHOLD = 0.5
    
    stanza.download('en') # download English model
    nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')
    
    
    model_train = fasttext.load_model(os.path.join(EMBEDDING_MODEL_PATH, 'cc.en.300.bin'))
    index_list = []
    special_indices = []
    words_close_premise = []
    words_close_hypothesis = []
    for i in range((adjective_pairs.shape[0])):
        sentence_premise = nlp(adjective_pairs['sentence1'][i]).to_dict()[0]
        sentences_hypothesis = nlp(adjective_pairs['sentence2'][i]).to_dict()[0]    
        if premise_adj_index[i]!= -1 and hyp_adj_index[i]!= -1:
            if (sentence_premise[premise_adj_index[i]]['deprel'] == 'amod') and (sentences_hypothesis[hyp_adj_index[i]]['deprel'] == 'amod'):
                if sentence_premise[premise_adj_index[i]]['head']!=0 and sentences_hypothesis[hyp_adj_index[i]]['head']!=0:
                    word1 = sentence_premise[sentence_premise[premise_adj_index[i]]['head'] - 1]['text']
                    word2 = sentences_hypothesis[sentences_hypothesis[hyp_adj_index[i]]['head'] - 1]['text']               
                    if (cos_similarity(model_train, word1, word2, THRESHOLD)):
                        index_list.append(i)
                        words_close_premise.append(word1)
                        words_close_hypothesis.append(word2)

                        
            elif (sentence_premise[premise_adj_index[i]]['deprel'] == 'amod'):
                if sentence_premise[premise_adj_index[i]]['head']!=0:
                    for word_hypothesis in sentences_hypothesis:
                        if word_hypothesis['deprel'] == 'nsubj' and word_hypothesis['head'] == (hyp_adj_index[i] + 1):
                            word2 = word_hypothesis['text']
                            word1 = sentence_premise[sentence_premise[premise_adj_index[i]]['head'] - 1]['text']
                            if (cos_similarity(model_train, word1, word2, THRESHOLD)):
                                index_list.append(i)
                                words_close_premise.append(word1)
                                words_close_hypothesis.append(word2)
                                break
                    
                    
                    
            elif (sentences_hypothesis[hyp_adj_index[i]]['deprel'] == 'amod'):
                if sentences_hypothesis[hyp_adj_index[i]]['head']!=0:
                    for word in sentence_premise:
                        if word['deprel'] == 'nsubj' and word['head'] == (premise_adj_index[i] + 1):
                            word1 = word['text']
                            word2 = sentences_hypothesis[sentences_hypothesis[hyp_adj_index[i]]['head'] - 1]['text']  
                            
                            if (cos_similarity(model_train, word1, word2, THRESHOLD)):
                                index_list.append(i)
                                words_close_premise.append(word1)
                                words_close_hypothesis.append(word2)
                                break
            
            else:
                for word in sentence_premise:
                    if word['deprel'] == 'nsubj' and word['head'] == (premise_adj_index[i] + 1):
                        for word_hypothesis in sentences_hypothesis:
                            if word_hypothesis['deprel'] == 'nsubj' and word_hypothesis['head'] == (hyp_adj_index[i] + 1):
                                
                                word1 = word['text']
                                word2 = word_hypothesis['text']               
                                if (cos_similarity(model_train, word1, word2, THRESHOLD)):
                                    index_list.append(i)
                                    special_indices.append(i)
                                    words_close_premise.append(word1)
                                    words_close_hypothesis.append(word2)
                                    break
                                
    adjective_pairs = adjective_pairs.iloc[index_list]
    adjective_pairs['word_connected_premise'] = words_close_premise
    adjective_pairs['word_connected_hypothesis'] = words_close_hypothesis                            
    
    return adjective_pairs
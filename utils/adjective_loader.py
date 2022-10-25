import copy
import pandas as pd

def adjective_scale(data, adjective_scale):
    gradable_adjectives = data[data['sentence1'].str.contains(adjective_scale[0])]
    changed_scale = pd.DataFrame(copy.deepcopy(gradable_adjectives['sentence1']))
    changed_scale['sentence2'] = gradable_adjectives['sentence1']
    changed_scale['sentence2'] = changed_scale['sentence2'].str.replace(adjective_scale[0], adjective_scale[1]) 
    
def scale_selected_adjectives(adjective, changed_scale):
    if adjective == ['big', 'huge']:
        sentence_locs = [2, 8, 13, 14, 627, 18, 600, 19, 1049, 21, 22, 25, 666, 26, 28, 29, 30, 33, 35, 39, 44, 45, 46, 48, 50, 73, 79, 92, 211, 212, 214,
                 263, 287, 382, 481, 572, 577, 580, 614, 620, 622, 632, 673, 1046, 1097, 1106, 3116, 3126, 3419, 3427]
        Semanticlabels =  [0,0,0,0,1,0,1,0,1,0,1,0,0,0,1,0,1,0,33,33,1,0,33,33,33,33,1,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,33,33,1,0,33,33,1,0,1,0,1,0,1,0,1,0,0,0]
        Pragmaticlabels = [0,0,0,0,2,2,2,2,2,2,2,2,0,0,2,2,2,2,33,33,2,2,33,33,33,33,2,2,0,0,0,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,0,0,33,33,2,2,33,33,2,2,2,2,2,2,2,2,2,2,0,0]
        Inference = pd.DataFrame(columns = ["Premise", "Replaced Sentence", "Semantic label", "Pragmatic label"], index=range(len(sentence_locs) * 2))
        for i in range (len(sentence_locs)):
            Inference.iloc[i*2] = [changed_scale['sentence1'].iloc[sentence_locs[i]], changed_scale['sentence2'].iloc[sentence_locs[i]], "S", "D"]
            Inference.iloc[i*2 + 1] = [changed_scale['sentence2'].iloc[sentence_locs[i]], changed_scale['sentence1'].iloc[sentence_locs[i]], "S", "D"]

            Inference["Semantic label"][:len(Semanticlabels)] = Semanticlabels
            Inference["Pragmatic label"][:len(Pragmaticlabels)] = Pragmaticlabels

            Inference["Semantic label"][Inference["Semantic label"] == 0] = "Entailment"
            Inference["Semantic label"][Inference["Semantic label"] == 1] = "Neutral"
            Inference["Semantic label"][Inference["Semantic label"] == 2] = "Contradiction"
            Inference["Pragmatic label"][Inference["Pragmatic label"] == 0] = "Entailment"
            Inference["Pragmatic label"][Inference["Pragmatic label"] == 1] = "Neutral"
            Inference["Pragmatic label"][Inference["Pragmatic label"] == 2] = "Contradiction"

            drop_indices = [18, 19, 22, 23,24,25,50,51,54,55]
            print (len(Semanticlabels) - len(drop_indices))
            Inference = Inference.drop(index = drop_indices)
            Inference = Inference.iloc[:58]
            
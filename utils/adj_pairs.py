import pandas as pd
def load_adjs(PATH):
        synonyms = []
        synonym_pair = []
        antonyms = []
        current_word = ''
        file = open(PATH)
        for line in file:
                line = line.replace(',', '')
                line = line.split()
                if line[0] =="===":
                        synonym_pair = list(dict.fromkeys(synonym_pair))
                        synonyms.append(synonym_pair)
                        antonyms.append(current_word)
                        synonym_pair = []
                        if line [-1] == 'END':
                                break
                        if line[-1] == '**':
                                synonym_pair.append(line[-2])
                                current_word = line[-3]
                                
                        elif line[1] == '**':
                                synonym_pair.append(line[2])
                                current_word = line[3]

                else:
                        synonym_pair.extend(line)
        return synonyms[1:], antonyms[1:]



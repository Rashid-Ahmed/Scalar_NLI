from scipy import spatial

def cos_similarity(model_train, word1, word2, Threshold):
    vector1 = model_train.get_word_vector(word1)
    vector2 = model_train.get_word_vector(word2)
    distance = spatial.distance.cosine(vector1, vector2)
    if distance<Threshold:
        return True
    else:
        return False
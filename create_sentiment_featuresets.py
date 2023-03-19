import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import random
import pickle
from collections import Counter
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
hm_lines = 100000

def create_lexicon(pos, neg):

    lexicon=[]
    with open(pos, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            l = l.decode('utf8')
            all_words = word_tokenize(l)
            lexicon += list(all_words)

    with open(neg, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            l = l.decode('utf8')
            all_words = word_tokenize(l)
            lexicon += list(all_words)

    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)
    l2=[]
    for w in w_counts:
        if 1000 > w_counts[w] > 50:
            l2.append(w)
    print(len(l2))
    return l2

def sample_handling(sample, lexicon, classification):
    featureset = []

    with open(sample, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            l = l.decode('utf8')
            currents_words = word_tokenize(l.lower())
            currents_words = [lemmatizer.lemmatize(i) for i in currents_words]
            features = np.zeros(len(lexicon))
            for word in currents_words:
                if word.lower() in lexicon:
                    features[lexicon.index(word.lower())] += 1

            features = list(features)
            featureset.append([features, classification])
    return featureset

def create_featuresets_and_labels(pos, neg, test_size=0.1):
    lexicon = create_lexicon(pos, neg)
    features = []
    features += sample_handling(pos, lexicon, [1,0])
    features += sample_handling(neg, lexicon, [0,1])
    random.shuffle(features)
    features = np.array(features)

    testing_size = int(test_size*len(features))
    print(features.shape)
    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])
    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])

    return train_x, train_y, test_x, test_y

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = create_featuresets_and_labels('pos.txt', 'neg.txt')

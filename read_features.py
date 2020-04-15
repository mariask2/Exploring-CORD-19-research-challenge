from gensim.models import KeyedVectors
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np

OUTSIDE_LABEL = "O"
INSIDE_LABEL = "I"
default_vector = [0] * 200

X, y = load_iris(return_X_y=True)
#print(X)
#print(y)
#clf = LogisticRegression(random_state=0, max_iter=1000).fit(X, y)
#print(clf.predict(X[:2, :]))



def get_vectors(words, word2vec_model):
    return_vectors = []
    for word in words:
        #if word in word2vec_model:
        #    vector = word2vec_model[word]
        #else:
        vector = default_vector
        return_vectors.append(vector)
    np_return_vectors = np.array([np.array(ti) for ti in return_vectors])
    return np_return_vectors
    

def read_data(file_name):
    words = []
    classifications = []
    with open(file_name) as f:
        for line in f:
            data = OUTSIDE_LABEL
            word = ""
            if line.strip() != "":
                sp = line.split("\t")
                word = sp[0].strip()
                if len(sp) > 1 and sp[2].strip() != "" and sp[2].strip() != OUTSIDE_LABEL:
                    data = INSIDE_LABEL
            words.append(word)
            classifications.append(data)
            
    return words, classifications
        
            
            
        
#word2vec_model = KeyedVectors.load_word2vec_format('/Users/marsk757/wordspaces/pubmed2018_w2v_200D.bin', binary=True)
#
#word2vec_model = None
#print(get_vectors(["diabetes", "mellitus", "seenrnrnrnqlsowk"], word2vec_model))

training_data, training_labels = read_data("manually_annotated_data.txt")
for a,b in zip(training_data, training_labels):
    print(a,b)


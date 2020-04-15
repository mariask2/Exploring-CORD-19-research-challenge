from gensim.models import KeyedVectors
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegressionCV
import numpy as np

OUTSIDE_LABEL = "O"
INSIDE_LABEL = "I"
WINDOW_SIZE = 5
VECTOR_LENGTH = 200
default_vector = np.array([0.0] * VECTOR_LENGTH)

#X, y = load_iris(return_X_y=True)
#print(X)
#print(y)
#exit(1)
#clf = LogisticRegression(random_state=0, max_iter=1000).fit(X, y)

#print(clf.predict(X[:2, :]))


def get_vector_for_word(word, word2vec_model):
    if word in word2vec_model:
        vector = word2vec_model[word]
    else:
        vector = default_vector
    return vector

def get_vectors(words, word2vec_model):
    return_vectors = []
    for i in range(0, len(words)):
        word = words[i]
        vector = get_vector_for_word(word, word2vec_model)
        
        for j in range(1, WINDOW_SIZE+1):
            before_index = i - j
            after_index = i -j
            before_vector = default_vector
            after_vector = default_vector
            if before_index >= 0:
                before_vector = get_vector_for_word(words[before_index], word2vec_model)
            if after_index >= len(words):
                after_vector = get_vector_for_word(words[after_index], word2vec_model)
            vector = np.concatenate((vector, before_vector))
            vector = np.concatenate((vector, after_vector))
        
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
        
            
            
def get_training_labels_vec(training_labels):
    labels = []
    for label in training_labels:
        if label == OUTSIDE_LABEL:
            labels.append(0)
        else:
            labels.append(1)
    return labels

#
#res = get_vectors(["diabetes", "mellitus", "seenrnrnrnqlsowk"], word2vec_model)
#for el in res:
#    print(el)
#    print(el.shape)

training_data, training_labels = read_data("manually_annotated_data.txt")

word2vec_model = KeyedVectors.load_word2vec_format('/Users/marsk757/wordspaces/pubmed2018_w2v_200D.bin', binary=True)
y = get_training_labels_vec(training_labels)
X = get_vectors(training_data, word2vec_model)
clf = LogisticRegressionCV(random_state=0, max_iter=1000, cv=5, class_weight="balanced", n_jobs=4).fit(X, y)
prediction_prob = clf.predict_proba(X)
predictions = clf.predict(X)

for x, y, predicted_prob, predicted in zip(training_data, y, prediction_prob, predictions):
    print(x, y, predicted, predicted_prob)

print(clf.classes_)

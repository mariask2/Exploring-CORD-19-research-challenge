from gensim.models import KeyedVectors
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegressionCV
from nltk.tokenize import TweetTokenizer
from joblib import dump, load
import glob
import json
import csv
import os.path
import numpy as np

OUTSIDE_LABEL = "O"
INSIDE_LABEL = "I"
WINDOW_SIZE = 5
VECTOR_LENGTH = 200
default_vector = np.array([0.0] * VECTOR_LENGTH)


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

word2vec_model = KeyedVectors.load_word2vec_format('/Users/marsk757/wordspaces/pubmed2018_w2v_200D.bin', binary=True)
model_file_name = 'risk_mention_model.joblib'

if not os.path.isfile(model_file_name):
    # If no model has been trained before
    training_data, training_labels = read_data("manually_annotated_data.txt")

    
    y = get_training_labels_vec(training_labels)
    X = get_vectors(training_data, word2vec_model)
    clf = LogisticRegressionCV(random_state=0, max_iter=1000, cv=5, class_weight="balanced", n_jobs=8).fit(X, y)

    dump(clf, model_file_name)

    clf_loaded = load(model_file_name)

    prediction_prob = clf_loaded.predict_proba(X)
    predictions = clf_loaded.predict(X)

    for x, y, predicted_prob, predicted in zip(training_data, y, prediction_prob, predictions):
        print(x, y, predicted, predicted_prob)

# Use the model
clf_loaded = load(model_file_name)

CONTENT_DIR = "../CORD-19-research-challenge"

headings_to_exclude_set = set()
with open("headings_to_exclude.txt") as f:
    for line in f:
        headings_to_exclude_set.add(line.strip())

meta_data_dict = {}
with open(os.path.join(CONTENT_DIR, 'metadata.csv')) as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        id = row[0].strip()
        if ";" in id:
            id_list = [x.strip() for x in id.split(";")]
        else:
            id_list = [id]
        for id_el in id_list:
            meta_data_dict[id_el] = row

tknzr = TweetTokenizer(preserve_case=True)

file_dict = {}
prediction_scores = [(0.0, "", "")]*20
for dir in ["biorxiv_medrxiv"]:# , comm_use_subset", "custom_license", "noncomm_use_subset"]:
    path = os.path.join(CONTENT_DIR, dir, dir)
    files = glob.glob(path + "/*.json")
    for file in files:
        with open(file) as f:
            file_text = []
            plain_text_lower = ""
            data = json.load(f)
            paper_id = data["paper_id"].strip()
            meta_data = []
            if paper_id in meta_data_dict:
                meta_data = meta_data_dict[paper_id]
            else:
                print(paper_id + " not in metadata")
                print(nr_of_sections)
                print()
                pass
                  
            current_section = None
            for el in data["body_text"]:
                if el["section"].lower() != current_section:
                    current_section = el["section"].lower()
                    if current_section not in headings_to_exclude_set:
                        text = el["text"]
                        plain_text_lower = plain_text_lower + " " + text.lower()
                        file_text.append(tknzr.tokenize(text))

            if "covid" in plain_text_lower or "coronavirus disease 2019" \
                in plain_text_lower or "sars-cov-2" in plain_text_lower \
                or "2019-nCoV" in plain_text_lower:
                
                for paragraph in file_text:
                    X = get_vectors(paragraph, word2vec_model)
                    prediction_prob = clf_loaded.predict_proba(X)
                    predictions = clf_loaded.predict(X)
                    if 1 in predictions:
                        for x,  predicted_prob, predicted in zip(paragraph, prediction_prob, predictions):
                            if predicted == 1:
                                prediction_scores.append((np.amax(predicted_prob), " ".join(paragraph), x))
                                prediction_scores.sort(reverse=True)
                                prediction_scores = prediction_scores[:-1]
                            #print(x, predicted, predicted_prob)
                            #print(prediction_scores)
                    file_dict[paper_id] = file_text

  
print(prediction_scores)
    

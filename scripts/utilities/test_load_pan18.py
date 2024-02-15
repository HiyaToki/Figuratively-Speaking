# -*- coding: utf-8 -*-

import os
import sys
import json
import random
import numpy as np

from nltk.tokenize import sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

# method to save json file
def save_json(data, filename):
    # open file and load
    with open(filename, 'w') as fp:
        json.dump(data, fp)

# load data from our json format
def load_json(filename):
    with open(filename, "r") as fp:
        dataset = json.load(fp)
        
    return dataset

# method to parse ground thruth for pan18
def get_ground_truth(problem_path):
    raw_ground_truth = load_json(os.path.abspath(os.path.join(problem_path, "ground-truth.json")))
    ground_truth = dict()
    
    for raw_element in raw_ground_truth["ground_truth"]:
        txt_id = raw_element["unknown-text"]
        author = raw_element["true-author"]
        
        ground_truth[txt_id] = author
        
    return ground_truth

# method to basic preprocess text
def process(text):
    text = TreebankWordDetokenizer().detokenize(text.split())
    return text

# method to load the pan18 dataset
def load_pan18(filepath):
    dataset = dict()

    for problem in ["problem00001", "problem00002", "problem00003", "problem00004"]:
        problem_path = os.path.join(filepath, problem)
        ground_truth = get_ground_truth(problem_path)
        dataset[problem] = []
        
        for candidate in os.listdir(problem_path):
            candidate_path = os.path.join(problem_path, candidate)
            
            # skip info files
            if candidate in ["ground-truth.json", "problem-info.json"]:
                continue
            
            for file in os.listdir(candidate_path):
                file_name = os.path.join(candidate_path, file)
        
                # open file to read lines
                with open(file_name, "r", encoding = "latin-1") as f:
                    text = process(f.read())
                    author = candidate
                    is_test = False
                    txt_id = file
                    
                    # retrieve author for test files
                    if candidate == "unknown":
                        author = ground_truth[txt_id]
                        is_test = True
                        
                    # save relative info 
                    data_item = dict()
                    data_item["text"] = text
                    data_item["author"] = author
                    data_item["text_id"] = txt_id
                    data_item["is_test"] = is_test
                    dataset[problem].append(data_item)
                    
    return dataset

# method to get features and labels from dataset
def get_documents(dataset):
    doc_train = [] 
    i_train = []
    doc_test = []
    i_test = []

    # get features and authors as labels
    for i in range(len(dataset)):
        x = dataset[i]["text"]
            
        # append features and class
        if dataset[i]["is_test"]:
            doc_test.append(x)
            i_test.append(i)
            
        else:
            doc_train.append(x)
            i_train.append(i)
            
    return doc_train, i_train, doc_test, i_test

# method to add n-gram features to a dataset
def add_tfidf_features(dataset, n_gram_type = "word"):
    print("EXTRACTING " + n_gram_type.upper() + " N-GRAMS...", end = " ")
    doc_train, i_train, doc_test, i_test = get_documents(dataset)

    # create appropriate key
    key = n_gram_type + "_n_gram_tfidf_features"
    
    # use stopwords?
    if n_gram_type == "word":
        english_stop_words = "english"
    
    else: # when its "char"
        english_stop_words = None
        
    # create the TfidfVectorizer instance with word and character n-grams
    vectorizer = TfidfVectorizer(analyzer = n_gram_type, 
                                 stop_words = english_stop_words,
                                 ngram_range = (1, 5),
                                 max_features = 1024,
                                 max_df = 0.99,
                                 min_df = 2,
                                 )

    # learn vocabulary on train set
    # and then generate the TF-IDF vectors for the documents
    train_tfidf_vectors = vectorizer.fit_transform(doc_train)
    features = vectorizer.get_feature_names_out()
    
    print("DONE!")
    return features

#### SCRIPT ####
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
dataset = load_pan18(base_dir + "/data/pan18/pan18-cross-domain-authorship-attribution-test-dataset")
problem = "problem00002"

features = add_tfidf_features(dataset[problem], "char")
features = features.tolist()

save_json(features, "./character_features_pan18_problem00002.json")
    
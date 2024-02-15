# -*- coding: utf-8 -*-

import os
import json
import warnings
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# ignore warnings
warnings.filterwarnings("ignore")

# method to save json file
def save_json(data, filename):
    # open file and load
    with open(filename, 'w') as fp:
        json.dump(data, fp)

# method to load a json file to memory
def load_json(filename):
    # open file and load
    with open(filename, 'r') as fp:
        data = json.load(fp)

    return data

# method to get features and labels from dataset
def get_x_y(dataset, key):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    
    # get features and authors as labels
    for i in range(len(dataset)):
        y = dataset[i]["author"]
        
        # look for feature concat
        if "+" in key:
            key_parts = key.split("+")
            x = []
            
            # concatenate listed features
            for key_part in key_parts:
                x.extend(dataset[i][key_part])
            
        else: # just a single feature
            x = dataset[i][key]
            
        # append features and class
        if dataset[i]["is_test"]:
            x_test.append(x)
            y_test.append(y)
            
        else:
            x_train.append(x)
            y_train.append(y)
            
    return x_train, y_train, x_test, y_test

### SCRIPT ###
keys = [
        "stylometric_features",
        "sbert_model_embedding",
        "joint_model_embedding",
        "word_n_gram_tfidf_features",
        "char_n_gram_tfidf_features",
        "joint_model_one_hot_prediction",
        "binary_model_one_hot_prediction",
        "sbert_model_embedding+joint_model_embedding",
        "word_n_gram_tfidf_features+joint_model_embedding",
        "char_n_gram_tfidf_features+joint_model_embedding",
        "word_n_gram_tfidf_features+char_n_gram_tfidf_features",
        "word_n_gram_tfidf_features+char_n_gram_tfidf_features+joint_model_embedding",
        ]

# load dataset
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
datasets_file_path = base_dir + "/data/pan06/pan06-authorship-attribution-test-dataset.json"
dataset = load_json(datasets_file_path)
results = dict()

# perform experiments
for key in keys:
    print("TRAINING MODEL WITH " + key.upper() + " FEATURES...", end = " ")
    x_train, y_train, x_test, y_test = get_x_y(dataset, key)
        
    # create an instance of StandardScaler
    scaler = StandardScaler()
    
    # fit and scale training features
    x_train_scaled = scaler.fit_transform(x_train)
        
    # create the MLP model
    mlp = MLPClassifier(hidden_layer_sizes = (1024,), 
                        learning_rate_init = 2e-5,
                        activation = 'relu', 
                        random_state = 42,
                        max_iter = 1000, 
                        solver = 'adam', 
                        )
    
    # fit MLP using training set
    mlp.fit(x_train_scaled, y_train)
    
    # scale test features
    x_test_scaled = scaler.transform(x_test)
    
    # predict on test set
    y_pred = mlp.predict(x_test_scaled)
    
    # calculate the F1-score using the weighted average
    f1 = f1_score(y_test, y_pred, average = 'weighted')
    
    # store results of experiment
    results[key] = f1
    print("DONE!")

output_file = base_dir + "/data/output/evaluations/pan06-scale-mlp-auth-attr-results.json"
os.makedirs(os.path.dirname(output_file), exist_ok = True)

# save experiment result to file
save_json(results, output_file)

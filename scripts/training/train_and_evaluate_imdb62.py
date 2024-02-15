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
    X = []
    y = []
    
    # get features and authors as labels
    for i in range(len(dataset)):
        y.append(dataset[i]["author"])
        
        # look for feature concat
        if "+" in key:
            key_parts = key.split("+")
            features = []
            
            # concatenate listed features
            for key_part in key_parts:
                features.extend(dataset[i][key_part])
            
            # now append to X
            X.append(features)
            
        else: # just a single feature
            X.append(dataset[i][key])
            
    return X, y

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
datasets_file_path = base_dir + "/data/imdb62/imdb62_train.json"
dataset = load_json(datasets_file_path)

# also load test set
test_datasets_file_path = base_dir + "/data/imdb62/imdb62_test.json"
test_dataset = load_json(test_datasets_file_path)
results = dict()

# perform experiments
for key in keys:
    print("TRAINING MODEL WITH " + key.upper() + " FEATURES...", end = " ")
    x_train, y_train = get_x_y(dataset, key)
    x_test, y_test = get_x_y(test_dataset, key)
     
    # create an instance of StandardScaler
    scaler = StandardScaler()
    
    # fit and scale training features
    x_train_scaled = scaler.fit_transform(x_train)
        
    # create and train the MLP model
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

output_file = base_dir + "/data/output/evaluations/imdb62-scale-mlp-auth-attr-results.json"
os.makedirs(os.path.dirname(output_file), exist_ok = True)

# save experiment result to file
save_json(results, output_file)

# -*- coding: utf-8 -*-

import os
import sys
import json
import torch
import warnings
import numpy as np
import torch.nn.functional as F
from transformers import logging
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ignore all warnings
logging.set_verbosity_error()
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# load data from our json format
def load_json(filename):
    with open(filename, "r") as fp:
        dataset = json.load(fp)
        
    return dataset

# method to get the dataset keys for each class
def get_feature_dataset_map(multi_label_dataset):
    feature_datasets = dict()
    for key in multi_label_dataset.keys():
        
        # skip train splits
        if "-train" in key:
            continue
        
        for i in range(len(multi_label_dataset[key])):
            for label in multi_label_dataset[key][i]["labels"]:
                
                # also skip "not_feature_X" labels and "literal"
                if "not_" in label or label == "literal":
                    continue
                
                elif label not in feature_datasets:
                    feature_datasets[label] = set()

                feature_datasets[label].add(key)
    
    # this will tell us what datasets contain the fig-lang feature we want
    return feature_datasets

# method to create/load appropriate label mappings
def get_label_map(feature_name, filepath):
    
    # load label2id and id2label maps
    if os.path.isfile(os.path.join(filepath, "label2id.json")):
        label2id = load_json(os.path.join(filepath, "label2id.json"))
        id2label = load_json(os.path.join(filepath, "id2label.json"))
        
    else:
        print("No mapping files found in: ", filepath)
        sys.exit(1)
    
    return label2id, id2label

# method to convert dataset to binary classification task
def get_binary_labels(dataset, feature_name):
    # create the negative feature name
    negative_feature_name = "not_" + feature_name
    literal_name = "literal"
    bin_labels = []
        
    for i in range(len(dataset)):
        labels = dataset[i]["labels"]
        
        # positive labels
        if feature_name in labels:
            bin_labels.append(feature_name)
        
        # negative and literal labels
        elif negative_feature_name in labels or literal_name in labels:
            bin_labels.append(negative_feature_name)
            
        else:
            # all other possible labels
            bin_labels.append(negative_feature_name)
            
    return bin_labels

# method to predict style class of a single sentence
def predict(text, model, tokenizer, id2label):

    # encode the text
    inputs = tokenizer(text,
                       padding = True,
                       max_length = 512,
                       truncation = True,
                       return_tensors = 'pt').to("cuda")

    with torch.no_grad():
        # pass encodings to the model
        logits = model(**inputs).logits

        # compute probability as softmax score
        probability = F.softmax(logits, dim = -1).tolist()
        predicted_class_index = np.argmax(probability)
        prediction = id2label[str(predicted_class_index)]

    return prediction

# ----- 1. Load data -----#

# Read data
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
dataset_file_path = base_dir + "/data/fig-lang-dataset.json"
multi_label_dataset = load_json(dataset_file_path)

# get list of dataset names based on fig lang features
feature_datasets = get_feature_dataset_map(multi_label_dataset)

for feature_name in feature_datasets.keys():
    print("\nEVALUATING MODEL: ", feature_name)
    
    # load classification model
    model_name = "roberta-large-" + feature_name 
    model_file_path = base_dir + "/models/" + model_name
    tokenizer = AutoTokenizer.from_pretrained(model_file_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_file_path).to("cuda")
    model.eval() # lock model in eval mode
    
    # get label2id and id2label mappings for this model
    label2id, id2label = get_label_map(feature_name, model_file_path)

    # get test set for each dataset
    for dataset_name in feature_datasets[feature_name]:
        print("\tFetching Test Set: ", dataset_name)
        
        # get test set and get binary ground truth
        dataset = multi_label_dataset[dataset_name]
        labels = get_binary_labels(dataset, feature_name)
        
        # predict
        predictions = []
        print("\tPredicting...", end = " ")
        for i in range(len(dataset)):
            predictions.append(predict(dataset[i]["text"], model, tokenizer, id2label))
        
        print("done! Create classification report...", end = " ")
        
        # classification report for each testing dataset
        report = classification_report(labels, predictions, zero_division = 0)
        
        # create output report file
        output_file = base_dir + "/data/output/evaluations/" + model_name + "-" + dataset_name + "-evaluation.txt"
        os.makedirs(os.path.dirname(output_file), exist_ok = True)
        
        # write overal report
        with open(output_file, "w") as fp:
            fp.write("\n\nCLASSIFICATION EVALUATION RESULTS: \n")
            fp.write("\t" + report + "\n")

        print("and done!")
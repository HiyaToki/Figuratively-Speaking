# -*- coding: utf-8 -*-

import os
import sys
import json
import torch
import warnings
import numpy as np
from tqdm import tqdm
from transformers import logging
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ignore all warnings
logging.set_verbosity_error()
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

# method to load appropriate label mappings
def get_label_map(filepath):
    
    # load label2id and id2label maps
    if os.path.isfile(os.path.join(filepath, "label2id.json")):
        label2id = load_json(os.path.join(filepath, "label2id.json"))
        id2label = load_json(os.path.join(filepath, "id2label.json"))
        
    else:
        print("No mapping files found in: ", filepath)
        sys.exit(1)
    
    return label2id, id2label

# method to predict fig lang classes of a single sentence
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

        # not softmax, need independent probability of class being true
        probability = torch.sigmoid(logits).squeeze().tolist()
        
    return probability

# method to convert a list of labels to a one hot vector
def convert_to_one_hot(labels, label2id):
    one_hot_vector = [0] * len(label2id)
    
    for label in labels:
        if label in label2id:
            one_hot_vector[label2id[label]] = 1
    
    return one_hot_vector

# method to run the prediction loop for the dataset
def prediction_loop(dataset, model, tokenizer, label2id, id2label):
    predictions = []
    labels = []
    
    for i in tqdm(range(len(dataset)), desc ="DEV SET PROGRESS"):
        # predict propability distribution of single text
        probability = predict(dataset[i]["text"], model, tokenizer, id2label)
        predictions.append(probability)
        
        # use human annotation to create one-hot-labels of ground truth
        # one_hot_vector = convert_to_one_hot(dataset[i]["labels"], label2id)
        # labels.append(one_hot_vector)
        
        # use binary model predictions to tune thresholds
        labels.append(dataset[i]["one_hot_prediction"])
        
    return labels, predictions

### SCRIPT ###

# load dataset and label maps
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
datasets_file_path = base_dir + "/data/multi_label_fig_lang_dev.json"
dataset = load_json(datasets_file_path)

model_names = ["roberta-large-joint-fig-lang"]

# start finetuning all models
for model_name in model_names: 

    print("\nFITTING MODEL THRESHOLDS ON THE DEV SET:", model_name)
    model_file_path = base_dir + "/models/" + model_name
    
    # load label to id maps, tokenizer, and model
    label2id, id2label = get_label_map(model_file_path)
    tokenizer = AutoTokenizer.from_pretrained(model_file_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_file_path).to("cuda")
    model.eval() # lock model in eval mode
    
    # run prediction loop and get predictions
    labels, predictions = prediction_loop(dataset, model, tokenizer, label2id, id2label)
    
    # start the fitting process
    print("\tSCANNING...")
    thresholds = np.arange(0.0, 1.01, 0.01)
    optimal_threshold = dict()
    
    for feature_name in list(label2id.keys()):
        optimal_f1_score = -100
        gt_labels = []
        
        # create the label's ground truth list
        for i in range(len(labels)):
            gt_labels.append(labels[i][label2id[feature_name]])
        
        # create the prediction output based on threshold
        for threshold in thresholds:
            
            pr_labels = []
            for j in range(len(predictions)):
                if predictions[j][label2id[feature_name]] >= threshold:
                    pr_labels.append(1)
                    
                else:
                    pr_labels.append(0)
                
            # based on f1 score
            score = f1_score(gt_labels, pr_labels)
            
            # save best score
            if score >= optimal_f1_score:
                optimal_f1_score = score
                optimal_threshold[feature_name] = threshold
    
    # print threshold values
    json_string = json.dumps(optimal_threshold, indent = 4)
    print("\tOPTIMAL THRESHOLDS FOUND:\n", json_string)
    print("******************************************")
    
    # save thresholds to model directory
    threshold_file_path = os.path.join(model_file_path, "thresholds_bin.json")
    save_json(optimal_threshold, threshold_file_path)
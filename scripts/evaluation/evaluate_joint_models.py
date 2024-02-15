# -*- coding: utf-8 -*-

import os
import sys
import json
import torch
import warnings
from tqdm import tqdm
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

# method to convert a list of labels to a one hot vector
def convert_to_one_hot(labels, label2id):
    one_hot_vector = [0] * len(label2id)
    
    for label in labels:
        if label in label2id:
            one_hot_vector[label2id[label]] = 1
    
    return one_hot_vector

# method to predict style class of a single sentence
def predict(text, model, tokenizer, id2label, thresholds):

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
        one_hot_prediction = [0] * len(id2label)
        prediction = []

        for i in range(len(probability)):
            feature_name = id2label[str(i)]
            if probability[i] >= thresholds[feature_name]:
                prediction.append(feature_name)
                one_hot_prediction[i] = 1

        # package probability scores and prediction into a neat dict
        predictions = {"probability": probability,
                       "predicted_labels": prediction,
                       "one_hot_prediction": one_hot_prediction,
                       }

    return predictions

# method to run the prediction loop for the dataset
def prediction_loop(dataset, name, model, tokenizer, label2id, id2label, thresholds):
    predictions = []
    labels = []
    
    for i in tqdm(range(len(dataset)), desc = name + " SET PROGRESS"):
        # predict propability distribution of single text
        prediction = predict(dataset[i]["text"], model, tokenizer, id2label, thresholds)
        predictions.append(prediction["one_hot_vector"])
        
        # use human annotation to create one-hot-labels of ground truth
        one_hot_vector = convert_to_one_hot(dataset[i]["labels"], label2id)
        labels.append(one_hot_vector)
        
    return labels, predictions


### SCRIPT ###

# load dataset and label maps
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
dataset_file_path = base_dir + "/data/multi-label-fig-lang-dataset.json"
multi_label_dataset = load_json(dataset_file_path)

model_names = ["roberta-large-joint-fig-lang"]

# start finetuning all models
for model_name in model_names: 

    print("\nEVALUATING MODEL ON TEST SETS:", model_name)
    model_file_path = base_dir + "/models/" + model_name
    
    # load label maps, thresholds, tokenizer, and model
    label2id, id2label = get_label_map(model_file_path)
    tokenizer = AutoTokenizer.from_pretrained(model_file_path)
    
    thresholds = load_json(os.path.join(model_file_path, "thresholds_bin.json"))
    model = AutoModelForSequenceClassification.from_pretrained(model_file_path).to("cuda")
    model.eval() # lock model in eval mode
    
    ordered_labels = []
    for i in range(len(id2label.keys())):
        ordered_labels.append(id2label[str(i)])
    
    for dataset in multi_label_dataset.keys():
        # skip training sets
        if "-train" in dataset:
            continue
    
        # run prediction loop and get predictions
        labels, predictions = prediction_loop(multi_label_dataset[dataset], 
                                              dataset, 
                                              model, 
                                              tokenizer, 
                                              label2id, 
                                              id2label, 
                                              thresholds)
        
        # classification report for Agenda multi-label eval using optimal threshold
        report = classification_report(labels, predictions, zero_division = 0, target_names = ordered_labels)

        # create output report file
        output_file = base_dir + "/data/output/evaluations/" + model_name + "-thr-bin-evaluation.txt"
        os.makedirs(os.path.dirname(output_file), exist_ok = True)

        # write overal report
        with open(output_file, "a") as fp:
            fp.write("\n\nCLASSIFICATION EVALUATION RESULTS ON DATASET: " + dataset + "\n")
            fp.write("\t" + report + "\n")

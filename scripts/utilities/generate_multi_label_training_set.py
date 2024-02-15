# -*- coding: utf-8 -*-


import os
import json
import random

label2index = {
    'metaphor': 0,
    'simile': 1, 
    'idiom': 2, 
    'sarcasm': 3, 
    'irony': 4,
    'hyperbole': 5, 
    }

# method to save training data into json format
def save_json(filename, dataset):
    with open(filename, "w") as fp:
        json.dump(dataset, fp)
        
# load data from our json format
def load_json(filename):
    print("Loading file: ", filename)
    with open(filename, "r") as fp:
        dataset = json.load(fp)
        
    return dataset

# method to get a dataset summary per label
def dataset_statistics(dataset):
    label2count = dict()
    for example in dataset:
        labels = example["predicted_labels"]
        if all(s.startswith("not_") for s in labels):
            labels = ["literal"]
        
        for label in labels:
            if label not in label2count:
                label2count[label] = 0
                
            label2count[label] += 1
        
    print("\nTotal label distribution: ")
    for label in label2count.keys():
        print("\tLabel: ", label, " -> ", label2count[label])

# method to convert labels
def expand_literal_label():
    bin_labels = []
    for feature_name in label2index.keys():
        negative_feature_name = "not_" + feature_name
        bin_labels.append(negative_feature_name)
                
    return bin_labels

# method to verify that human labels and predictions do not contradict
def verify_predictions(predicted_labels, labels):
    
    # If labels say "literal" we cannot have "feature_X" in the binary labels.
    # But, it's okay if predictions contain "not_feature_X" only.
    if len(labels) == 1:
        if labels[0] == "literal":
            # so, if label is "literal" then it's equivalent with all fig lang 
            # features being "not_"
            labels = expand_literal_label()
    
    # we must check that predictions and human labels agree
    are_valid_labels = True
    for label in labels:
        if "not_" in label: # If labels say "not_feature_X" we cannot have "feature_X" in the binary labels.
            positive_label = label.replace("not_", "")
            if positive_label in predicted_labels:
                are_valid_labels = False
            
        else: # If labels say "feature_X" we cannot have "not_feature_X" in the binary labels.
            negative_label = "not_" + label
            if negative_label in predicted_labels:
                are_valid_labels = False
            
    return are_valid_labels

# method to generate one training set for each fig lang feature
def generate_training(base_dir, multi_label_dataset):
    counter = 0
    valid_dataset = []
    for dataset_name in multi_label_dataset.keys():
        
        # skip testing sets from multi-label tagging
        if "-test" in dataset_name:
            continue 
        
        # get predicted labels and annotations examples under the specified dataset name
        for i in range(len(multi_label_dataset[dataset_name])):
            labels = multi_label_dataset[dataset_name][i]["labels"]
            predicted_labels = multi_label_dataset[dataset_name][i]["predicted_labels"]
            counter += 1

            # check that human labels and predictions do not contradict
            if verify_predictions(predicted_labels, labels):
                valid_dataset.append(multi_label_dataset[dataset_name][i])
            
    print("Total valid training examples: ", len(valid_dataset))
    print("All possible examples: ", counter)
    dataset_statistics(valid_dataset)
    random.shuffle(valid_dataset)    
    
    # reserve 10% of all valid examples as dev set
    dev_set_size = int(0.1 * len(valid_dataset))
    training_set = valid_dataset[dev_set_size:]
    dev_set = valid_dataset[:dev_set_size]
        
    # save to file
    file_path = base_dir +"/data/multi_label_fig_lang_" 
    if not os.path.isfile(file_path + "training.json"):
        save_json(file_path + "training.json", training_set)
        save_json(file_path + "dev.json", dev_set)

### SCRIPT ###
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
dataset_file_path = base_dir + "/data/multi-label-fig-lang-dataset.json"
multi_label_dataset = load_json(dataset_file_path)
generate_training(base_dir, multi_label_dataset)
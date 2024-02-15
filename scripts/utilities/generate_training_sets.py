# -*- coding: utf-8 -*-

import os
import json
import random

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

# method to get the dataset keys for each class
def get_feature_dataset_map(multi_label_dataset):
    feature_datasets = dict()
    for key in multi_label_dataset.keys():
        
        # skip test splits
        if "-test" in key:
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

# method to get positive examples for training
def get_positive_examples(feature_name, dataset_names, multi_label_dataset):
    positive_examples = []
    for dataset_name in dataset_names:
        for example in multi_label_dataset[dataset_name]:
            
            # get examples that are labeled with the positive class
            # for examples, the positive_examples will contain "irony" examples
            # when we train the "irony" detection model
            if feature_name in example["labels"]:
                positive_examples.append(example)
    
    # shuffle list of examples
    random.shuffle(positive_examples)   
    
    return positive_examples
        
# method to get negative examples for training
def get_negative_examples(feature_name, dataset_names, multi_label_dataset):
    literal_examples = []
    negative_examples = []
    ood_negative_examples = []
    
    # get negative feature name
    negative_feature_name = "not_" + feature_name
    
    # get all negative examples
    for dataset_name in dataset_names:
        for example in multi_label_dataset[dataset_name]:
            # get examples that are labeled with the negative class
            # negative_examples will contain "not_feature_X" labels
            if negative_feature_name in example["labels"]:
                negative_examples.append(example)
            
            # literal_examples will contain "literal" labels
            elif "literal" in example["labels"]:
                literal_examples.append(example)
    
    # add more literal examples from other datasets
    for dataset_name in multi_label_dataset.keys():
        # skip test splits and dataset names we already saw above
        if "-test" in dataset_name or dataset_name in dataset_names:
            continue
        
        for example in multi_label_dataset[dataset_name]:
            # get examples that are labeled with the negative class
            # literal_examples will contain "literal" labels
            if "literal" in example["labels"]:
                literal_examples.append(example)
    
    # get some out of domain examples to boost negative class
    for dataset_name in multi_label_dataset.keys():
        # skip test splits
        if "-test" in dataset_name:
            continue
        
        for example in multi_label_dataset[dataset_name]:
            # get examples that are not labeled with our tast related labels for example,
            # if we train "irony" model the ood_negative_examples will contain
            # examples that are not "irony", not "not_irony" and not "literal", 
            # such as "metaphor", "simile", "idiom", etc. 
            if ("literal" not in example["labels"]
            and "not_irony" not in example["labels"]
            and  feature_name not in example["labels"] 
            and "not_sarcasm" not in example["labels"]
            and "not_metaphor" not in example["labels"]
            and negative_feature_name not in example["labels"]):
                ood_negative_examples.append(example)
    
    # shuffle lists of negative examples
    random.shuffle(literal_examples)
    random.shuffle(negative_examples)
    random.shuffle(ood_negative_examples)
    
    return literal_examples, negative_examples, ood_negative_examples
    
# method to generate one training set for each fig lang feature
def generate_training(base_dir, multi_label_dataset, ood_percentage = 0.1):
    feature_datasets = get_feature_dataset_map(multi_label_dataset)
    
    for feature_name in feature_datasets.keys():
        training_set = []
        print("\nGenerating a training set for the ", feature_name, " detection task.")
        
        # get a list of training datasets per task
        dataset_names = feature_datasets[feature_name]
        print("\tPositive/negative examples come from the following datasets: ", ", ".join(dataset_names))
        
        # get positive examples to form training set
        positive_examples = get_positive_examples(feature_name, dataset_names, multi_label_dataset)
        print("\t\tNumber of total positive examples: ", len(positive_examples))
    
        # get negative examples to form training set
        literal_examples, negative_examples, ood_negative_examples = get_negative_examples(feature_name, dataset_names, multi_label_dataset)
        print("\t\tNumber of total negative examples: ", len(negative_examples))
        print("\t\tNumber of total literal examples: ", len(literal_examples))
        
        # balance positive and negative sets
        if len(negative_examples) > 0:
            balance_size = min(len(positive_examples), len(negative_examples) + len(literal_examples))
            lit_neg_balance_size = int(balance_size * 0.5)
            print("\t\tBalancing at: ", balance_size, " positive examples.")
            
            # add positive examples
            training_set.extend(positive_examples[:balance_size])
            
            # add negative examples
            if lit_neg_balance_size > len(negative_examples):
                # add all negative examples
                training_set.extend(negative_examples)
                print("\t\tBalancing at: ", len(negative_examples), " negative examples.")
                
                # add fill the rest with literal examples
                lit_balance_size = balance_size - len(negative_examples)
                training_set.extend(literal_examples[:lit_balance_size])
                print("\t\tBalancing at: ", lit_balance_size, " literal examples.")
                
            else:
                training_set.extend(negative_examples[:lit_neg_balance_size])
                training_set.extend(literal_examples[:lit_neg_balance_size])
                print("\t\tBalancing at: ", lit_neg_balance_size, " negative examples.")
                print("\t\tBalancing at: ", lit_neg_balance_size, " literal examples.")

        else:
            balance_size = min(len(positive_examples), len(literal_examples))
            print("\t\tBalancing at: ", balance_size, " positive examples.")
            print("\t\tBalancing at: ", balance_size, " literal examples.")
        
            # add positive examples
            training_set.extend(positive_examples[:balance_size])
            
            # add literal examples
            training_set.extend(literal_examples[:balance_size])
        
        # # add 10% of ood examples
        # ood_size = int(balance_size * ood_percentage)
        # training_set.extend(ood_negative_examples[:ood_size])
        # print("\t\tAdding ", ood_size, " extra OOD examples.")
        
        # shuffle final training set
        random.shuffle(training_set)
        print("\tTotal training set size: ", len(training_set))
        
        # save to file
        file_path = base_dir +"/data/" + feature_name + "_training.json"
        if not os.path.isfile(file_path):
            save_json(file_path, training_set)

### SCRIPT ###
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
dataset_file_path = base_dir + "/data/fig-lang-dataset.json"
multi_label_dataset = load_json(dataset_file_path)
generate_training(base_dir, multi_label_dataset)

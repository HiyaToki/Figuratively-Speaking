# -*- coding: utf-8 -*-

import os
import json
import time
import torch
import random
import warnings
from transformers import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, DefaultDataCollator

# ignore all warnings
logging.set_verbosity_error()
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

# method to get model inputs for classification training
def get_model_inputs(dataset, tokenizer, label2id):
    model_inputs = []

    for i in range(len(dataset)):
        text = dataset[i]["text"]
        label =  dataset[i]["label"]

        # ecode text
        model_input = tokenizer(text,
                                max_length = 512,
                                truncation = True,
                                padding = "max_length",
                                )

        # encode label
        model_input["labels"] = torch.tensor(label2id[label])
        model_inputs.append(model_input)
    
    # shuffle model inputs
    random.shuffle(model_inputs)
    
    return model_inputs

# method to create/load appropriate label mappings
def get_label_map(feature_name, filepath):
    # create the negative feature name
    negative_feature_name = "not_" + feature_name
    
    # create label to id and id to label maps
    if not os.path.isfile(os.path.join(filepath, "label2id.json")):
        label2id = {feature_name: 1,
                    negative_feature_name: 0}
        
        id2label = {1: feature_name,
                    0: negative_feature_name}
        
        # save them as json files
        save_json(label2id, os.path.join(filepath, "label2id.json"))
        save_json(id2label, os.path.join(filepath, "id2label.json"))
        
    else:
        label2id = load_json(os.path.join(filepath, "label2id.json"))
        id2label = load_json(os.path.join(filepath, "id2label.json"))
    
    return label2id, id2label

# method to convert dataset to binary classification task
def binarize_dataset(dataset, feature_name):
    # create the negative feature name
    negative_feature_name = "not_" + feature_name
    literal_name = "literal"
        
    for i in range(len(dataset)):
        labels = dataset[i]["labels"]
        
        # positive labels
        if feature_name in labels:
            label = feature_name
        
        # negative and literal labels
        elif negative_feature_name in labels or literal_name in labels:
            label = negative_feature_name
            
        else:
            # all other OOD labels
            label = negative_feature_name
            
        dataset[i]["label"] = label
    
    return dataset
            
# Read data
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
feature_names = ["metaphor", "simile", "idiom", "hyperbole", "sarcasm", "irony"]

for feature_name in feature_names:
    print("\nTRAINING MODEL: ", feature_name)
    
    model_name = "roberta-large-" + feature_name
    model_file_path = base_dir + "/models/" + model_name
    datasets_file_path = base_dir + "/data/" + feature_name + "_training.json"
    
    # make output model directory
    os.makedirs(model_file_path, exist_ok = True)
    
    # load dataset and label maps
    dataset = load_json(datasets_file_path)
    dataset = binarize_dataset(dataset, feature_name)
    label2id, id2label = get_label_map(feature_name, model_file_path)

    # preprocess the dataset using the proper tokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    train_dataset = get_model_inputs(dataset, tokenizer, label2id)

    # and load a RoBERTa Large model
    model = AutoModelForSequenceClassification.from_pretrained("roberta-large",
                                                               num_labels = len(label2id),
                                                               )
    # default collator
    data_collator = DefaultDataCollator()

    # standard arguments
    args = TrainingArguments(
        output_dir = model_file_path,
        per_device_train_batch_size = 16,
        overwrite_output_dir = True,
        save_strategy  = "epoch",
        num_train_epochs = 5,
        save_total_limit = 1,
        learning_rate = 2e-5,
        weight_decay = 0.01,
        warmup_ratio = 0.1,
        seed = 42
    )

    trainer = Trainer(
        args = args,
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        data_collator = data_collator,
    )
    
    # measure time required for training
    start = time.time()

    # start model training
    trainer.train()
    
    # compute running time
    time_elapsed = time.time() - start
    print("\nMODEL: ", feature_name, " TRAINING TIME: ", time_elapsed, " seconds.")
    
    # save model
    trainer.save_model(model_file_path)
    print("\nSAVING TRAINED MODEL INTO: ", model_file_path)
    
    del data_collator
    del tokenizer
    del trainer
    del model
    del args

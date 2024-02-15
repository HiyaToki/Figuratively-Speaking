# -*- coding: utf-8 -*-

import os
import json
import time
import torch
import random
import warnings
import numpy as np
from transformers import logging
from torch.utils.data import DataLoader
from sentence_transformers import InputExample
from transformers import TrainingArguments, Trainer, DefaultDataCollator
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, SentencesDataset, losses

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

# method to create/load appropriate label mappings
def get_label_map(filepath):
    
    # create label to id and id to label maps
    if not os.path.isfile(os.path.join(filepath, "label2id.json")):
        label2id = {
            'metaphor': 0,
            'simile':   1, 
            'idiom':    2, 
            'sarcasm':  3, 
            'irony':    4,
            'hyperbole':5,
            }

        id2label = {
            0: 'metaphor',
            1: 'simile', 
            2: 'idiom', 
            3: 'sarcasm', 
            4: 'irony',
            5: 'hyperbole', 
            }
        
        # save them as json files
        save_json(label2id, os.path.join(filepath, "label2id.json"))
        save_json(id2label, os.path.join(filepath, "id2label.json"))
        
    else:
        label2id = load_json(os.path.join(filepath, "label2id.json"))
        id2label = load_json(os.path.join(filepath, "id2label.json"))
    
    return label2id, id2label

# method of dataset peprocessing to get model inputs for classification training
def get_model_inputs(dataset, tokenizer):
    model_inputs = []

    for i in range(len(dataset)):
        labels = [float(j) for j in dataset[i]["one_hot_vector"]]
        text = dataset[i]["text"]

        # ecode text
        model_input = tokenizer(text,
                                max_length = 512,
                                truncation = True,
                                padding = "max_length",
                                )

        # encode one-hot vector
        model_input["labels"] = torch.tensor(labels)
        model_inputs.append(model_input)

    random.shuffle(model_inputs)
    return model_inputs

# method to compute the simple matching coefficient between two vectors
def simple_matching_coefficient(v1, v2):
    # count the number of matches (both 1s and 0s)
    matches = np.sum(v1 == v2)
    
    # divide by the total number of elements
    smc = matches / len(v1)
    
    return smc

# method to compute pairwise simple matching coefficient
def compute_pairwise_smc(vectors):
    M = np.array(vectors)
    
    # get the number of rows and columns
    rows, cols = M.shape
    
    # initialize an empty matrix to store the pairwise SMC values
    SMC = np.zeros((rows, rows))
    
    # loop over the rows and compute the SMC for each pair
    for i in range(rows):
        for j in range(i + 1, rows):
            SMC[i, j] = SMC[j, i] = simple_matching_coefficient(M[i], M[j])

    # return the SMC matrix
    return SMC

# function to pick related and unrelated indices
def pick_related_to_i(candidate_indices, seen_indices, n_start = 3, n_threshold = 10):
    
    n = n_start
    related_to_i = random.choice(candidate_indices[-n:])
    
    # avoid picking seen indices
    while related_to_i in seen_indices:
       related_to_i = random.choice(candidate_indices[-n:])
       n += 1
       
    n = n_start
    unrelated_to_i = random.choice(candidate_indices[:n])
    
    # avoid picking seen indices
    while unrelated_to_i in seen_indices:
       unrelated_to_i = random.choice(candidate_indices[:n])
       n += 1
       
    return related_to_i, unrelated_to_i

# method of dataset peprocessing to get model inputs for aux contrasive loss training
def get_model_inputs_2(dataset, sampling_rate = 0.3):
    vectors = []
    
    print("Dataset length: ", len(dataset))
    
    for i in range(len(dataset)):
        vectors.append(dataset[i]["one_hot_vector"])
        
    # compute SMC score for dataset
    SMC = compute_pairwise_smc(vectors)
    rows, cols = SMC.shape
    
    print("SMC matrix shape: (", rows, ",", cols, ")")
    
    # make a list of dataset indicies and shuffle to pick a random sample
    indices = [i for i in range(len(dataset))]
    random.shuffle(indices)
    seen_indices = set()
    model_inputs = []
    
    for i in indices:
        
        # skip already used data points
        if i in seen_indices: 
            continue
        
        # we saw 30% of the dataset, we gotta stop sampling
        if len(seen_indices) >= (sampling_rate * len(dataset)):
            break
        
        # get text for index i, add it to seen set
        text_i = dataset[i]["text"]
        seen_indices.add(i)
        
        # get a list of indices, ordered based on the corresponding SMC scores
        sorted_indices = np.argsort(SMC[i])
        
        # remove all seen indieces from the sorted indices list, without changing the order
        candidate_indices = sorted_indices[~np.in1d(sorted_indices, list(seen_indices))]
        
        # pick an index that is very related to i, among the top-3 most related elements
        # alsp pick an index that is very unrelated to i, among the 3 most unrelated elements
        related_to_i, unrelated_to_i = pick_related_to_i(candidate_indices, seen_indices)
           
        # get text for index un/related to i
        related_text_i = dataset[related_to_i]["text"]
        unrelated_text_i = dataset[unrelated_to_i]["text"]

        # get SMC score for index un/related to i
        related_smc_score = SMC[i][related_to_i]
        unrelated_smc_score = SMC[i][unrelated_to_i]
        
        # add un/related indices to the seen set
        seen_indices.add(related_to_i)
        seen_indices.add(unrelated_to_i)
        
        # create model input pairs based on text i and its un/related samples
        model_inputs.append(InputExample(texts=[text_i, related_text_i], label = float(related_smc_score)))
        model_inputs.append(InputExample(texts=[text_i, unrelated_text_i], label = float(unrelated_smc_score)))
        
        # print("\n\n*********************\n")
        # print("Text i:", text_i)
        # print("\tHuman annotations:", dataset[i]["labels"])
        # print("\tBin-model predictions:", dataset[i]["predicted_labels"])
        
        # print("\nRelated to text i:", related_text_i)
        # print("\tHuman annotations:", dataset[related_to_i]["labels"])
        # print("\tBin-model predictions:", dataset[related_to_i]["predicted_labels"])
        # print("Score with i:", related_smc_score)
        
        # print("\nUnrelated to text i:", unrelated_text_i)
        # print("\tHuman annotations:", dataset[unrelated_to_i]["labels"])
        # print("\tBin-model predictions:", dataset[unrelated_to_i]["predicted_labels"])
        # print("Score with i:", unrelated_smc_score)
    
    # collect unseen datapoint and return
    unseen_dataset = []
    for i in range(len(dataset)):
        if i not in seen_indices:
            unseen_dataset.append(dataset[i])
    
    print("Unseen examples:", len(unseen_dataset))
    print("Seen examples:", len(seen_indices))
    print("Seen pairs:", len(model_inputs))
    
    # shuffle model inputs
    random.shuffle(model_inputs)
    return model_inputs, unseen_dataset

### SCRIPT ###

# load dataset
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
datasets_file_path = base_dir + "/data/multi_label_fig_lang_training.json"
dataset = load_json(datasets_file_path)

model_names = {"roberta-large-joint-fig-lang": "roberta-large", # first train a simple roberta-large model with binary cross entropy loss
               # "roberta-large-sent-transformer-joint-fig-lang": "sentence-transformers/all-roberta-large-v1", # then train a sent-transformer roberta-large model using only binary cross entropy loss
               # "roberta-large-aux-loss-sent-transformer-fig-lang": "all-roberta-large-v1", # then finetune a sent-transformer roberta-large model using auxiliary contrasive loss
               # "roberta-large-aux-loss-sent-transformer-joint-fig-lang": "roberta-large-aux-loss-sent-transformer-fig-lang", # finally do training with binary cross entropy loss
               }

# start finetuning all models
for model_name in model_names.keys(): 
    
    print("\nFINE-TUNING MODEL:", model_name)
    model_file_path = base_dir + "/models/" + model_name
        
    # make output model directory
    os.makedirs(model_file_path, exist_ok = True)
    label2id, id2label = get_label_map(model_file_path)
    
    # this is the auxiliary fine-tuning step
    if "aux-loss" in model_name and "joint" not in model_name: 
        
        # set the model directory for the next model to load
        model_names["roberta-large-aux-loss-sent-transformer-joint-fig-lang"] = model_file_path
    
        # define model and loss function
        model = SentenceTransformer(model_names[model_name])
        cos_sim_loss = losses.CosineSimilarityLoss(model)
    
        # preprocess dataset with regards to the training objectives
        aux_loss_train_examples, unseen_dataset = get_model_inputs_2(dataset)
        aux_loss_data_sampler = SentencesDataset(aux_loss_train_examples, model)
        aux_loss_dataloader = DataLoader(aux_loss_data_sampler, shuffle = True, batch_size = 16)
    
        # params
        num_epochs = 5
        warmup_steps = int(len(aux_loss_dataloader) * num_epochs  * 0.1)
    
        # measure time required for training
        start = time.time()
    
        model.fit(
            train_objectives = [(aux_loss_dataloader, cos_sim_loss)],
            output_path = model_file_path,
            warmup_steps = warmup_steps,
            epochs = num_epochs,
        )
    
        # compute running time
        print("\nFINE-TUNING TIME: ", time.time() - start, " SEC.")
        print("SAVING FINE-TUNINED MODEL: ", model_file_path)
    
    # use remaining training data after auxiliary fine-tuning
    elif "aux-loss" in model_name and "joint" in model_name: 
        dataset = unseen_dataset
        
    # preprocess the dataset using the proper tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_names[model_name])
    train_dataset = get_model_inputs(dataset, tokenizer)
    
    # and load specified model
    model = AutoModelForSequenceClassification.from_pretrained(model_names[model_name],
                                                               problem_type = "multi_label_classification",
                                                               num_labels = len(label2id),
                                                               id2label = id2label,
                                                               label2id = label2id,
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
    print("\nFINE-TUNING TIME: ", time.time() - start, " SEC.")
    
    # save model
    trainer.save_model(model_file_path)
    print("SAVING FINE-TUNINED MODEL: ", model_file_path)
    print("********************************************")
    
    del data_collator
    del tokenizer
    del trainer
    del model
    del args
# -*- coding: utf-8 -*-

import os
import sys
import json
import torch
import random
import warnings
from tqdm import tqdm
from transformers import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# get the absolute path of the current script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../scripts/heuristics/")))
import Metaphors, Sarcasm, Hyperbole, Similes, Irony, Idioms

# ignore all warnings
logging.set_verbosity_error()
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# load data from our json format
def load_json(filename):
    with open(filename, "r") as fp:
        dataset = json.load(fp)
        
    return dataset

# method to convert labels
def expand_literal_label(label2id):
    bin_labels = []
    for feature_name in label2id.keys():
        negative_feature_name = "not_" + feature_name
        bin_labels.append(negative_feature_name)
                
    return bin_labels

# method to verify that human labels and predictions do not contradict
def verify_predictions(predicted_labels, labels, label2id):
    
    # If labels say "literal" we cannot have "feature_X" in the binary labels.
    # But, it's okay if predictions contain "not_feature_X" only.
    if len(labels) == 1:
        if labels[0] == "literal":
            # so, if label is "literal" then it's equivalent with all fig lang 
            # features being "not_"
            labels = expand_literal_label(label2id)
    
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

# method to load appropriate label mappings
def get_label_map_joint_model(filepath):
    
    # load label2id and id2label maps
    if os.path.isfile(os.path.join(filepath, "label2id.json")):
        label2id = load_json(os.path.join(filepath, "label2id.json"))
        id2label = load_json(os.path.join(filepath, "id2label.json"))
        
    else:
        print("No mapping files found in: ", filepath)
        sys.exit(1)
    
    return label2id, id2label

# convert label to integer
def get_int_result(result):
    if "not_" in result:
        return 0
    
    else:
        return 1

# method to predict style class of a single sentence
def binary_ensemble_predict(text, label2id):

    one_hot_prediction = [0] * len(label2id)
    predicted_labels = []

    # metaphor
    metaphor = Metaphors.predict(text)
    one_hot_prediction[label2id['metaphor']] = get_int_result(metaphor)
    predicted_labels.append(metaphor)
    
    # simile
    simile = Similes.predict(text)
    one_hot_prediction[label2id['simile']] = get_int_result(simile)
    predicted_labels.append(simile)
    
    # idioms
    idioms = Idioms.predict(text)
    one_hot_prediction[label2id['idiom']] = get_int_result(idioms)
    predicted_labels.append(idioms)
    
    # sarcasm
    sarcasm = Sarcasm.predict(text)
    one_hot_prediction[label2id['sarcasm']] = get_int_result(sarcasm)
    predicted_labels.append(sarcasm)
    
    # irony
    irony = Irony.predict(text)
    one_hot_prediction[label2id['irony']] = get_int_result(irony)
    predicted_labels.append(irony)
    
    # hyperbole
    hyperbole = Hyperbole.predict(text)
    one_hot_prediction[label2id['hyperbole']] = get_int_result(hyperbole)
    predicted_labels.append(hyperbole)
    
    # package probability scores and prediction into a neat dict
    predictions = {"predicted_labels": predicted_labels,
                   "one_hot_prediction": one_hot_prediction,
                   }
    
    return predictions

# method to predict style class of a single sentence
def joint_predict(text, model, tokenizer, id2label, thresholds):

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
        predictions = {"predicted_labels": prediction,
                       "one_hot_prediction": one_hot_prediction,
                       }

    return predictions

# method to run the prediction loop for the dataset
def prediction_loop(dataset, name, joint_model, joint_tokenizer, 
                    joint_label2id, joint_id2label, joint_thresholds):
    
    binary_predictions = []
    joint_predictions = []
    labels = []
    texts = []
    
    for i in tqdm(range(len(dataset)), desc = name.upper() + " SET PROGRESS", ncols = 100):
        text = dataset[i]["text"]
        # predict with joint model
        joint_prediction = joint_predict(text, 
                                         joint_model, 
                                         joint_tokenizer, 
                                         joint_id2label, 
                                         joint_thresholds)
        
        # ensemble prediction with all binary models
        binary_prediction = binary_ensemble_predict(text, 
                                                    joint_label2id)
        
        # save predictions in the lists
        joint_predictions.append(joint_prediction["predicted_labels"])
        binary_predictions.append(binary_prediction["predicted_labels"])
        labels.append(dataset[i]["labels"])
        texts.append(text)
        
    return texts, labels, binary_predictions, joint_predictions

# method to check which indices agree on label results
def check_agreement(labels, binary_predictions, joint_predictions, label2id):
    models_agree_with_gt_indices = []
    joint_agrees_with_gt_inidices = []
    binary_agrees_with_gt_indices = []
    models_disagree_with_gt_indices = []
    
    models_agree_indices = []
    models_disagree_indices = []
    
    # loop through predictions and ground truths
    for i in range(len(labels)):
        pr_b_labels = binary_predictions[i]
        pr_j_labels = joint_predictions[i]
        gt_labels = labels[i]        

        # check if joint labels violate GT
        joint_agrees_with_labels = verify_predictions(pr_j_labels, gt_labels, label2id)

        # check if binary labels violate GT
        binary_agrees_with_labels = verify_predictions(pr_b_labels, gt_labels, label2id)
           
        # check if joint labels violate violate binary labels
        joint_agrees_with_binary = verify_predictions(pr_j_labels, pr_b_labels, label2id)
        
        # check when both joint and binary predictions agree with GT
        if joint_agrees_with_labels and binary_agrees_with_labels:
            models_agree_with_gt_indices.append(i)
        
        # check when joint predictions agree with GT, but binary predictions disagree
        elif joint_agrees_with_labels and not binary_agrees_with_labels:
            joint_agrees_with_gt_inidices.append(i)
        
        # check when binary predictions agree with GT, but joint predictions disagree
        elif not joint_agrees_with_labels and binary_agrees_with_labels:
            binary_agrees_with_gt_indices.append(i)
            
        else:
            models_disagree_with_gt_indices.append(i)
        
        # check if models agree with each other
        if joint_agrees_with_binary:
            models_agree_indices.append(i)
        
        else:
            models_disagree_indices.append(i)
            
    return models_agree_with_gt_indices, joint_agrees_with_gt_inidices, binary_agrees_with_gt_indices, models_disagree_with_gt_indices, models_agree_indices, models_disagree_indices

### SCRIPT ###

# load dataset and label maps
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
multi_label_dataset = load_json(base_dir + "/data/multi-label-fig-lang-dataset.json")

# joint model
# load maps, thresholds, tokenizer, and model
joint_model_name = "roberta-large-joint-fig-lang"
joint_model_file_path = base_dir + "/models/" + joint_model_name

print("LOADING JOINT MODEL: ", joint_model_name, end = "...")
joint_tokenizer = AutoTokenizer.from_pretrained(joint_model_file_path)
joint_label2id, joint_id2label = get_label_map_joint_model(joint_model_file_path)
joint_thresholds = load_json(os.path.join(joint_model_file_path, "thresholds_bin.json"))
joint_model = AutoModelForSequenceClassification.from_pretrained(joint_model_file_path).to("cuda")
joint_model.eval() # lock model in eval mode
print(" DONE.")

# predict for all test sets
for dataset_name in multi_label_dataset.keys():
    
    # skip training sets
    if "-train" in dataset_name:
        continue
    
    # get specified dataset
    dataset = multi_label_dataset[dataset_name]
    
    # run prediction loop and get predictions
    texts, labels, binary_predictions, joint_predictions = prediction_loop(dataset, 
                                                                           dataset_name, 
                                                                           joint_model, 
                                                                           joint_tokenizer, 
                                                                           joint_label2id, 
                                                                           joint_id2label, 
                                                                           joint_thresholds)
        
    # get the indices where models and GT agree, disagree, etc.
    models_agree_with_gt_indices, joint_agrees_with_gt_inidices, binary_agrees_with_gt_indices, models_disagree_with_gt_indices, models_agree_indices, models_disagree_indices = check_agreement(labels, binary_predictions, joint_predictions, joint_label2id)
    n = 3
    print("\n******************************************************************")
    print("Error analysis for ", dataset_name)
    # pick N examples to print from the above categories
    for j in range(n):
        if len(models_agree_with_gt_indices) > 0:
            if j == 0:
                print("")
                print(len(models_agree_with_gt_indices), "\tExamples where joint and binary model predictions agree with Ground Truth...")
                
            i = random.choice(models_agree_with_gt_indices)
            print("\tExample " + str(j) + ": " + texts[i].encode('cp1252', errors='ignore').decode('cp1252'))
            print("\t\tGround Thruth: ", labels[i])
            
    # pick N examples to print from the above categories
    for j in range(n):
        if len(joint_agrees_with_gt_inidices) > 0:
            if j == 0:
                print("")
                print(len(joint_agrees_with_gt_inidices), "\tExamples where only joint model predictions agrees with Ground Truth...")
                
            i = random.choice(joint_agrees_with_gt_inidices)
            print("\tExample " + str(j) + ": " + texts[i].encode('cp1252', errors='ignore').decode('cp1252'))
            print("\t\tGround Thruth: ", labels[i])
            print("\t\tBinary Preds: ", binary_predictions[i])
            
    # pick N examples to print from the above categories
    for j in range(n):
        if len(binary_agrees_with_gt_indices) > 0:
            if j == 0:
                print("")
                print(len(binary_agrees_with_gt_indices), "\tExamples where only binary model predictions agrees with Ground Truth...")
                
            i = random.choice(binary_agrees_with_gt_indices)
            print("\tExample " + str(j) + ": " + texts[i].encode('cp1252', errors='ignore').decode('cp1252'))
            print("\t\tGround Thruth: ", labels[i])
            print("\t\tJoint Preds: ", joint_predictions[i])
            
    # pick N examples to print from the above categories
    for j in range(n):
        if len(models_disagree_with_gt_indices) > 0:
            if j == 0:
                print("")
                print(len(models_disagree_with_gt_indices), "\tExamples where model predictions disagree with Ground Truth...")
                
            i = random.choice(models_disagree_with_gt_indices)
            print("\tExample " + str(j) + ": " + texts[i].encode('cp1252', errors='ignore').decode('cp1252'))
            print("\t\tGround Thruth: ", labels[i])
            print("\t\tBinary Preds: ", binary_predictions[i])
            print("\t\tJoint Preds: ", joint_predictions[i])
            
    # pick N examples to print from the above categories
    for j in range(n):
        if len(models_agree_indices) > 0:
            if j == 0:
                print("")
                print(len(models_agree_indices), "\tExamples where model predictions agree with eachother...")
                
            i = random.choice(models_agree_indices)
            print("\tExample " + str(j) + ": " + texts[i].encode('cp1252', errors='ignore').decode('cp1252'))
            print("\t\tGround Thruth: ", labels[i])
            print("\t\tPredictions: ", joint_predictions[i])
            
    # pick N examples to print from the above categories
    for j in range(n):
        if len(models_disagree_indices) > 0:
            if j == 0:
                print("")
                print(len(models_disagree_indices), "\tExamples where model predictions disagree with eachother...")
                
            i = random.choice(models_disagree_indices)
            print("\tExample " + str(j) + ": " + texts[i].encode('cp1252', errors='ignore').decode('cp1252'))
            print("\t\tGround Thruth: ", labels[i])
            print("\t\tBinary Preds: ", binary_predictions[i])
            print("\t\tJoint Preds: ", joint_predictions[i])
            
    print("******************************************************************\n")
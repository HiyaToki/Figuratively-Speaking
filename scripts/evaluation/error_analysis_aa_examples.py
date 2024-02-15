# -*- coding: utf-8 -*-

import os
import json
import random
import warnings
from sklearn.neural_network import MLPClassifier

# ignore warnings
warnings.filterwarnings("ignore")

# method to save json file
def save_json(data, filename):
    # open file and load
    with open(filename, "w") as fp:
        json.dump(data, fp)

# method to load a json file to memory
def load_json(filename):
    # open file and load
    with open(filename, "r", encoding = "latin1") as fp:
        data = json.load(fp)

    return data

# method to get features and labels from dataset
def get_x_y(dataset, key):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    t_test = []
    
    # get features and authors as labels
    for i in range(len(dataset)):
        y = dataset[i]["author"]
        t = dataset[i]["text"]
        
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
            t_test.append(t)
            
        else:
            x_train.append(x)
            y_train.append(y)
            
    return x_train, y_train, x_test, y_test, t_test

# method to check which indices agree on label results
def check_agreement(labels, char_predictions, joint_predictions):
    char_agrees_with_gt_indices = []
    models_agree_with_gt_indices = []
    joint_agrees_with_gt_inidices = []
    models_disagree_with_gt_indices = []
    
    models_agree_indices = []
    models_disagree_indices = []
    
    # loop through predictions and ground truths
    for i in range(len(labels)):
        pr_c_label = char_predictions[i]
        pr_j_label = joint_predictions[i]
        gt_label = labels[i]
        
        if pr_c_label == gt_label:
            char_agrees_with_labels = True
            
        else:
            char_agrees_with_labels = False
        
        if pr_j_label == gt_label:
            joint_agrees_with_labels = True
            
        else:
            joint_agrees_with_labels = False
            
        if pr_j_label == pr_c_label:
            joint_agrees_with_char = True
            
        else:
            joint_agrees_with_char = False
            

        # check when both joint and binary predictions agree with GT
        if joint_agrees_with_labels and char_agrees_with_labels:
            models_agree_with_gt_indices.append(i)
        
        # check when joint predictions agree with GT, but binary predictions disagree
        elif joint_agrees_with_labels and not char_agrees_with_labels:
            joint_agrees_with_gt_inidices.append(i)
        
        # check when binary predictions agree with GT, but joint predictions disagree
        elif not joint_agrees_with_labels and char_agrees_with_labels:
            char_agrees_with_gt_indices.append(i)
            
        else:
            models_disagree_with_gt_indices.append(i)
        
        # check if models agree with each other
        if joint_agrees_with_char:
            models_agree_indices.append(i)
        
        else:
            models_disagree_indices.append(i)
            
    return models_agree_with_gt_indices, joint_agrees_with_gt_inidices, char_agrees_with_gt_indices, models_disagree_with_gt_indices, models_agree_indices, models_disagree_indices


### SCRIPT ###
keys = ["joint_model_embedding",
        "char_n_gram_tfidf_features"
        ]

# load dataset
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
datasets_file_path = base_dir + "/data/pan18/pan18-authorship-attribution-test-dataset.json"
dataset = load_json(datasets_file_path)
results = dict()

# set specific problem
problem = "problem00002"
dataset_name = "PAN18 " + problem

texts = []
labels = []
char_predictions = []
joint_predictions = []
    
for key in keys:
    print("TRAINING MODEL ON " + problem.upper() + " WITH " + key.upper() + " FEATURES...", end = " ")
    x_train, y_train, x_test, y_test, t_test = get_x_y(dataset[problem], key)
        
    # create and train the MLP model
    mlp = MLPClassifier(hidden_layer_sizes = (1024,), 
                        learning_rate_init = 2e-5,
                        activation = 'relu', 
                        random_state = 42,
                        max_iter = 1000, 
                        solver = 'adam', 
                        )
    
    # fit MLP using training set
    mlp.fit(x_train, y_train)
    
    # predict on test set
    y_pred = mlp.predict(x_test)
    
    texts = t_test
    labels = y_test
    if key == "joint_model_embedding":
        joint_predictions = y_pred
        
    elif key == "char_n_gram_tfidf_features":
        char_predictions = y_pred
    
    else:
        pass
    
    
    print("DONE!")
print("")

# get the indices where models and GT agree, disagree, etc.
models_agree_with_gt_indices, joint_agrees_with_gt_inidices, char_agrees_with_gt_indices, models_disagree_with_gt_indices, models_agree_indices, models_disagree_indices = check_agreement(labels, char_predictions, joint_predictions)
n = 5

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
        print("\t\tBinary Preds: ", char_predictions[i])
        
# pick N examples to print from the above categories
for j in range(n):
    if len(char_agrees_with_gt_indices) > 0:
        if j == 0:
            print("")
            print(len(char_agrees_with_gt_indices), "\tExamples where only binary model predictions agrees with Ground Truth...")
            
        i = random.choice(char_agrees_with_gt_indices)
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
        print("\t\tBinary Preds: ", char_predictions[i])
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
        print("\t\tBinary Preds: ", char_predictions[i])
        print("\t\tJoint Preds: ", joint_predictions[i])
        
print("******************************************************************\n")

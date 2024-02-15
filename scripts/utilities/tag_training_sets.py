
import os
import sys
import json

# get the absolute path of the current script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../scripts/heuristics/")))
import Metaphors, Sarcasm, Hyperbole, Similes, Irony, Idioms

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

# convert label to integer
def get_int_result(result):
    if "not_" in result:
        return 0
    
    else:
        return 1

# method to use heuristic model to tag sentences
def extract(text):
    one_hot_prediction = [0] * len(label2index)
    predicted_labels = []

    # metaphor
    metaphor = Metaphors.predict(text)
    one_hot_prediction[label2index['metaphor']] = get_int_result(metaphor)
    predicted_labels.append(metaphor)
    
    # simile
    simile = Similes.predict(text)
    one_hot_prediction[label2index['simile']] = get_int_result(simile)
    predicted_labels.append(simile)
    
    # idioms
    idioms = Idioms.predict(text)
    one_hot_prediction[label2index['idiom']] = get_int_result(idioms)
    predicted_labels.append(idioms)
    
    # sarcasm
    sarcasm = Sarcasm.predict(text)
    one_hot_prediction[label2index['sarcasm']] = get_int_result(sarcasm)
    predicted_labels.append(sarcasm)
    
    # irony
    irony = Irony.predict(text)
    one_hot_prediction[label2index['irony']] = get_int_result(irony)
    predicted_labels.append(irony)
    
    # hyperbole
    hyperbole = Hyperbole.predict(text)
    one_hot_prediction[label2index['hyperbole']] = get_int_result(hyperbole)
    predicted_labels.append(hyperbole)
    
    return one_hot_prediction, predicted_labels

# method to tag training sets with multi-label fig lang features
def tag_training(base_dir, multi_label_dataset):
    
    for dataset_name in multi_label_dataset.keys():
        
        # skip testing sets from multi-label tagging
        if "-test" in dataset_name:
            continue
        
        # get all examples under the specified dataset name
        for i in range(len(multi_label_dataset[dataset_name])):

            # get Fig lan predictions using binary models
            one_hot_prediction, predicted_labels = extract(multi_label_dataset[dataset_name][i]["text"])
            
            # add results into the dataset
            multi_label_dataset[dataset_name][i]["one_hot_prediction"] = one_hot_prediction
            multi_label_dataset[dataset_name][i]["predicted_labels"] = predicted_labels
        
    # save to file
    file_path = base_dir +"/data/multi-label-fig-lang-dataset.json"
    if not os.path.isfile(file_path):
        save_json(file_path, multi_label_dataset)

### SCRIPT ###
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
dataset_file_path = base_dir + "/data/fig-lang-dataset.json"
multi_label_dataset = load_json(dataset_file_path)
tag_training(base_dir, multi_label_dataset)

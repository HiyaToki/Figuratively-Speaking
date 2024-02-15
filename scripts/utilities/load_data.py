# -*- coding: utf-8 -*-

import os
import csv
import json
import xmltodict
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from nltk.tokenize.treebank import TreebankWordDetokenizer

global multi_label_dataset
multi_label_dataset = dict() 

# method to save all loaded data into our json format
def save_json(filename, dataset):
    with open(filename, "w") as fp:
        json.dump(dataset, fp)

# load data from our json format
def load_json(filename):
    print("Loading file: ", filename)
    with open(filename, "r") as fp:
        dataset = json.load(fp)
        
    return dataset

def dataset_statistics(multi_label_dataset, key):
    if key == "stats":
        label2count = dict()
        for key_1 in multi_label_dataset.keys():
            for i in range(len(multi_label_dataset[key_1])):
                for label in multi_label_dataset[key_1][i]["labels"]:
                    if label not in label2count:
                        label2count[label] = set()

                    label2count[label].add(key_1)
                    
        print("\nClass to dataset: ")
        for label in label2count.keys():
            corpora = [name.replace("-train", "").replace("-test", "") for name in list(label2count[label])]
            print("\tClass: ", label, " -> ", ", ".join(set(corpora)))
    
    elif key == "all":        
        for key_1 in multi_label_dataset.keys():
            label2count = dict()
            
            for i in range(len(multi_label_dataset[key_1])):
                for label in multi_label_dataset[key_1][i]["labels"]:
                    if label not in label2count:
                        label2count[label] = 0

                    label2count[label] += 1
                    
            print("\nLabel distribution in", key_1)
            for label in label2count.keys():
                print("\tLabel: ", label, " -> ", label2count[label])
                
    elif key == "all-type":        
        label2count = dict()
        for key_1 in multi_label_dataset.keys():
            key_1_mod = key_1.replace("-train", "").replace("-test", "")
            
            if key_1_mod not in label2count:
                label2count[key_1_mod] = dict()
            
            for i in range(len(multi_label_dataset[key_1])):
                for label in multi_label_dataset[key_1][i]["labels"]:
                    if label not in label2count[key_1_mod]:
                        label2count[key_1_mod][label] = 0

                    label2count[key_1_mod][label] += 1
                    
        
        for key_1_mod, labels in label2count.items():
            print("\nLabel distribution in", key_1_mod)
            for label in labels:
                print("\tLabel: ", label, " -> ", label2count[key_1_mod][label])
        
    elif key == "total":
        label2count = dict()
        for key_1 in multi_label_dataset.keys():
            for i in range(len(multi_label_dataset[key_1])):
                for label in multi_label_dataset[key_1][i]["labels"]:
                    if label not in label2count:
                        label2count[label] = 0

                    label2count[label] += 1
                    
        print("\nTotal label distribution: ")
        for label in label2count.keys():
            print("\tLabel: ", label, " -> ", label2count[label])
                    
    else:
        label2count = dict()
        for i in range(len(multi_label_dataset[key])):
            for label in multi_label_dataset[key][i]["labels"]:
                if label not in label2count:
                    label2count[label] = 0
    
                label2count[label] += 1

        print("\nLabel distribution in", key)
        for label in label2count.keys():
            print("\tLabel: ", label, " -> ", label2count[label])

# method to preprocess text
def prepcocess_sentence(text):
    text = text.replace("‘", "'")
    text = text.replace("’", "'")
    text = text.replace("<b>", "")
    text = text.replace("</b>", "")
    return text

# method to de-tokenize text
def detokenize_sentence(text): 
    text = prepcocess_sentence(text) # convert to tokens and detokenize 
    sentence = TreebankWordDetokenizer().detokenize(text.split())
    return sentence

# define a function to remove the nested tags
def remove_nested_tags(xml_data, nested_tags = ["<LmTarget>", "</LmTarget>", "<LmSource>", "</LmSource>"]):
    for tag in nested_tags:
        xml_data = xml_data.replace(tag, "")
        
    return xml_data

# method to load irony sem eval 2018
def load_irony_eval(filepath):
    iron_classes = []
    iron_sentences = []
    print("Loading Irony SemEval 2018 dataset...", end = " ")
    
    # open train labels
    with open(os.path.join(filepath, "train_labels.txt"), "r", encoding = 'utf-8') as fp:
        for line in fp:
            if line.strip() == "1":
                iron_classes.append(["irony"])
                
            else:
                iron_classes.append(["not_irony"])
     
    # load train text
    with open(os.path.join(filepath, "train_text.txt"), "r", encoding = 'utf-8') as fp:
        for line in fp:
                iron_sentences.append(prepcocess_sentence(line.strip()))
                
    # open val labels, merge with train set
    with open(os.path.join(filepath, "val_labels.txt"), "r", encoding = 'utf-8') as fp:
        for line in fp:
            if line.strip() == "1":
                iron_classes.append(["irony"])
                
            else:
                iron_classes.append(["not_irony"])
                
    # load val text, merge with train set
    with open(os.path.join(filepath, "val_text.txt"), "r", encoding = 'utf-8') as fp:
        for line in fp:
                iron_sentences.append(prepcocess_sentence(line.strip()))
                
    # store train set
    for i in range(len(iron_sentences)):
        multi_label_example = dict()
        multi_label_example["text"] = iron_sentences[i]
        
        # this dataset is binary so labels will contain "literal" or "irony"
        multi_label_example["labels"] = iron_classes[i]
        
        if "Irony-Eval18-train" not in multi_label_dataset:
            multi_label_dataset["Irony-Eval18-train"] = []
            
        # append to multilabel example
        multi_label_dataset["Irony-Eval18-train"].append(multi_label_example)
        
    # reset
    iron_classes = []
    iron_sentences = []
    
    # open test labels
    with open(os.path.join(filepath, "test_labels.txt"), "r", encoding = 'utf-8') as fp:
        for line in fp:
            if line.strip() == "1":
                iron_classes.append(["irony"])
                
            else:
                iron_classes.append(["not_irony"])
                
    # load test texts
    with open(os.path.join(filepath, "test_text.txt"), "r", encoding = 'utf-8') as fp:
        for line in fp:
                iron_sentences.append(prepcocess_sentence(line.strip()))
                
    # store train set
    for i in range(len(iron_sentences)):
        multi_label_example = dict()
        multi_label_example["text"] = iron_sentences[i]
        
        # this dataset is binary so labels will contain "literal" or "irony"
        multi_label_example["labels"] = iron_classes[i]
        
        if "Irony-Eval18-test" not in multi_label_dataset:
            multi_label_dataset["Irony-Eval18-test"] = []
            
        # append example
        multi_label_dataset["Irony-Eval18-test"].append(multi_label_example)
    
    print("Irony SemEval18 dataset loaded. ")
    print("\t", len(multi_label_dataset["Irony-Eval18-train"]), "training examples. ")
    print("\t", len(multi_label_dataset["Irony-Eval18-test"]), "testing examples. ")
    
# method to load iScarsam Eval 2022 Dataset
def load_isarcasm(filepath):
    sarc_classes = []
    sarc_sentences = []   
    print("Loading iScarsam Eval 2022 dataset...", end = " ")
    
    # load iSarcasm training set
    with open(os.path.join(filepath, "train.csv"), "r", encoding = "utf-8") as csv_file:
        reader = csv.reader(csv_file)
        
        for row in reader: # loop through csv lines
            if "id" in row[0]: # skip title line
                continue
            
            if row[2] == "1": # this example also provides a literal rephrase
                labels = []
                if row[4] == "1":
                    labels.append("sarcasm")
                    
                else:
                    labels.append("not_sarcasm")
                    
                if row[5] == "1":
                    labels.append("irony")
                    
                else:
                    labels.append("not_irony")
                    
                if row[8] == "1":
                    labels.append("hyperbole")
                    
                else:
                    labels.append("not_hyperbole")
                    
                # dataset also has satire and understatement
                # but we are not interested in these
                if len(labels) > 0: # check if there are any valid labels
                    sentence = prepcocess_sentence(row[1])
                    sarc_sentences.append(sentence)
                    sarc_classes.append(labels)
                
                # the rephrase of the original sarcastic sentence
                literal_sent = prepcocess_sentence(row[3])
                sarc_sentences.append(literal_sent)
                sarc_classes.append(["not_sarcasm", "not_irony", "not_hyperbole"])
            
            else: # just a literal example
                sentence = prepcocess_sentence(row[1])
                sarc_classes.append(["not_sarcasm", "not_irony", "not_hyperbole"])
                sarc_sentences.append(sentence)
                
    # store train set
    for i in range(len(sarc_sentences)):
        multi_label_example = dict()
        multi_label_example["text"] = sarc_sentences[i]
        
        # this dataset is multi-label so labels will contain "literal" or "sarcasm", "irony" or "hyperbole"
        multi_label_example["labels"] = sarc_classes[i]
        
        if "iSarcasm-train" not in multi_label_dataset:
            multi_label_dataset["iSarcasm-train"] = []
            
        # append to multilabel example
        multi_label_dataset["iSarcasm-train"].append(multi_label_example)
            
    # reset
    sarc_classes = []
    sarc_sentences = []
    
    # load iSarcasm test set (TASK B test set)
    with open(os.path.join(filepath, "test.csv"), 'r', encoding = 'utf-8') as csv_file:
        reader = csv.reader(csv_file)
        
        for row in reader: # loop through csv lines
            if "id" in row[0]: # skip title line
                continue
            
            labels = []
            if row[1] == "1":
                labels.append("sarcasm")
                
            else:
                labels.append("not_sarcasm")
                
            if row[2] == "1":
                labels.append("irony")
                
            else:
                labels.append("not_irony")
                
            if row[5] == "1":
                labels.append("hyperbole")
            else:
                labels.append("not_hyperbole")
                
            # dataset also has satire and understatement
            # but we are not interested in those
            if len(labels) > 0: # check if there are any valid labels
                sentence = prepcocess_sentence(row[0])
                sarc_sentences.append(sentence)
                sarc_classes.append(labels)
                
            else:
                sentence = prepcocess_sentence(row[0])
                sarc_classes.append(["not_sarcasm", "not_irony", "not_hyperbole"])
                sarc_sentences.append(sentence)
                
    # store test set
    for i in range(len(sarc_sentences)):
        multi_label_example = dict()
        multi_label_example["text"] = sarc_sentences[i]
        
        # this dataset is multi-label so labels will contain "literal" or "sarcasm", "irony" or "hyperbole"
        multi_label_example["labels"] = sarc_classes[i]
        
        if "iSarcasm-test" not in multi_label_dataset:
            multi_label_dataset["iSarcasm-test"] = []
            
        # append to multilabel example
        multi_label_dataset["iSarcasm-test"].append(multi_label_example)
        
    print("iSarcasm Eval 2022 dataset loaded. ")
    print("\t", len(multi_label_dataset["iSarcasm-train"]), "training examples. ")
    print("\t", len(multi_label_dataset["iSarcasm-test"]), "testing examples. ")
    
# method to load flute-m4 dataset
def load_flute_m4(filepath):
    flute_classes = []
    flute_sentences = []
    print("Loading FLUTE-metaphor dataset...", end = " ")
    
    # load FLUTE-metaphor training set
    with open(os.path.join(filepath, "metaphor_train.jsonl"), "r", encoding = "utf-8") as fp:
        for line in fp: # loop through each line
            data = json.loads(line)
            
            # look at entailment pairs
            if data["label"] == "Entailment":
                # get metaphoric sentence
                sentence = detokenize_sentence(data["hypothesis"])
                flute_sentences.append(sentence)
                
                labels = ["metaphor"]
                flute_classes.append(labels)
                
                # now get the literal sentence
                sentence = detokenize_sentence(data["premise"])
                flute_sentences.append(sentence)
                
                labels = ["literal"]
                flute_classes.append(labels)
            
            # look at contradiction pairs
            elif data["label"] == "Contradiction":
                
                # now get the other literal sentence
                sentence = detokenize_sentence(data["premise"])
                flute_sentences.append(sentence)
                
                labels = ["literal"]
                flute_classes.append(labels)
                
        # store train set
        for i in range(len(flute_sentences)):
            multi_label_example = dict()
            multi_label_example["text"] = flute_sentences[i]
            
            # this dataset is binary so labels will contain "literal" or "metaphor"
            multi_label_example["labels"] = flute_classes[i]
            
            if "FLUTE-metaphor-train" not in multi_label_dataset:
                multi_label_dataset["FLUTE-metaphor-train"] = []
                
            # append to multilabel example
            multi_label_dataset["FLUTE-metaphor-train"].append(multi_label_example)
    
    # and reset
    flute_classes = []
    flute_sentences = []
    
    # load FLUTE-metaphor test set
    with open(os.path.join(filepath, "metaphor_test.jsonl"), "r") as fp:
        for line in fp: # loop through each line
            data = json.loads(line)
            
            # look only at entailment pairs
            if data["label"] == "Entailment":
                # get metaphoric sentence
                sentence = detokenize_sentence(data["hypothesis"])
                flute_sentences.append(sentence)
                
                labels = ["metaphor"]
                flute_classes.append(labels)
                
                # now get the literal sentence
                sentence = detokenize_sentence(data["premise"])
                flute_sentences.append(sentence)
                
                labels = ["literal"]
                flute_classes.append(labels)
                
            # look at contradiction pairs
            elif data["label"] == "Contradiction":
                
                # now get the other literal sentence
                sentence = detokenize_sentence(data["premise"])
                flute_sentences.append(sentence)
                
                labels = ["literal"]
                flute_classes.append(labels)
                
        # store test set
        for i in range(len(flute_sentences)):
            multi_label_example = dict()
            multi_label_example["text"] = flute_sentences[i]
            
            # this dataset is binary so labels will contain "literal" or "metaphor"
            multi_label_example["labels"] = flute_classes[i]
            
            if "FLUTE-metaphor-test"not in multi_label_dataset:
                multi_label_dataset["FLUTE-metaphor-test"] = []
                
            # append to multilabel example
            multi_label_dataset["FLUTE-metaphor-test"].append(multi_label_example)
            
    print("FLUTE-metaphor dataset loaded. ")
    print("\t", len(multi_label_dataset["FLUTE-metaphor-train"]), "training examples. ")
    print("\t", len(multi_label_dataset["FLUTE-metaphor-test"]), "testing examples. ")
                             
# method to load flute-idiom dataset
def load_flute_idiom(filepath):
    flute_classes = []
    flute_sentences = []
    print("Loading FLUTE-idiom dataset...", end = " ")
    
    # load FLUTE-idiom training set
    with open(os.path.join(filepath, "idiom_train.jsonl"), "r") as fp:
        for line in fp: # loop through each line
            data = json.loads(line)
            
            # look at entailment pairs
            if data["label"] == "Entailment":
                # get idiomatic sentence
                sentence = detokenize_sentence(data["hypothesis"])
                flute_sentences.append(sentence)
                
                labels = ["idiom"]
                flute_classes.append(labels)
                
                # now get the literal sentence
                sentence = detokenize_sentence(data["premise"])
                flute_sentences.append(sentence)
                
                labels = ["literal"]
                flute_classes.append(labels)

            # look at contradiction pairs
            elif data["label"] == "Contradiction":
                    
                # now get the other literal sentence
                sentence = detokenize_sentence(data["premise"])
                flute_sentences.append(sentence)
                
                labels = ["literal"]
                flute_classes.append(labels)
                          
        # store train set
        for i in range(len(flute_sentences)):
            multi_label_example = dict()
            multi_label_example["text"] = flute_sentences[i]
            
            # this dataset is binary so labels will contain "literal" or "idiom"
            multi_label_example["labels"] = flute_classes[i]
            
            if "FLUTE-idiom-train" not in multi_label_dataset:
                multi_label_dataset["FLUTE-idiom-train"] = []
                
            # append to multilabel example
            multi_label_dataset["FLUTE-idiom-train"].append(multi_label_example)
    
    # and reset
    flute_classes = []
    flute_sentences = []
    
    # load FLUTE-idiom test set
    with open(os.path.join(filepath, "idiom_test.jsonl"), "r") as fp:
        for line in fp: # loop through each line
            data = json.loads(line)
            
            # look at entailment pairs
            if data["label"] == "Entailment":
                # get idiomatic sentence
                sentence = detokenize_sentence(data["hypothesis"])
                flute_sentences.append(sentence)
                
                labels = ["idiom"]
                flute_classes.append(labels)
                      
                # now get the literal sentence
                sentence = detokenize_sentence(data["premise"])
                flute_sentences.append(sentence)
                
                labels = ["literal"]
                flute_classes.append(labels)
            
            # look at contradiction pairs
            elif data["label"] == "Contradiction":
                      
                # now get the other literal sentence
                sentence = detokenize_sentence(data["premise"])
                flute_sentences.append(sentence)
                
                labels = ["literal"]
                flute_classes.append(labels)
                
        # store test set
        for i in range(len(flute_sentences)):
            multi_label_example = dict()
            multi_label_example["text"] = flute_sentences[i]
            
            # this dataset is binary so labels will contain "literal" or "idiom"
            multi_label_example["labels"] = flute_classes[i]
            
            if "FLUTE-idiom-test" not in multi_label_dataset:
                multi_label_dataset["FLUTE-idiom-test"] = []
                
            # append to multilabel example
            multi_label_dataset["FLUTE-idiom-test"].append(multi_label_example)
            
    print("FLUTE-idiom dataset loaded. ")
    print("\t", len(multi_label_dataset["FLUTE-idiom-train"]), "training examples. ")
    print("\t", len(multi_label_dataset["FLUTE-idiom-test"]), "testing examples. ")

# method to load flute-simile dataset
def load_flute_simile(filepath):
    flute_classes = []
    flute_sentences = []
    print("Loading FLUTE-simile dataset...", end = " ")
    
    # load FLUTE-simile training set
    with open(os.path.join(filepath, "simile_train.jsonl"), "r") as fp:
        for line in fp: # loop through each line
            data = json.loads(line)
            
            # look at entailment pairs
            if data["label"] == "Entailment":
                # get simileatic sentence
                sentence = detokenize_sentence(data["hypothesis"])
                flute_sentences.append(sentence)
                
                labels = ["simile"]
                flute_classes.append(labels)
                
                # now get the literal sentence
                sentence = detokenize_sentence(data["premise"])
                flute_sentences.append(sentence)
                
                labels = ["literal"]
                flute_classes.append(labels)

            # look at contradiction pairs
            elif data["label"] == "Contradiction":
                    
                # now get the other literal sentence
                sentence = detokenize_sentence(data["premise"])
                flute_sentences.append(sentence)
                
                labels = ["literal"]
                flute_classes.append(labels)
                          
        # store train set
        for i in range(len(flute_sentences)):
            multi_label_example = dict()
            multi_label_example["text"] = flute_sentences[i]
            
            # this dataset is binary so labels will contain "literal" or "simile"
            multi_label_example["labels"] = flute_classes[i]
            
            if "FLUTE-simile-train" not in multi_label_dataset:
                multi_label_dataset["FLUTE-simile-train"] = []
                
            # append to multilabel example
            multi_label_dataset["FLUTE-simile-train"].append(multi_label_example)
    
    # and reset
    flute_classes = []
    flute_sentences = []
    
    # load FLUTE-simile test set
    with open(os.path.join(filepath, "simile_test.jsonl"), "r") as fp:
        for line in fp: # loop through each line
            data = json.loads(line)
            
            # look at entailment pairs
            if data["label"] == "Entailment":
                # get simileatic sentence
                sentence = detokenize_sentence(data["hypothesis"])
                flute_sentences.append(sentence)
                
                labels = ["simile"]
                flute_classes.append(labels)
                      
                # now get the literal sentence
                sentence = detokenize_sentence(data["premise"])
                flute_sentences.append(sentence)
                
                labels = ["literal"]
                flute_classes.append(labels)
            
            # look at contradiction pairs
            elif data["label"] == "Contradiction":
                      
                # now get the other literal sentence
                sentence = detokenize_sentence(data["premise"])
                flute_sentences.append(sentence)
                
                labels = ["literal"]
                flute_classes.append(labels)
                
        # store test set
        for i in range(len(flute_sentences)):
            multi_label_example = dict()
            multi_label_example["text"] = flute_sentences[i]
            
            # this dataset is binary so labels will contain "literal" or "simile"
            multi_label_example["labels"] = flute_classes[i]
            
            if "FLUTE-simile-test" not in multi_label_dataset:
                multi_label_dataset["FLUTE-simile-test"] = []
                
            # append to multilabel example
            multi_label_dataset["FLUTE-simile-test"].append(multi_label_example)
            
    print("FLUTE-simile dataset loaded. ")
    print("\t", len(multi_label_dataset["FLUTE-simile-train"]), "training examples. ")
    print("\t", len(multi_label_dataset["FLUTE-simile-test"]), "testing examples. ")

# method to load flute-sarcasm dataset
def load_flute_sarcasm(filepath):
    flute_classes = []
    flute_sentences = []
    print("Loading FLUTE-sarcasm dataset...", end = " ")
    
    # load FLUTE-sarcasm training set
    with open(os.path.join(filepath, "sarcasm_train.jsonl"), "r") as fp:
        for line in fp: # loop through each line
            data = json.loads(line)
            
            # look at entailment pairs
            if data["label"] == "Entailment":
                # get sarcasmatic sentence
                sentence = detokenize_sentence(data["hypothesis"])
                flute_sentences.append(sentence)
                
                # in entailment, hypothesis are not sarcasm, but also not guarantee literal
                labels = ["not_sarcasm"]
                flute_classes.append(labels)
                
                # now get the literal sentence
                sentence = detokenize_sentence(data["premise"])
                flute_sentences.append(sentence)
                
                labels = ["literal"]
                flute_classes.append(labels)

            # look at contradiction pairs
            elif data["label"] == "Contradiction":
                    
                # now get the other literal sentence
                sentence = detokenize_sentence(data["hypothesis"])
                flute_sentences.append(sentence)
                
                labels = ["sarcasm"]
                flute_classes.append(labels)
                          
        # store train set
        for i in range(len(flute_sentences)):
            multi_label_example = dict()
            multi_label_example["text"] = flute_sentences[i]
            
            # this dataset is binary so labels will contain "literal" or "sarcasm"
            multi_label_example["labels"] = flute_classes[i]
            
            if "FLUTE-sarcasm-train" not in multi_label_dataset:
                multi_label_dataset["FLUTE-sarcasm-train"] = []
                
            # append to multilabel example
            multi_label_dataset["FLUTE-sarcasm-train"].append(multi_label_example)
    
    # and reset
    flute_classes = []
    flute_sentences = []
    
    # load FLUTE-sarcasm test set
    with open(os.path.join(filepath, "sarcasm_test.jsonl"), "r") as fp:
        for line in fp: # loop through each line
            data = json.loads(line)
            
            # look at entailment pairs
            if data["label"] == "Entailment":
                # get sarcasmatic sentence
                sentence = detokenize_sentence(data["hypothesis"])
                flute_sentences.append(sentence)
                
                # in entailment, hypothesis are not sarcasm, but also not guarantee literal
                labels = ["not_sarcasm"]
                flute_classes.append(labels)
                      
                # now get the literal sentence
                sentence = detokenize_sentence(data["premise"])
                flute_sentences.append(sentence)
                
                labels = ["literal"]
                flute_classes.append(labels)
            
            # look at contradiction pairs
            elif data["label"] == "Contradiction":
                      
                # now get the other literal sentence
                sentence = detokenize_sentence(data["hypothesis"])
                flute_sentences.append(sentence)
                
                labels = ["sarcasm"]
                flute_classes.append(labels)
                
        # store test set
        for i in range(len(flute_sentences)):
            multi_label_example = dict()
            multi_label_example["text"] = flute_sentences[i]
            
            # this dataset is binary so labels will contain "literal" or "sarcasm"
            multi_label_example["labels"] = flute_classes[i]
            
            if "FLUTE-sarcasm-test" not in multi_label_dataset:
                multi_label_dataset["FLUTE-sarcasm-test"] = []
                
            # append to multilabel example
            multi_label_dataset["FLUTE-sarcasm-test"].append(multi_label_example)
            
    print("FLUTE-sarcasm dataset loaded. ")
    print("\t", len(multi_label_dataset["FLUTE-sarcasm-train"]), "training examples. ")
    print("\t", len(multi_label_dataset["FLUTE-sarcasm-test"]), "testing examples. ")
    
    
# method to load LCC dataset
def load_lcc(filepath, test_size = 0.1):
    lcc_classes = []
    lcc_sentences = []
    print("Loading LCC dataset...", end = " ")
    
    #lcc
    with open(os.path.join(filepath, "en_small.xml"), "r", encoding = "utf-8") as fp:
        # remove some nested tags that are not useful for our purposes
        xml_data = remove_nested_tags(fp.read())
        
    # parse the XML data into a python dictionary
    xml_dict = xmltodict.parse(xml_data)
    
    # get the list of LmInstance elements from the dictionary
    lm_instances = xml_dict["LCC-Metaphor-SMALL"]["LmInstance"]
    
    # loop through the list of LmInstance elements
    for lm_instance in lm_instances:
        
        # get the score attribute from the MetaphoricityAnnotation element
        if isinstance(lm_instance["Annotations"]["MetaphoricityAnnotations"]["MetaphoricityAnnotation"], list):
            scores = []
            for m4_annotation in lm_instance["Annotations"]["MetaphoricityAnnotations"]["MetaphoricityAnnotation"]:
                scores.append(float(m4_annotation["@score"]))
            
            # get mean annotation score
            score = int(sum(scores)/len(scores))
            
        else: 
            score = int(float(lm_instance["Annotations"]["MetaphoricityAnnotations"]["MetaphoricityAnnotation"]["@score"]))
        
        # convert scores to labels
        if score == 0:
            labels = ["not_metaphor"]
            lcc_classes.append(labels)
            
        elif score == 2:
            labels = ["metaphor"]
            lcc_classes.append(labels)
        
        elif score == 3:
            labels = ["metaphor"]
            lcc_classes.append(labels)
            
        else: # skip scores of 1.0 because they are weak, potential not metaphors
            continue
        
        # get the Current attribute from the TextContent element
        sentence = prepcocess_sentence(lm_instance["TextContent"]["Current"])
        lcc_sentences.append(sentence)
        
    # LCC does not have a train/test split, so lets get a 10% split to evaluate later
    lcc_labels = [label for labels in lcc_classes for label in labels]    
    sss = StratifiedShuffleSplit(n_splits = 1, random_state = 42, test_size = test_size)
    train_index, test_index = next(sss.split(lcc_sentences, lcc_labels))
    
    # store train set
    for i in train_index:        
        multi_label_example = dict()
        multi_label_example["text"] = lcc_sentences[i]
        
        # this dataset is binary so labels will contain "literal" or "metaphor"
        multi_label_example["labels"] = lcc_classes[i]
        
        if "LCC-train" not in multi_label_dataset:
            multi_label_dataset["LCC-train"] = []
            
        # append to multilabel example
        multi_label_dataset["LCC-train"].append(multi_label_example)
    
    # store test set
    for i in test_index:
        multi_label_example = dict()
        multi_label_example["text"] = lcc_sentences[i]
        
        # this dataset is binary so labels will contain "literal" or "metaphor"
        multi_label_example["labels"] = lcc_classes[i]
        
        if "LCC-test" not in multi_label_dataset:
            multi_label_dataset["LCC-test"] = []
            
        # append to multilabel example
        multi_label_dataset["LCC-test"].append(multi_label_example)
    
    print("LCC dataset loaded. ")
    print("\t", len(multi_label_dataset["LCC-train"]), "training examples. ")
    print("\t", len(multi_label_dataset["LCC-test"]), "testing examples. ")
    
    
# method to load MOH dataset
def load_moh(filepath, test_size = 0.1):
    moh_classes = []
    moh_sentences = []
    print("Loading MOH dataset...", end = " ")
    
    # moh
    with open(os.path.join(filepath, "Data-metaphoric-or-literal.txt"), "r") as fp:
        for line in fp:
            parts = line.strip().split("\t")
            
            # skip title line
            if "term" in parts[0]:
                continue
            
            sentence = prepcocess_sentence(parts[2])
            moh_sentences.append(sentence)
            
            if "metaphorical" in parts[3]:
                labels = ["metaphor"]
                moh_classes.append(labels)
                
            else: # convert them to a easy to understand string
                labels = ["literal"]
                moh_classes.append(labels)
            
    # MOH does not have a train/test split, so lets get a 10% split to evaluate later
    moh_labels = [label for labels in moh_classes for label in labels]    
    sss = StratifiedShuffleSplit(n_splits = 1, random_state = 42, test_size = test_size)
    train_index, test_index = next(sss.split(moh_sentences, moh_labels))
    
    # store train set
    for i in train_index:        
        multi_label_example = dict()
        multi_label_example["text"] = moh_sentences[i]
        
        # this dataset is binary so labels will contain "literal" or "metaphor"
        multi_label_example["labels"] = moh_classes[i]
        
        if "MOH-train" not in multi_label_dataset:
            multi_label_dataset["MOH-train"] = []
            
        # append to multilabel example
        multi_label_dataset["MOH-train"].append(multi_label_example)
    
    # store test set
    for i in test_index:
        multi_label_example = dict()
        multi_label_example["text"] = moh_sentences[i]
        
        # this dataset is binary so labels will contain "literal" or "metaphor"
        multi_label_example["labels"] = moh_classes[i]
        
        if "MOH-test" not in multi_label_dataset:
            multi_label_dataset["MOH-test"] = []
            
        # append to multilabel example
        multi_label_dataset["MOH-test"].append(multi_label_example)
    
    print("MOH dataset loaded. ")
    print("\t", len(multi_label_dataset["MOH-train"]), "training examples. ")
    print("\t", len(multi_label_dataset["MOH-test"]), "testing examples. ")
    
# method to load EPIE dataset
def load_epie(filepath, test_size = 0.1):
    epie_classes = []
    epie_sentences = []
    print("Loading EPIE dataset...", end = " ")

    # load list of sentences first
    with open(os.path.join(filepath, "Formal_Idioms_Words.txt"), "r") as fp:
        for line in fp:

            # lines have added spaces due to tokenization
            sentence = detokenize_sentence(line)
            epie_sentences.append(sentence)
            
    # load idiom labels
    with open(os.path.join(filepath, "Formal_Idioms_Labels.txt"), "r") as fp:
        for line in fp:
           
            # labels are 0 or 1
            if "1" in line:
                labels = ["idiom"]
                
            else: # convert them to a easy to understand string
                labels = ["literal"]
            
            # collect labels
            epie_classes.append(labels)
            
    # EPIE does not have a train/test split, so lets get a 10% split to evaluate later
    epie_labels = [label for labels in epie_classes for label in labels]    
    sss = StratifiedShuffleSplit(n_splits = 1, random_state = 42, test_size = test_size)
    train_index, test_index = next(sss.split(epie_sentences, epie_labels))
    
    # store train set
    for i in train_index:        
        multi_label_example = dict()
        multi_label_example["text"] = epie_sentences[i]
        
        # this dataset is binary so labels will contain "literal" or "idiom"
        multi_label_example["labels"] = epie_classes[i]
        
        if "EPIE-train" not in multi_label_dataset:
            multi_label_dataset["EPIE-train"] = []
            
        # append to multilabel example
        multi_label_dataset["EPIE-train"].append(multi_label_example)
    
    # store test set
    for i in test_index:
        multi_label_example = dict()
        multi_label_example["text"] = epie_sentences[i]
        
        # this dataset is binary so labels will contain "literal" or "idiom"
        multi_label_example["labels"] = epie_classes[i]
        
        if "EPIE-test" not in multi_label_dataset:
            multi_label_dataset["EPIE-test"] = []
            
        # append to multilabel example
        multi_label_dataset["EPIE-test"].append(multi_label_example)
    
    print("EPIE dataset loaded. ")
    print("\t", len(multi_label_dataset["EPIE-train"]), "training examples. ")
    print("\t", len(multi_label_dataset["EPIE-test"]), "testing examples. ")
    
# special classd to load PIE-EN
# this code was taken from the original authors from their github repository:
#    https://github.com/tosingithub/idesk/blob/master/loaddata.py
class MakeSentence(object):
    """
    Makes sentences of the data of tokens passed to it
    """
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(t, c) for t, c in zip(s["token"].values.tolist(),                                                           
                                                     s["class"].values.tolist())]
        
        self.grouped = self.data.groupby("id").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        
        except ValueError:
            return None
        
# method to load PIE-EN dataset, using special class above
def load_pie_en(filepath, test_size = 0.1):
    piee_classes = []
    piee_sentences = []
    
    # classes of pie-en to ignore
    piee_ignore = set(["euphemism", "personification", "paradox", "oxymoron", "parallelism"])
    print("Loading PIE-EN dataset...", end = " ")

    # load pie-en
    filename = os.path.join(filepath, "idiomscorpus.csv")
    data = pd.read_csv(filename, encoding = "latin1").ffill()
    get_sentences = MakeSentence(data) # create the sentence maker
    
    # get sentence tokens and construct true labels for each sentence
    piee_tokens = [[s[0] for s in sentence] for sentence in get_sentences.sentences]
    piee_classes_m = [[s[1] for s in sentence] for sentence in get_sentences.sentences]
    
    for i in range(len(piee_tokens)):
        
        # sentences with less than 6 tokens are idiom types
        if len(piee_tokens[i]) > 6: 
            
            # get label and detokenize sentences
            piee_class = piee_classes_m[i][0].lower()
            sentence = TreebankWordDetokenizer().detokenize(piee_tokens[i])
            
            # then clean up sentence
            sentence = prepcocess_sentence(sentence)
            
            if piee_class == "literal":
                labels = [piee_class]
                piee_classes.append(labels)
                piee_sentences.append(sentence)
            
            # anything non-literal is an class + "idiom"
            elif piee_class not in piee_ignore: 
                labels = [piee_class, "idiom"]
                piee_classes.append(labels)
                piee_sentences.append(sentence)
            
    # PIE-EN does not have a train/test split, so lets get a 10% split to evaluate later
    # stratiufication based on labels because in this dataset, 
    # anything non-literal is an idiom
    piee_labels = [labels[0] for labels in piee_classes]    
    sss = StratifiedShuffleSplit(n_splits = 1, random_state = 42, test_size = test_size)
    train_index, test_index = next(sss.split(piee_sentences, piee_labels))
    
    # store train set
    for i in train_index:        
        multi_label_example = dict()
        multi_label_example["text"] = piee_sentences[i]
        
        # this dataset is multi-label, so labels will contain "literal" or "idiom" + other class 
        multi_label_example["labels"] = piee_classes[i]
        
        if "PIE-EN-train" not in multi_label_dataset:
            multi_label_dataset["PIE-EN-train"] = []
            
        # append to multilabel example
        multi_label_dataset["PIE-EN-train"].append(multi_label_example)
    
    # store test set
    for i in test_index:
        multi_label_example = dict()
        multi_label_example["text"] = piee_sentences[i]
        
        # this dataset is multi-label, so labels will contain "literal" or "idiom" + other class 
        multi_label_example["labels"] = piee_classes[i]
        
        if "PIE-EN-test" not in multi_label_dataset:
            multi_label_dataset["PIE-EN-test"] = []
            
        # append to multilabel example
        multi_label_dataset["PIE-EN-test"].append(multi_label_example)  
    
    print("PIE-EN dataset loaded. ")
    print("\t", len(multi_label_dataset["PIE-EN-train"]), "training examples. ")
    print("\t", len(multi_label_dataset["PIE-EN-test"]), "testing examples. ")
    
# method to load Scarsam Corpus v2 Dataset
def load_sarcasm_corpus(filepath, test_size = 0.1):
    sarc_classes = []
    sarc_sentences = []   
    print("Loading Sarcasm Corpus v2 dataset...", end = " ")
    
    for file in os.listdir(filepath):
        filename = os.path.join(filepath, file)
        with open(filename, 'r', encoding = 'utf-8') as csv_file:
            reader = csv.reader(csv_file)
            
            for row in reader: # loop through csv lines
                if "label" in row[0]: # skip title line
                    continue
                
                # get label information
                if row[0] == "sarc":
                    if "HYP-" in file:
                        # this dataset is multi-label, so labels will contain "sarcasm" or "sarcasm" + "hyperbole"
                        labels = ["hyperbole", "sarcasm"]
                        
                    else:
                        labels = ["sarcasm"]
                    
                elif row[0] == "notsarc": # convert them to a easy to understand string
                    if "HYP-" in file:
                        labels = ["hyperbole", "not_sarcasm"]
                        
                    else: # non-sarcastic labels are hyperbole or literal
                        labels = ["not_sarcasm"]
                    
                else: 
                    continue
                
                # then clean up the sentence
                sentence = prepcocess_sentence(row[2])
                sarc_sentences.append(sentence)
                sarc_classes.append(labels)
            
    # Sarc Corpus v2 does not have a train/test split
    # so lets get a 10% split to evaluate later
    # use stratification to get good split between hyperbole, sarcasm and "literal"
    sarc_labels = [labels[0] for labels in sarc_classes]
    sss = StratifiedShuffleSplit(n_splits = 1, random_state = 42, test_size = test_size)
    train_index, test_index = next(sss.split(sarc_sentences, sarc_labels))
    
    # store train set
    for i in train_index:        
        multi_label_example = dict()
        multi_label_example["text"] = sarc_sentences[i]
        
        # this dataset is multi-label, so labels will contain hyperbole, sarcasm and "literal"
        multi_label_example["labels"] = sarc_classes[i]
        
        if "SARCv2-train" not in multi_label_dataset:
            multi_label_dataset["SARCv2-train"] = []
            
        # append to multilabel example
        multi_label_dataset["SARCv2-train"].append(multi_label_example)
    
    # store test set
    for i in test_index:
        multi_label_example = dict()
        multi_label_example["text"] = sarc_sentences[i]
        
        # this dataset is multi-label, so labels will contain hyperbole, sarcasm and "literal" 
        multi_label_example["labels"] = sarc_classes[i]
        
        if "SARCv2-test" not in multi_label_dataset:
            multi_label_dataset["SARCv2-test"] = []
            
        # append to multilabel example
        multi_label_dataset["SARCv2-test"].append(multi_label_example)  
    
    print("Sarcasm Corpus v2 dataset loaded. ")
    print("\t", len(multi_label_dataset["SARCv2-train"]), "training examples. ")
    print("\t", len(multi_label_dataset["SARCv2-test"]), "testing examples. ")

# method to load Ironic Corpus Dataset
def load_ironic_corpus(filepath, test_size = 0.1):
    iron_classes = []
    iron_sentences = []   
    print("Loading Ironic Corpus dataset...", end = " ")
    
    with open(os.path.join(filepath, "irony-labeled.csv"), 'r', encoding = 'utf-8') as csv_file:
        reader = csv.reader(csv_file)
        
        for row in reader: # loop through csv lines
            if "comment_text" in row[0]: # skip title line
                continue
            
            # get label information
            if row[1] == "1":
                labels = ["irony"]
                
            elif row[1] == "-1": # convert them to a easy to understand string
                labels = ["not_irony"]
            
            # then clean up the sentence
            sentence = prepcocess_sentence(row[0])
            iron_sentences.append(sentence)
            iron_classes.append(labels)
            
    # Sarc Corpus v2 does not have a train/test split
    # so lets get a 10% split to evaluate later
    # use stratification to get good split between irony and "literal"
    iron_labels = [label for labels in iron_classes for label in labels] 
    sss = StratifiedShuffleSplit(n_splits = 1, random_state = 42, test_size = test_size)
    train_index, test_index = next(sss.split(iron_sentences, iron_labels))
    
    # store train set
    for i in train_index:        
        multi_label_example = dict()
        multi_label_example["text"] = iron_sentences[i]
        
        # this dataset is binary, so labels will contain "literal" or "irony"
        multi_label_example["labels"] = iron_classes[i]
        
        if "IRONY-train" not in multi_label_dataset:
            multi_label_dataset["REDDIT-IRONY-train"] = []
            
        # append to multilabel example
        multi_label_dataset["REDDIT-IRONY-train"].append(multi_label_example)
    
    # store test set
    for i in test_index:
        multi_label_example = dict()
        multi_label_example["text"] = iron_sentences[i]
        
        # this dataset is binary, so labels will contain "literal" or "irony"
        multi_label_example["labels"] = iron_classes[i]
        
        if "IRONY-test" not in multi_label_dataset:
            multi_label_dataset["REDDIT-IRONY-test"] = []
            
        # append to multilabel example
        multi_label_dataset["REDDIT-IRONY-test"].append(multi_label_example)  
    
    print("Ironic Corpus dataset loaded. ")
    print("\t", len(multi_label_dataset["REDDIT-IRONY-train"]), "training examples. ")
    print("\t", len(multi_label_dataset["REDDIT-IRONY-test"]), "testing examples. ")
    
# method to load HYPOGen dataset
def load_hypo_gen(filepath, test_size = 0.1):
    hyp_classes = []
    hyp_sentences = []   
    print("Loading HYPOGen dataset...", end = " ")
    
    with open(os.path.join(filepath, "hypo_red.txt"), 'r', encoding = 'utf-8') as fp:
        for line in fp:
            row = line.strip().split("\t")
            if "Text" in row[0]: # skip title line
                continue
            
            # get label information
            if row[1] == "0" and row[2] == "1":
                labels = ["hyperbole"]
            
            # convert them to a easy to understand string
            elif row[1] == "1" and row[2] == "0": 
                labels = ["literal"]
                
            else:
                continue
            
            # then clean up the sentence
            sentence = prepcocess_sentence(row[0])
            hyp_sentences.append(sentence)
            hyp_classes.append(labels)
            
    # HYPOGen does not have a train/test split
    # so lets get a 10% split to evaluate later
    hyp_labels = [label for labels in hyp_classes for label in labels]
    sss = StratifiedShuffleSplit(n_splits = 1, random_state = 42, test_size = test_size)
    train_index, test_index = next(sss.split(hyp_sentences, hyp_labels))
    
    # store train set
    for i in train_index:        
        multi_label_example = dict()
        multi_label_example["text"] = hyp_sentences[i]
        
        # this dataset is binary, so labels will contain "literal" or "hyperbole"
        multi_label_example["labels"] = hyp_classes[i]
        
        if "HYPOGen-train" not in multi_label_dataset:
            multi_label_dataset["HYPOGen-train"] = []
            
        # append to multilabel example
        multi_label_dataset["HYPOGen-train"].append(multi_label_example)
    
    # store test set
    for i in test_index:
        multi_label_example = dict()
        multi_label_example["text"] = hyp_sentences[i]
        
        # this dataset is binary, so labels will contain "literal" or "hyperbole"
        multi_label_example["labels"] = hyp_classes[i]
        
        if "HYPOGen-test" not in multi_label_dataset:
            multi_label_dataset["HYPOGen-test"] = []
            
        # append to multilabel example
        multi_label_dataset["HYPOGen-test"].append(multi_label_example)  
    
    print("HYPOGen dataset loaded. ")
    print("\t", len(multi_label_dataset["HYPOGen-train"]), "training examples. ")
    print("\t", len(multi_label_dataset["HYPOGen-test"]), "testing examples. ")
    
# method to load MOVER dataset
def load_mover(filepath, test_size = 0.1):
    hyp_classes = []
    hyp_sentences = []   
    print("Loading MOVER dataset...", end = " ")
    
    with open(os.path.join(filepath, "HYPO-L.csv"), 'r', encoding = 'utf-8') as csv_file:
        reader = csv.reader(csv_file)

        for row in reader:
            
            # skip title line
            if "Label" in row[1]:
                continue
            
            # get label information
            if row[1] == "1":
                labels = ["hyperbole"]
            
            else:
                labels = ["literal"]
            
            # then clean up the sentence
            sentence = prepcocess_sentence(row[0])
            hyp_sentences.append(sentence)
            hyp_classes.append(labels)
            
    # MOVER does not have a train/test split
    # so lets get a 10% split to evaluate later
    hyp_labels = [label for labels in hyp_classes for label in labels]
    sss = StratifiedShuffleSplit(n_splits = 1, random_state = 42, test_size = test_size)
    train_index, test_index = next(sss.split(hyp_sentences, hyp_labels))
    
    # store train set
    for i in train_index:        
        multi_label_example = dict()
        multi_label_example["text"] = hyp_sentences[i]
        
        # this dataset is binary, so labels will contain "literal" or "hyperbole"
        multi_label_example["labels"] = hyp_classes[i]
        
        if "MOVER-train" not in multi_label_dataset:
            multi_label_dataset["MOVER-train"] = []
            
        # append to multilabel example
        multi_label_dataset["MOVER-train"].append(multi_label_example)
    
    # store test set
    for i in test_index:
        multi_label_example = dict()
        multi_label_example["text"] = hyp_sentences[i]
        
        # this dataset is binary, so labels will contain "literal" or "hyperbole"
        multi_label_example["labels"] = hyp_classes[i]
        
        if "MOVER-test" not in multi_label_dataset:
            multi_label_dataset["MOVER-test"] = []
            
        # append to multilabel example
        multi_label_dataset["MOVER-test"].append(multi_label_example)  
    
    print("MOVER dataset loaded. ")
    print("\t", len(multi_label_dataset["MOVER-train"]), "training examples. ")
    print("\t", len(multi_label_dataset["MOVER-test"]), "testing examples. ")

# method to load MSD23 dataset
def load_msd23(filepath, test_size = 0.1):
    sim_classes = []
    sim_sentences = []   
    print("Loading MSD23 dataset...", end = " ")
    
    with open(os.path.join(filepath, "literals.json"), 'r', encoding = 'utf-8') as fp:
        literals_json = json.load(fp)

        for literal in literals_json:
            # clean up the sentence and store label info
            sentence = prepcocess_sentence(literal["literal"])
            sim_classes.append(["literal"])
            sim_sentences.append(sentence)
            
        with open(os.path.join(filepath, "similes.json"), 'r', encoding = 'utf-8') as fp:
            similes_json = json.load(fp)

            for similes in similes_json:
                # clean up the sentence and store label info
                sentence = prepcocess_sentence(similes["simile"])
                sim_classes.append(["simile"])
                sim_sentences.append(sentence)
            
    # MSD23 does not have a train/test split
    # so lets get a 10% split to evaluate later
    sim_labels = [label for labels in sim_classes for label in labels]
    sss = StratifiedShuffleSplit(n_splits = 1, random_state = 42, test_size = test_size)
    train_index, test_index = next(sss.split(sim_sentences, sim_labels))
    
    # store train set
    for i in train_index:        
        multi_label_example = dict()
        multi_label_example["text"] = sim_sentences[i]
        
        # this dataset is binary, so labels will contain "literal" or "simile"
        multi_label_example["labels"] = sim_classes[i]
        
        if "MSD23-train" not in multi_label_dataset:
            multi_label_dataset["MSD23-train"] = []
            
        # append to multilabel example
        multi_label_dataset["MSD23-train"].append(multi_label_example)
    
    # store test set
    for i in test_index:
        multi_label_example = dict()
        multi_label_example["text"] = sim_sentences[i]
        
        # this dataset is binary, so labels will contain "literal" or "simile"
        multi_label_example["labels"] = sim_classes[i]
        
        if "MSD23-test" not in multi_label_dataset:
            multi_label_dataset["MSD23-test"] = []
            
        # append to multilabel example
        multi_label_dataset["MSD23-test"].append(multi_label_example)  
    
    print("MSD23 dataset loaded. ")
    print("\t", len(multi_label_dataset["MSD23-train"]), "training examples. ")
    print("\t", len(multi_label_dataset["MSD23-test"]), "testing examples. ")
    
# method to load Figurative Comparisons dataset
def load_fig_comp(filepath, test_size = 0.1):
    sim_classes = []
    sim_sentences = []   
    print("Loading Figurative Comparisons dataset...", end = " ")
    
    with open(os.path.join(filepath, "literals.json"), 'r', encoding = 'utf-8') as fp:
        literals_json = json.load(fp)

        for literal in literals_json:
            # clean up the sentence and store label info
            sentence = prepcocess_sentence(literal["text"])
            sim_classes.append(["literal"])
            sim_sentences.append(sentence)
            
        with open(os.path.join(filepath, "similes.json"), 'r', encoding = 'utf-8') as fp:
            similes_json = json.load(fp)

            for similes in similes_json:
                # clean up the sentence and store label info
                sentence = prepcocess_sentence(similes["text"])
                sim_classes.append(["simile"])
                sim_sentences.append(sentence)
            
    # Figurative Comparisons does not have a train/test split
    # so lets get a 10% split to evaluate later
    sim_labels = [label for labels in sim_classes for label in labels]
    sss = StratifiedShuffleSplit(n_splits = 1, random_state = 42, test_size = test_size)
    train_index, test_index = next(sss.split(sim_sentences, sim_labels))
    
    # store train set
    for i in train_index:        
        multi_label_example = dict()
        multi_label_example["text"] = sim_sentences[i]
        
        # this dataset is binary, so labels will contain "literal" or "simile"
        multi_label_example["labels"] = sim_classes[i]
        
        if "FigComp-train" not in multi_label_dataset:
            multi_label_dataset["FigComp-train"] = []
            
        # append to multilabel example
        multi_label_dataset["FigComp-train"].append(multi_label_example)
    
    # store test set
    for i in test_index:
        multi_label_example = dict()
        multi_label_example["text"] = sim_sentences[i]
        
        # this dataset is binary, so labels will contain "literal" or "simile"
        multi_label_example["labels"] = sim_classes[i]
        
        if "FigComp-test" not in multi_label_dataset:
            multi_label_dataset["FigComp-test"] = []
            
        # append to multilabel example
        multi_label_dataset["FigComp-test"].append(multi_label_example)  
    
    print("Figurative Comparisons dataset loaded. ")
    print("\t", len(multi_label_dataset["FigComp-train"]), "training examples. ")
    print("\t", len(multi_label_dataset["FigComp-test"]), "testing examples. ")

### SCRIPT ###
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# check if file already exists
if os.path.isfile(base_dir + "/data/fig-lang-dataset.json"):
    multi_label_dataset = load_json(base_dir + "/data/fig-lang-dataset.json")
    dataset_statistics(multi_label_dataset, "all-type")
    
else:
    # # Datasets that have their own test set
    load_flute_m4(base_dir + "/data/flute/") # actual lietarls
    load_isarcasm(base_dir + "/data/iSarcasm/") # not actual literals
    load_flute_idiom(base_dir + "/data/flute/") # actual lietarls
    load_flute_simile(base_dir + "/data/flute/") # actual lietarls
    load_flute_sarcasm(base_dir + "/data/flute/") # actual lietarls
    load_irony_eval(base_dir + "/data/Irony SemEval18/")  # not actual literals
    
    # # Datasets that do not have a test set, we take 10% as test
    load_lcc(base_dir + "/data/LCC/") # not actual literals
    load_moh(base_dir + "/data/MOH/") # actual lietarls
    load_epie(base_dir + "/data/epie/") # actual lietarls
    load_mover(base_dir + "/data/MOVER/") # actual lietarls
    load_pie_en(base_dir + "/data/pie-en/") # actual lietarls
    load_msd23(base_dir + "/data/MSD23-v1.0/") # actual lietarls
    load_hypo_gen(base_dir + "/data/HYPOGen/") # actual lietarls
    load_ironic_corpus(base_dir + "/data/Reddit Irony Corpus/") # not actual literals
    load_sarcasm_corpus(base_dir + "/data/Sarcasm Corpus V2/") # not actual literals
    load_fig_comp(base_dir + "/data/figurative-comparisons-data/") # actual lietarls
    
    # now save this data as json file
    save_json(base_dir + "/data/fig-lang-dataset.json", multi_label_dataset)

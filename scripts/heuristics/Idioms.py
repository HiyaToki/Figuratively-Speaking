# -*- coding: utf-8 -*-

import re
import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# method to load a json file to memory
def load_json(filename):

    # open file and load
    with open(filename, 'r') as fp:
        data = json.load(fp)

    return data

# method to preprocess the text
def preprocess_text(text):

    # just remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text

# method to predict style class of a single sentence
def predict(text):

    text = preprocess_text(text)

    # encode the text
    inputs = tokenizer(text,
                       padding = True,
                       max_length = 512,
                       truncation = True,
                       return_tensors = 'pt').to("cuda")

    with torch.no_grad():

        # pass encodings to the model
        logits = model(**inputs).logits

        # compute probability as softmax score
        probability = F.softmax(logits, dim = -1).tolist()

        # get class based on maximum probability
        predicted_class_index = np.argmax(probability)
        prediction = id2label[str(predicted_class_index)]

    return prediction

# method to predict Simile labels
def predict_loop(texts):

    results = []
    for text in texts:
        result = predict(text)
        results.append(result)

    return results

# Read data, get the absolute path of the current script
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/roberta-large-idiom"))
label2id = load_json(model_dir + "/label2id.json")
id2label = load_json(model_dir + "/id2label.json")

# HF model for seq classification
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir).to("cuda")
model.eval() # lock model in eval mode

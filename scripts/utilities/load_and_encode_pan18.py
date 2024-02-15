# -*- coding: utf-8 -*-

import os
import sys
import json
import torch
import warnings
import numpy as np
from tqdm import tqdm
from transformers import logging
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# get the absolute path of the current script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../scripts/heuristics/")))
import Metaphors, Sarcasm, Hyperbole, Similes, Irony, Idioms, Stylometrics

# ignore all warnings
logging.set_verbosity_error()
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# method to save json file
def save_json(data, filename):
    # open file and load
    with open(filename, 'w') as fp:
        json.dump(data, fp)

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

# method to compute a document embedding from sentence embeddings
def compute_document_embedding(sentence_results, key = "embedding", pooling_method = "mean"):
    embeddings = []

    for sentence_result in sentence_results:
        embeddings.append(sentence_result[key])

    # do mean pool to get document embedding
    if pooling_method == "mean":
        document_embedding = np.mean(embeddings, axis = 0).tolist()

    # alternatively we could try max pooling
    elif pooling_method == "max":
        document_embedding = np.amax(embeddings, axis = 0).tolist()

    return document_embedding

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

# method for mean pooling that takes attention mask into account for correct averaging
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min = 1e-9)
    
    return sum_embeddings / sum_mask

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
        model_output = model(**inputs, output_hidden_states = True)
        
        # get logits (output layer)
        logits = model_output.logits
        
        # get sentence embedding using the model's last hidden layer (layer just before the output layer)
        output_tokens = model_output.hidden_states[model.config.num_hidden_layers - 1]
        embedding = mean_pooling(output_tokens, inputs["attention_mask"])        
        embedding = embedding.detach()
        embedding = embedding.cpu()
        
        # convert to python list for json serialization
        embedding = np.array(embedding[0]).tolist()

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
                       "embedding": embedding,
                       }

    return predictions

# method to parse ground thruth for pan18
def get_ground_truth(problem_path):
    raw_ground_truth = load_json(os.path.abspath(os.path.join(problem_path, "ground-truth.json")))
    ground_truth = dict()
    
    for raw_element in raw_ground_truth["ground_truth"]:
        txt_id = raw_element["unknown-text"]
        author = raw_element["true-author"]
        
        ground_truth[txt_id] = author
        
    return ground_truth

# method to basic preprocess text
def process(text):
    text = TreebankWordDetokenizer().detokenize(text.split())
    return text

# method to load the pan18 dataset
def load_pan18(filepath):
    dataset = dict()

    for problem in ["problem00001", "problem00002", "problem00003", "problem00004"]:
        problem_path = os.path.join(filepath, problem)
        ground_truth = get_ground_truth(problem_path)
        dataset[problem] = []
        
        for candidate in os.listdir(problem_path):
            candidate_path = os.path.join(problem_path, candidate)
            
            # skip info files
            if candidate in ["ground-truth.json", "problem-info.json"]:
                continue
            
            for file in os.listdir(candidate_path):
                file_name = os.path.join(candidate_path, file)
        
                # open file to read lines
                with open(file_name, "r", encoding = "latin-1") as f:
                    text = process(f.read())
                    author = candidate
                    is_test = False
                    txt_id = file
                    
                    # retrieve author for test files
                    if candidate == "unknown":
                        author = ground_truth[txt_id]
                        is_test = True
                        
                    # save relative info 
                    data_item = dict()
                    data_item["text"] = text
                    data_item["author"] = author
                    data_item["text_id"] = txt_id
                    data_item["is_test"] = is_test
                    dataset[problem].append(data_item)
                    
    return dataset
            
# method to get all prediction results for one sentence at a time
def sentence_predictions(sentences, sbert_model, joint_model, joint_tokenizer, label2id, id2label, thresholds):
    sentence_results = []
    
    # loop through sentences
    for sentence in sentences:
        
        # use prediction methods to get results
        joint_prediction = joint_predict(sentence, joint_model, joint_tokenizer, id2label, thresholds)
        sbert_embedding = sbert_model.encode(sentence, convert_to_numpy = True)
        binary_prediction = binary_ensemble_predict(sentence, label2id)
        
        sentnece_result = dict()
        sentnece_result["sentence"] = sentence
        
        #collect sbert embeddings
        sentnece_result["sbert_model_embedding"] = sbert_embedding
        
        # collect results from joint model
        sentnece_result["joint_model_embedding"] = joint_prediction["embedding"]
        sentnece_result["joint_model_predicted_labels"] = joint_prediction["predicted_labels"]
        sentnece_result["joint_model_one_hot_prediction"] = joint_prediction["one_hot_prediction"]
        
        # collect results from binary model
        sentnece_result["binary_model_predicted_labels"] = binary_prediction["predicted_labels"]
        sentnece_result["binary_model_one_hot_prediction"] = binary_prediction["one_hot_prediction"]
        
        sentence_results.append(sentnece_result)

    return sentence_results
    
# method to split a document into sentences and obtain embeddings and RSR classification result
def document_predictions(dataset, sbert_model, joint_model, joint_tokenizer, label2id, id2label, thresholds):
    
    for problem in dataset.keys():
        for i in tqdm(range(len(dataset[problem])), desc = "PAN18 " + problem.upper() + " PROGRESS", ncols = 100):
            text = dataset[problem][i]["text"]
            # use nltk to tokenize a document's text into sentences
            sentences = sent_tokenize(text)
            
            # the run prediction for each sentence
            sentence_results = sentence_predictions(sentences, 
                                                    sbert_model, 
                                                    joint_model, 
                                                    joint_tokenizer, 
                                                    label2id, 
                                                    id2label, 
                                                    thresholds)
            
            # Stylometrics are computed document-wise
            dataset[problem][i]["stylometric_features"] = Stylometrics.extract(text) 
            dataset[problem][i]["sbert_model_embedding"] = compute_document_embedding(sentence_results, "sbert_model_embedding")
            dataset[problem][i]["joint_model_embedding"] = compute_document_embedding(sentence_results, "joint_model_embedding")
            dataset[problem][i]["joint_model_one_hot_prediction"] = compute_document_embedding(sentence_results, "joint_model_one_hot_prediction")
            dataset[problem][i]["binary_model_one_hot_prediction"] = compute_document_embedding(sentence_results, "binary_model_one_hot_prediction")

    return dataset

# method to get features and labels from dataset
def get_documents(dataset):
    doc_train = [] 
    i_train = []
    doc_test = []
    i_test = []

    # get features and authors as labels
    for i in range(len(dataset)):
        x = dataset[i]["text"]
            
        # append features and class
        if dataset[i]["is_test"]:
            doc_test.append(x)
            i_test.append(i)
            
        else:
            doc_train.append(x)
            i_train.append(i)
            
    return doc_train, i_train, doc_test, i_test

# method to add n-gram features to a dataset
def add_tfidf_features(dataset, n_gram_type = "word"):
    print("EXTRACTING " + n_gram_type.upper() + " N-GRAMS...", end = " ")
    doc_train, i_train, doc_test, i_test = get_documents(dataset)

    # create appropriate key
    key = n_gram_type + "_n_gram_tfidf_features"
    
    # use stopwords?
    if n_gram_type == "word":
        english_stop_words = "english"
    
    else: # when its "char"
        english_stop_words = None
        
    # create the TfidfVectorizer instance with word and character n-grams
    vectorizer = TfidfVectorizer(analyzer = n_gram_type, 
                                 stop_words = english_stop_words,
                                 ngram_range = (1, 5),
                                 max_features = 1024,
                                 max_df = 0.99,
                                 min_df = 2,
                                 )

    # learn vocabulary on train set
    # and then generate the TF-IDF vectors for the documents
    train_tfidf_vectors = vectorizer.fit_transform(doc_train)
    train_dense_vectors = train_tfidf_vectors.todense()
    
    # generate tf-idf vectors for test set
    test_tfidf_vectors = vectorizer.transform(doc_test)
    test_dense_vectors = test_tfidf_vectors.todense()

    # save dense vectors in the dataset
    for i in range(len(train_dense_vectors)):
        dataset[i_train[i]][key] = train_dense_vectors[i].tolist()[0]
    
    # using the proper index
    for i in range(len(test_dense_vectors)):
        dataset[i_test[i]][key] = test_dense_vectors[i].tolist()[0]
    
    print("DONE!")
    return dataset

#### SCRIPT ####

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
dataset = load_pan18(base_dir + "/data/pan18/pan18-cross-domain-authorship-attribution-test-dataset")

# sbert model
sbert_model_name = "all-roberta-large-v1"

print("LOADING SBERT MODEL: ", sbert_model_name, end = "...")
sbert_model = SentenceTransformer(sbert_model_name)
print(" DONE.")

# load maps, thresholds, tokenizer, and model
joint_model_name = "roberta-large-joint-fig-lang"
joint_model_file_path = base_dir + "/models/" + joint_model_name

print("LOADING JOINT MODEL: ", joint_model_name, end = "...")
joint_label2id, joint_id2label = get_label_map(joint_model_file_path)
joint_tokenizer = AutoTokenizer.from_pretrained(joint_model_file_path)
joint_thresholds = load_json(os.path.join(joint_model_file_path, "thresholds_bin.json"))
joint_model = AutoModelForSequenceClassification.from_pretrained(joint_model_file_path).to("cuda")
joint_model.eval() # lock model in eval mode
print(" DONE.")

dataset = document_predictions(dataset, 
                               sbert_model, 
                               joint_model, 
                               joint_tokenizer, 
                               joint_label2id, 
                               joint_id2label, 
                               joint_thresholds)

# add word and char n-gram features
for problem in dataset.keys():
    # add word n-grams
    dataset[problem] = add_tfidf_features(dataset[problem])
    
    # add char n-grams
    dataset[problem] = add_tfidf_features(dataset[problem], "char")

save_json(dataset, base_dir + "/data/pan18/pan18-authorship-attribution-test-dataset.json")
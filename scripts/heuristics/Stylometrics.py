
import os
import math
import json
import collections
import pandas as pd
from nltk.corpus import stopwords
from cophi.text import complexity
from textstat.textstat import textstat
from nltk.tokenize import word_tokenize, sent_tokenize

# fetch stopwords from nltk
stop = stopwords.words('english')

# load word frequency class mapping, get the absolute path of the current script
with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../scripts/heuristics/external_data/word_class_mapping.txt")), "r") as f:
    word_class_mapping = json.load(f)

# load function words lexicon, get the absolute path of the current script
with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../scripts/heuristics/external_data/functionwords.txt")), "r") as f:
    functionwords = f.read().splitlines()

feature_names = ["Average Word Length",
                 "Average Syllables Per Word",
                 "Average Sentence Length",
                 "Average Sentence Length Chars",
                 "Average Word Frequency Class",
                 "Type Token Ratio",
                 "Special Characters Ratio",
                 "Puncuations Ratio",
                 "Uppercase Ratio",
                 "Digit Ratio",
                 "Stopword Ratio",
                 "Functional Words Ratio",
                 "Hapax Legomena Ratio",
                 "Hapax Dislegomena Ratio",
                 "Automated Readability Metric",
                 "Flesch Reading Ease Metric",
                 "Flesch Kincaid Grade Metric",
                 "Dale Chall Readability Metric",
                 "New Dale Chall Readability Metric",
                 "Spache Readability Metric",
                 "Gunning Fog Metric",
                 "Lix Metric",
                 "Rix Metric",
                 "Fernandez Huerta Metric",
                 "Szigriszt Pazos Metric",
                 "Gutierrez Polini Metric",
                 "Crawford Metric",
                 "Mcalpine Eflaw Metric",
                 "Guiraud R Metric",
                 "Herdan C Metric",
                 "Dugast K Metric",
                 "Maas A2 Metric",
                 "Dugast U Metric",
                 "Tuldava LN Metric",
                 "Brunet W Metric",
                 "Corrected Token Type Ratio",
                 "Summer S Metric",
                 "Sichel S Metric",
                 "Michea M Metric",
                 "Honore H Metric",
                 "Entropy Metric",
                 "Yule K Metric",
                 "Simpson D Metric",
                 "Herdan VM Metric",
                 "Coleman Liau Metric",
                 "Linsear Write Metric",
                 "Smog Metric",
                 "Threshold Word Length H Ratio",
                 "Threshold Word Length L Ratio",
                 "Threshold Syllables Per Word H Ratio",
                 "Threshold Syllables Per Word L Ratio",
                 "Threshold Sentence Length H Ratio",
                 "Threshold Sentence Length L Ratio"]

def clean_text(tokens):
    words = [word.lower() for word in tokens if word.isalnum()]
    return words

def average_word_length(tokens):
    avg_word_len = 0
    sum_words_len = 0
    if len(tokens) > 0:
        for word in tokens:
            sum_words_len += len(word)

        avg_word_len = sum_words_len / len(tokens)

    return avg_word_len

def average_syllables_per_word(words):
    avg_word_len = 0
    sum_words_len = 0
    if len(words) > 0:
        for word in words:
            sum_words_len += textstat.syllable_count(word)

        avg_word_len = sum_words_len / len(words)

    return avg_word_len

def average_sentence_length(sentences):
    sent_len_sum = 0
    avg_sent_len = 0
    if len(sentences) > 0:
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            sent_len_sum += len(tokens)

        avg_sent_len = sent_len_sum / len(sentences)

    return avg_sent_len

def average_sentence_length_chars(sentences):
    sent_len_sum = 0
    avg_sent_len = 0
    if len(sentences) > 0:
        for sentence in sentences:
            sent_len_sum += len(sentence)

        avg_sent_len = sent_len_sum / len(sentences)

    return avg_sent_len

def average_word_frequency_class(words):
    sum_class_freq = 0
    avg_class_freq = 0
    if len(words) > 0:
        for word in words:
            sum_class_freq += word_class_mapping.get(word, 20)

        avg_class_freq = sum_class_freq / len(words)

    return avg_class_freq

def threshold_syllables_per_word(words, threshold = 2):
    count_hi = 0
    count_lo = 0

    if len(words) > 0:
        for word in words:
            if textstat.syllable_count(word) >= threshold:
                count_hi += 1

            else:
                count_lo += 1

        count_hi = count_hi / len(words)
        count_lo = count_lo / len(words)

    return count_hi, count_lo

def threshold_word_length(tokens, threshold = 5):
    words = [word for word in tokens if word not in stop]
    count_hi = 0
    count_lo = 0

    if len(words) > 0:

        for word in words:
            if len(word) >= threshold:
                count_hi += 1

            else:
                count_lo += 1

        count_hi = count_hi / len(words)
        count_lo = count_lo / len(words)

    return count_hi, count_lo

def threshold_sentence_length(sentences, threshold = 17):
    count_hi = 0
    count_lo = 0

    if len(sentences) > 0:
        for text in sentences:
            tokens = word_tokenize(text)

            if len(tokens) >= threshold:
                count_hi += 1

            else:
                count_lo += 1

        count_hi = count_hi / len(sentences)
        count_lo = count_lo / len(sentences)

    return count_hi, count_lo

def type_token_ratio(tokens):
    ttr = 0
    if len(tokens) > 0:
        ttr = len(set(tokens)) / len(tokens)

    return ttr

def special_characters_ratio(text):
    sum_special_chars = 0
    special_chars = ['~', '@', '#', '$', '%', '^', '&', '*','-', '_',
                     '=', '+', '>','<', '[', ']', '{', '}', '/', '\\', '|']
    for i in text:
        if i in special_chars:
            sum_special_chars += 1

    return sum_special_chars / len(text)

def puncuations_ratio(text):
    punctuations = [',', '.', '?', '!', ':', ';', '’', '"']
    sum_punct = 0

    for i in text:
        if i in punctuations:
            sum_punct += 1

    return sum_punct / len(text)

def uppercase_ratio(text):
    sum_uppercase = 0
    for char in text:
        if char.isupper():
            sum_uppercase += 1

    return sum_uppercase / len(text)

def digit_ratio(text):
    sum_digits = 0
    for char in text:
        if char.isdigit():
            sum_digits += 1

    return sum_digits / len(text)

def stopword_ratio(words):
    sum_stopwords = 0
    avg_stopwords = 0
    if len(words) > 0:
        for word in words:
            if word in stop:
                sum_stopwords += 1

        avg_stopwords = sum_stopwords / len(words)

    return avg_stopwords

def functional_words_ratio(words):
    sum_function_words = 0
    avg_function_words = 0
    if len(words) > 0:
        for word in words:
            if word in functionwords:
                sum_function_words += 1

        avg_function_words = sum_function_words / len(words)

    return avg_function_words

def hapax_legomena_ratio(tokens):
    V1 = 0
    V2 = 0
    freqs = dict()
    for word in tokens:
        if word not in freqs:
            freqs[word] = 1

        else:
            freqs[word] += 1

    for word in freqs:
        if freqs[word] == 1:
            V1 += 1

        elif freqs[word] == 2:
            V2 += 1

    h = V1 / len(tokens)
    d = V2 / len(tokens)

    return h, d

def flesch_reading_ease_metric(text):
    return textstat.flesch_reading_ease(text)

def flesch_kincaid_grade_metric(text):
    return textstat.flesch_kincaid_grade(text)

def dale_chall_readability_metric(text):
    return textstat.dale_chall_readability_score(text)

def new_dale_chall_readability_metric(text):
   return textstat.dale_chall_readability_score_v2(text)

def gunning_fog_metric(text):
    return textstat.gunning_fog(text)

def lix_metric(text):
    return textstat.lix(text)

def rix_metric(text):
    return textstat.rix(text)

def spache_readability_metric(text):
    return textstat.spache_readability(text)

def fernandez_huerta_metric(text):
    return textstat.fernandez_huerta(text)

def szigriszt_pazos_metric(text):
    return textstat.szigriszt_pazos(text)

def gutierrez_polini_metric(text):
    return textstat.gutierrez_polini(text)

def crawford_metric(text):
    return textstat.crawford(text)

def mcalpine_eflaw_metric(text):
    return textstat.mcalpine_eflaw(text)

def automated_readability_metric(text):
    return textstat.automated_readability_index(text)

def coleman_liau_metric(text):
    return textstat.coleman_liau_index(text)

def linsear_write_metric(text):
    return textstat.linsear_write_formula(text)

def smog_metric(text):
    return textstat.smog_index(text)

def guiraud_r_metric(num_types, num_tokens):
    try:
        r = complexity.guiraud_r(num_types, num_tokens)

    except:
        r = 0.0

    return r

def herdan_c_metric(num_types, num_tokens):
    try:
        c = complexity.herdan_c(num_types, num_tokens)

    except:
        c = 0.0

    return c

def dugast_k_metric(num_types, num_tokens):
    try:
        k = complexity.dugast_k(num_types, num_tokens)

    except:
        k = 0.0

    return k

def maas_a2_metric(num_types, num_tokens):
    try:
        a2 = complexity.maas_a2(num_types, num_tokens)

    except:
        a2 = 0.0

    return a2

def dugast_u_metric(num_types, num_tokens):
    try:
        u = complexity.dugast_u(num_types, num_tokens)

    except:
        u = 0.0

    return u

def tuldava_ln_metric(num_types, num_tokens):
    try:
        ln = complexity.tuldava_ln(num_types, num_tokens)

    except:
        ln = 0.0

    return ln

def brunet_w_metric(num_types, num_tokens):
    try:
        w = complexity.brunet_w(num_types, num_tokens)

    except:
        w = 0.0

    return w

def cttr_metric(num_types, num_tokens):
    try:
        cttr = complexity.cttr(num_types, num_tokens)

    except:
        cttr = 0.0

    return cttr

def summer_s_metric(num_types, num_tokens):
    try:
        s = complexity.summer_s(num_types, num_tokens)

    except:
        s = 0.0

    return s

def sichel_s_metric(num_types, freq_spectrum):
    try:
        s = complexity.sichel_s(num_types, freq_spectrum)

    except:
        s = 0.0

    return s

def michea_m_metric(num_types, freq_spectrum):
    try:
        m = complexity.michea_m(num_types, freq_spectrum)

    except:
        m = 0.0

    return m

def honore_h_metric(num_types, num_tokens, freq_spectrum):
    try:
        h = complexity.honore_h(num_types, num_tokens, freq_spectrum)

    except:
        h = 0.0

    return h

def entropy_metric(num_tokens, freq_spectrum):
    try:
        e = complexity.entropy(num_tokens, freq_spectrum)

    except:
        e = 0.0

    return e

def yule_k_metric(num_tokens, freq_spectrum):
    try:
        k = complexity.yule_k(num_tokens, freq_spectrum)

    except:
        k = 0.0

    return k

def simpson_d_metric(num_tokens, freq_spectrum):
    try:
        d = complexity.simpson_d(num_tokens, freq_spectrum)

    except:
        d = 0.0

    return d

def herdan_vm_metric(num_types, num_tokens, freq_spectrum):
    try:
        vm = complexity.herdan_vm(num_types, num_tokens, freq_spectrum)

    except:
        vm = 0.0

    return vm

def extract(text):

    sentences = sent_tokenize(text)
    tokens = word_tokenize(text)
    words = clean_text(tokens)

    bow = collections.Counter(tokens)
    num_tokens = len(tokens)
    num_types = len(bow)

    freq_spectrum = collections.Counter(bow.values())
    pd_freq_spectrum = pd.Series(freq_spectrum)

    vector = []

    a = average_word_length(tokens)
    vector.append(a)

    a = average_syllables_per_word(words)
    vector.append(a)

    a = average_sentence_length(sentences)
    vector.append(a)

    a = average_sentence_length_chars(sentences)
    vector.append(a)

    a = average_word_frequency_class(words)
    vector.append(a)

    r = type_token_ratio(tokens)
    vector.append(r)

    r = special_characters_ratio(text)
    vector.append(r)

    r = puncuations_ratio(text)
    vector.append(r)

    r = uppercase_ratio(text)
    vector.append(r)

    r = digit_ratio(text)
    vector.append(r)

    r = stopword_ratio(words)
    vector.append(r)

    r = functional_words_ratio(words)
    vector.append(r)

    h, d = hapax_legomena_ratio(tokens)
    vector.append(h)
    vector.append(d)

    m = automated_readability_metric(text)
    vector.append(m)

    m = flesch_reading_ease_metric(text)
    vector.append(m)

    m = flesch_kincaid_grade_metric(text)
    vector.append(m)

    m = dale_chall_readability_metric(text)
    vector.append(m)

    m = new_dale_chall_readability_metric(text)
    vector.append(m)

    m = spache_readability_metric(text)
    vector.append(m)

    m = gunning_fog_metric(text)
    vector.append(m)

    m = lix_metric(text)
    vector.append(m)

    m = rix_metric(text)
    vector.append(m)

    m = fernandez_huerta_metric(text)
    vector.append(m)

    m = szigriszt_pazos_metric(text)
    vector.append(m)

    m = gutierrez_polini_metric(text)
    vector.append(m)

    m = crawford_metric(text)
    vector.append(m)

    m = mcalpine_eflaw_metric(text)
    vector.append(m)

    m = guiraud_r_metric(num_types, num_tokens)
    vector.append(m)

    m = herdan_c_metric(num_types, num_tokens)
    vector.append(m)

    m = dugast_k_metric(num_types, num_tokens)
    vector.append(m)

    m = maas_a2_metric(num_types, num_tokens)
    vector.append(m)

    m = dugast_u_metric(num_types, num_tokens)
    vector.append(m)

    m = tuldava_ln_metric(num_types, num_tokens)
    vector.append(m)

    m = brunet_w_metric(num_types, num_tokens)
    vector.append(m)

    m = cttr_metric(num_types, num_tokens)
    vector.append(m)

    m = summer_s_metric(num_types, num_tokens)
    vector.append(m)

    m = sichel_s_metric(num_types, freq_spectrum)
    vector.append(m)

    m = michea_m_metric(num_types, freq_spectrum)
    vector.append(m)

    m = honore_h_metric(num_types, num_tokens, freq_spectrum)
    vector.append(m)

    m = entropy_metric(num_tokens, pd_freq_spectrum)
    vector.append(m)

    m = yule_k_metric(num_tokens, pd_freq_spectrum)
    vector.append(m)

    m = simpson_d_metric(num_tokens, pd_freq_spectrum)
    vector.append(m)

    m = herdan_vm_metric(num_types, num_tokens, pd_freq_spectrum)
    vector.append(m)

    m = coleman_liau_metric(text)
    vector.append(m)

    m = linsear_write_metric(text)
    vector.append(m)

    m = smog_metric(text)
    vector.append(m)

    h, l = threshold_word_length(tokens)
    vector.append(h)
    vector.append(l)

    h, l = threshold_syllables_per_word(words)
    vector.append(h)
    vector.append(l)

    h, l = threshold_sentence_length(sentences)
    vector.append(h)
    vector.append(l)

    processed_vector = [0.0 if math.isnan(x) else x for x in vector]

    return processed_vector

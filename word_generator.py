import pandas as pd
import numpy as np
import re
import string

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
import spacy
nlp = spacy.load("en_core_web_sm")
from sklearn.model_selection import train_test_split

# Data Attributes
# def averages(text):
#     """
#     This function calculates the average number of words per sentence and the average number of sentences per entry.
#     @param text: pandas series; array of strings
#     @rvalue avg_words_per_sentence: a double representing the average number of words per sentence across all sentences
#     @rvalue avg_num_sentences: a double representing average number of sentences for each review 
#     """
#     total_sentences = 0
#     total_words = 0
#     # for each processed document of the text
#     for doc in nlp.pipe(text, disable=["ner", "tagger"]): # nlp.pipe includes different components of the text
#         sentences = list(doc.sents) # extract all sentences for each entry
#         total_sentences += len(sentences) # total number of sentences in text
#         total_words += sum([len(sentence) for sentence in sentences]) # total number of words in text
    
#     avg_words_per_sentence = total_words / total_sentences if total_sentences else 0 # average number of words per sentence
#     avg_num_sentences = total_sentences / len(text) if len(text) else 0 # average number of sentences across text
    
#     return avg_words_per_sentence, avg_num_sentences

# avg_words, avg_sent = averages(data['review'][:len(data['review'])//10])
# output: (20.16343979755327, 13.3566)

# Truncate text
def truncate_text(text, min_words, min_sent):
    """
    This function takes in a text and truncates it so that there are only two sentences and if the number of words in the 
    text is less than min_words, it adds another sentence to reach at least min_words.
    @param text: string object to be truncated
    @param min_words: int object representing minimum number of words text should have
    @param min_sent: int object representing minimum number of sentences the text should have
    @rvalue: string object representing truncated text
    """
    doc = nlp(text) # create a document that stores different components of text
    sentences = list(doc.sents) # list of sentences of text
    
    new_text = []
    word_count = 0 # records number of words in text
    sent_count = 0 # records number of sentences in text
    
    # for each sentence in text
    for sentence in sentences:
        sent_word_count = len(sentence.text.split()) # count number of words in each sentence
        # add a sentence to meet requirements
        if word_count + sent_word_count <= min_words or sent_count < min_sent:
            new_text.append(sentence.text) # add sentence
            word_count += sent_word_count # keep track of words added
            sent_count += 1 # keep track of sentences added
        # once requirements are met break
        if word_count >= min_words and sent_count >= min_sent:
            break
    
    # if there are less than min_words in text, and no remaining sentences to add, keep text as is
    return ' '.join(new_text)

def modify_data(data, col_name, div, min_words, min_sent):
    """
    This function returns a new dataset with a "new_data" column where the html tags are removed and then the truncate_text
    function is applied to the col_name of the data. 
    @param data: dataset with a text column
    @param col_name: string representing column name of text column
    @param div: int, determines what fraction of data to keep
    @param min_words: int object representing minimum number of words text should have
    @param min_sent: int object representing minimum number of sentences the text should have
    @rvalue new_data: dataset, with a "new_data" column
    """
    new_data = data.iloc[:len(data)//div].copy()
    new_data[col_name] = new_data[col_name].apply(lambda x: re.sub('<.*?>', ' ', x))
    new_data["new_data"] = new_data[col_name].apply(lambda x: truncate_text(x, min_words = min_words, min_sent = min_sent))
    return new_data

# Preprocessing Data
def clean(data, column):
    """
    This function takes in a data frame and column name of the text data. It converts all letters to lowercase, removes HTML tags,
    removes punctuation, removes unnecessary spaces, and removes duplicates.
    @param data: data frame with string column
    @param column: column name of string column
    @rvalue clean_data: pandas data frame with cleaned text
    """
    clean_data = (data[column] # Reduce the data to a specific column
                .str.lower() # Convert to lowercase
                .apply(lambda x: re.sub('<.*?>', ' ', x)) # Replace HTML tags with a space
                .apply(lambda x: re.sub(r'[^\w\s]', '', x)) # Remove punctuation
                .apply(lambda x: re.sub(r'\s{2,}', ' ', x)) # Replace 2+ consecutive spaces with a single space
                .drop_duplicates()) # Remove duplicates
    return clean_data

# Tokenize Data
# represent each word as a numerical value
def tokenize(text):
    """
    This function tokenizes text data, representing each word as a numerical value.
    @param text: string object
    @rvalue text: string object tokenized
    @rvalue total_words: length of word index + 1
    @rvalue tokenizer: tokenizer from tensorflow package
    """
    tokenizer = Tokenizer() 
    tokenizer.fit_on_texts(text) # fit on series of text
    total_words = len(tokenizer.word_index) + 1 # length of word index
    return text, total_words, tokenizer


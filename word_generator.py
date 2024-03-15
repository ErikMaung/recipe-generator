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

def split_data(data):
    """
    The function splits the data into training, validation, and testing sets. The training data is 80% of the data, while the 
    validation and testing sets equally make up the other 20% of the data (they are both 10% of data).
    @param data: a dataset
    @rvalue train_data: 80% of original data
    @rvalue val_data: 10% of original data
    @rvalue test_data: 10% of original data
    """
    # training data is 80% of data
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42) 
    # validation data and testing data are both 10% of data
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42) 
    return train_data, val_data, test_data

def tokenize(train_data):
    """
    This function tokenizes the training text data and then returns the train_data, total_words, and tokenizer.
    @param text: string object
    @rvalue text: string object tokenized
    @rvalue total_words: length of word index + 1
    @rvalue tokenizer: tokenizer from tensorflow package
    """
    tokenizer = Tokenizer() 
    tokenizer.fit_on_texts(train_data) # fit on series of text
    total_words = len(tokenizer.word_index) + 1 # length of word index
    return train_data, tokenizer

def input_seq(train, val, test, tokenizer):
    """
    This function creates input sequences for the Sequential model. The input sequences are created by using n grams of 
    each tokenized review. Each n gram in the sequence is then padded with zeros to the length of the largest n gram sequence
    in the training data.The x variable represents the predictor data, meaning it is all the data in the input sequence, minus the last token. The 
    y variable is the target and what the model aims to predict. Therefor the y variable is the last token "word" in the input
    sequences.
    @param train, val, test: text objects with training, validation, and testing data
    @param tokenizer: tokenizer object from tensorflow
    @rvalue train_x, val_x, test_x: numpy arrays of integers (represents predictor data)
    @rvalue train_y, val_y, test_y: numpy arrays of integers (represents target data)
    @rvalue total_words: int, length of word index
    """
    # create n gram sequences
    def create_sequences(text):
        """
        This function creates n gram sequences for the text.
        @param text: string object
        """
        input_sequences = []
        for line in text: # for each review
            token_list = tokenizer.texts_to_sequences([line])[0] # map each unique word to an integer with tokenizer
            # Creating n gram for each review
            for i in range(1, len(token_list)): 
                n_gram_sequence = token_list[:i+1]
                input_sequences.append(n_gram_sequence) # input_sequences is a list of sequences from tokenized reviews
        return input_sequences
    
    # total words
    total_words = len(tokenizer.word_index) + 1
    
    # create n gram sequences for training, validation, and testing data
    train_seq = create_sequences(train)
    val_seq = create_sequences(val)
    test_seq = create_sequences(test)
    
    # padding sequences so each sequence in input_sequences has the same length
    max_sequence_len = max(max([len(x) for x in train_seq]),
                           max([len(x) for x in val_seq]),
                           max([len(x) for x in test_seq])) # identify length of largest sequence
    
    # pre pad sequences with zeros if length of max_sequence_len is not met
    train_seq = pad_sequences(train_seq, maxlen=max_sequence_len, padding='pre')
    val_seq = pad_sequences(val_seq, maxlen=max_sequence_len, padding='pre')
    test_seq = pad_sequences(test_seq, maxlen=max_sequence_len, padding='pre')

    # create predictor and target data 
    def split_sequences(sequences):
        """
        @param sequences:
        @rvalue x: numpy array of integers (represents predictor data)
        @rvalue y: numpy array of integers (represents target data)
        """
        x = sequences[:, :-1] # the tokenized sequences minus the last token
        y = sequences[:, -1] # the last token for each tokenized sequence
        return x, y
    
    # generate predictor and target data for training, validation, and testing data
    train_x, train_y = split_sequences(train_seq)
    val_x, val_y = split_sequences(val_seq)
    test_x, test_y = split_sequences(test_seq)
    
    return train_x, train_y, val_x, val_y, test_x, test_y, total_words, max_sequence_len

def prepare_for_model(train_x, train_y, val_x, val_y, test_x, test_y, total_words):
    """
    This function applies the one-hot encoding to the predictor data and then converts all the data into numpy arrays.
    @param train_x, val_x, test_x: numpy arrays of integers (represents predictor data)
    @param train_y, val_y, test_y: numpy arrays of integers (represents target data)
    @rvalue train_x, val_x, test_x: numpy arrays of integers (represents predictor data)
    @rvalue train_y, val_y, test_y: numpy arrays of one-hot encoded data (represents target data)
    """
    # convert target data to binary class matrices
    train_y = np.array(tf.keras.utils.to_categorical(train_y, num_classes=total_words))
    val_y = np.array(tf.keras.utils.to_categorical(val_y, num_classes=total_words))
    test_y = np.array(tf.keras.utils.to_categorical(test_y, num_classes=total_words))
    
    # convert data to numpy arrays to ensure model compatability
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    val_x = np.array(val_x)
    val_y = np.array(val_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    
    return train_x, train_y, val_x, val_y, test_x, test_y




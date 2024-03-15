import re # library for working with regular expressions
import pandas as pd
import numpy as np
import spacy # library for nlp tasks
nlp = spacy.load("en_core_web_sm")
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer # library for tokenizing data
from tensorflow.keras.preprocessing.sequence import pad_sequences





def help_truncate(text, min_words, min_sent):
    """
    This function takes in a string and truncates it so that there are only two sentences, and if the number of words in the 
    string is less than min_words, it adds another sentence to reach at least min_words.
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
    
    # if there are less than min_words in text and no remaining sentences to add, keep text as is
    return ' '.join(new_text)





def truncate(data, col_name, div, min_words, min_sent):
    """
    This function takes in a dataset, reduces it to the column containing the text data, subsets the data for efficiency, removes 
    HTML tags, fixes spacing issues, truncates each text element, and returns the processed text data.
    @param data: pandas DataFrame object with a text column
    @param col_name: string object representing column name of text column
    @param div: int object determines what fraction of data to keep
    @param min_words: int object representing minimum number of words text should have
    @param min_sent: int object representing minimum number of sentences the text should have
    @rvalue truncated_data: pandas Series object containing the truncated texts
    """
    truncated_data = (data[col_name] # reduce the data to a specific column
            .iloc[:len(data)//div] # subset the data
            .apply(lambda x: re.sub('<.*?>', ' ', x)) # replace HTML tags with a space
            .apply(lambda x: re.sub(r'\s{2,}', ' ', x)) # replace 2+ consecutive spaces with a single space
            .apply(lambda x: help_truncate(text = x, min_words = min_words, min_sent = min_sent))) # truncate each element
    return truncated_data





def preprocess(data):
    """
    This function takes in textual data and converts all letters to lowercase, removes punctuation, and removes duplicates.
    @param data: pandas Series object containing textual data
    @rvalue preprocessed_text: pandas Series object containing processed textual data
    """
    preprocessed_text = (data
                        .str.lower() # convert to lowercase
                        .apply(lambda x: re.sub(r'[^\w\s]', '', x)) # remove punctuation
                        .drop_duplicates()) # remove duplicates
    return preprocessed_text





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





def tokenize(data):
    """
    This function takes in textual data and tokenizes the words.
    @param data: pandas Series object containing textual data
    @rvalue tokenizer: Tokenizer object
    """
    tokenizer = Tokenizer() # initialize a Tokenizer
    tokenizer.fit_on_texts(data) # tokenize each word
    return tokenizer





def help_create_sequences(texts, tokenizer):
    """
    This function creates n-gram sequences for the each text element.
    @param texts: pandas Series of texts
    @param tokenizer: Tokenizer object
    @rvalue input_sequences: list of lists representing n-grams
    """
    input_sequences = []
    for line in texts: # for each review
        token_list = tokenizer.texts_to_sequences([line])[0] # map each unique word to an integer with tokenizer
        # creating n gram for each review
        for i in range(1, len(token_list)): 
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence) # input_sequences is a list of sequences from tokenized reviews
    return input_sequences





def help_split_sequences(sequences):
    """
    This function separates the last element of each input sequence for the purpose of testing model accuracy.
    @param sequences: 
    @rvalue x: numpy array of integers representing the predictors
    @rvalue y: numpy array of integers representing the targets
    """
    x = sequences[:, :-1] # the tokenized sequences without the last token
    y = sequences[:, -1] # the last token for each tokenized sequence
    return x, y





def input_seq(train, val, test, tokenizer):
    """
    This function creates input sequences for the Keras Sequential model. The input sequences are created by using n-grams of 
    each tokenized text element. Each n-gram in the sequence is padded with zeros to the length of the largest n-gram sequence
    in the training data. The x variable represents the predictor data, meaning it is all the data in the input sequence, minus the last token. The 
    y variable represents the target data, which is what the model aims to predict. Therefore, the y variable is the last token "word" in the input
    sequences.
    @param train, val, test: text objects representing training, validation, and testing data
    @param tokenizer: tokenizer object from tensorflow
    @rvalue train_x, val_x, test_x: numpy arrays of integers representing predictor data
    @rvalue train_y, val_y, test_y: numpy arrays of integers representing target data
    @rvalue total_words: int object representing the length of the word index
    @rvalue max_sequence_len: int object representing the max input sequence length
    """
    # total words
    total_words = len(tokenizer.word_index) + 1
    
    # create n gram sequences for training, validation, and testing data
    train_seq = help_create_sequences(train, tokenizer)
    val_seq = help_create_sequences(val, tokenizer)
    test_seq = help_create_sequences(test, tokenizer)
    
    # padding sequences so each sequence in input_sequences has the same length
    max_sequence_len = max(max([len(x) for x in train_seq]),
                       max([len(x) for x in val_seq]),
                       max([len(x) for x in test_seq])) # identify length of largest sequence
    
    # pre-pad sequences with zeros if length of max_sequence_len is not met
    train_seq = pad_sequences(train_seq, maxlen=max_sequence_len, padding='pre')
    val_seq = pad_sequences(val_seq, maxlen=max_sequence_len, padding='pre')
    test_seq = pad_sequences(test_seq, maxlen=max_sequence_len, padding='pre')
    
    # generate predictor and target data for training, validation, and testing data
    train_x, train_y = help_split_sequences(sequences = train_seq)
    val_x, val_y = help_split_sequences(sequences = val_seq)
    test_x, test_y = help_split_sequences(sequences = test_seq)
    
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
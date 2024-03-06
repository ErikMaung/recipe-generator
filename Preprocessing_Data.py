import pandas as pd
import re # Module for working with regular expressions
from tensorflow.keras.preprocessing.text import Tokenizer # Module for tokenizing data

def clean(data, column):
    clean_data = (data[column] # Reduce the data to a specific column
                .str.lower() # Convert to lowercase
                .apply(lambda x: re.sub('<.*?>', ' ', x)) # Replace HTML tags with a space
                .apply(lambda x: re.sub(r'[^\w\s]', '', x)) # Remove punctuation
                .apply(lambda x: re.sub(r'\s{2,}', ' ', x)) # Replace 2+ consecutive spaces with a single space
                .drop_duplicates()) # Remove duplicates
    return clean_data

def tokenize(text_data):
    tokenizer = Tokenizer() # Initialize a Tokenizer
    tokenizer.fit_on_texts(text_data) # Learn the vocabulary from the text data
    return tokenizer

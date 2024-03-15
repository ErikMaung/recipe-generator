# review-generator
UCLA Math 156 Group 4 Final Project Winter 2024

## Instructions for Replication
In order to run our project and reproduce our results, follow these steps:
1. Clone our repository by running `git clone https://github.com/ErikMaung/review-generator.git` in your command line or with GitHub Desktop.
2. Double check that you have the following files: `word_generator.py`, `Project.ipynb`, `reviews.csv`, `README.md`.
2.5. If you would like, you may also obtain the reviews dataset straight from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). However, **be sure to rename the dataset** to `reviews.csv` to ensure the notebook reads the file.
3. Open `Project.ipynb` in your desired client. Ensure `word_generator.py` and `reviews.csv` are in your environment.
4. Run all cells sequentially in the notebook. Additional guidelines and troubleshooting information can be found in comments in each cell.

TEMPORARY:
```python
def generate_review(model, tokenizer, seed_text, max_sequence_len, review_length):
    """
    Generate a review from a seed text.

    Parameters:
    - model: Trained Keras model for text generation.
    - tokenizer: Tokenizer used for training the model.
    - seed_text: Initial text to start the review generation.
    - max_sequence_len: Maximum length of sequences used during training.
    - review_length: Desired length of the generated review.

    Returns:
    - A string containing the generated review.
    """
    for _ in range(review_length):
        # tokenize current text
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        # pad sequence
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        # predict next word (probability distribution over vocab)
        predicted = model.predict(token_list, verbose=0)
        # convert to single word index
        predicted_index = np.argmax(predicted, axis=-1)[0]
        # convert word index back to word
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        # append predicted word to text
        seed_text += " " + output_word
        # consider putting stopping criterion?
        # EX:  if output_word == 'endToken': break
    return seed_text

# Example usage
# Assuming you have a 'tokenizer' that was used during model training
seed_text = "This movie"
review_length = 50  # Generate a review of approximately 50 words
generated_review = generate_review(model, tokenizer, seed_text, max_sequence_len, review_length)
print(generated_review)

```

import re
import numpy as np
import pandas as pd

from ftfy import fix_text
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity


import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from pyresparser import ResumeParser


stop_words  = set(stopwords.words('english'))


def text_preprocessing(input_text, n=3):
    """
    This function preprocesses the input text and returns ngrams

    [Arguments]:
    input_text: input text to preprocess
    n: (default=3), the function preprocesses the input_text

    [Return]
    list_ngrams: list of tokens with each token having size of `n`
    
    """

    # Fix text
    input_text = fix_text(input_text)

    # Remove non-ASCII characters
    input_text = input_text.encode("ascii", errors="ignore").decode()

    # Convert the string to lowercase
    input_text = input_text.lower()

    # Define a list of characters to remove
    chars_to_remove = [")", "(", ".", "|", "[", "]", "{", "}", "'"]

    # Create a regular expression pattern to match any of these characters
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'

    # Remove characters specified in chars_to_remove list
    input_text = re.sub(rx, '', input_text)

    # Replace '&' with 'and'
    input_text = input_text.replace('&', 'and')

    # Replace ',', '-' with space
    input_text = input_text.replace(',', ' ')
    input_text = input_text.replace('-', ' ')

    # Normalize case - capital at the start of each word
    input_text = input_text.title()

    # Get rid of multiple spaces and replace with a single
    input_text = re.sub(' +', ' ', input_text).strip()

    # Pad names for n-grams
    input_text = ' ' + input_text + ' '

    # Remove characters [,.-/] and spaces followed by 'BD'
    input_text = re.sub(r'[,-./]|\sBD', r'', input_text)

    # Create n-grams from the string
    ngrams = zip(*[input_text[i:] for i in range(n)])
    list_ngrams = [''.join(ngram) for ngram in ngrams]

    # Return the list of n-grams
    return list_ngrams

    

def fit_knn_model(job_vectors):
    """
    Fits a KNN model using the provided job vectors.

    Parameters:
    job_vectors (numpy.ndarray): The job vectors for the KNN model.

    Returns:
    NearestNeighbors: The fitted KNN model.
    """
    nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(job_vectors)
    return nbrs


def get_nearest_job_knn(user_vector, nbrs):
    """
    Finds the nearest job using the KNN model.

    Parameters:
    user_vector (numpy.ndarray): The vector representing the user.
    nbrs (NearestNeighbors): The fitted KNN model.

    Returns:
    tuple: A tuple containing distances and indices of nearest jobs.
    """
    distances, indices = nbrs.kneighbors(user_vector)
    return distances, indices

def calculate_cosine_similarity(job_vectors, user_vector):
    """
    Calculates cosine similarity between job vectors and user vector.

    Parameters:
    job_vectors (numpy.ndarray): The job vectors.
    user_vector (numpy.ndarray): The vector representing the user.

    Returns:
    numpy.ndarray: The cosine similarity scores.
    """
    similarities = cosine_similarity(job_vectors, user_vector)
    return similarities


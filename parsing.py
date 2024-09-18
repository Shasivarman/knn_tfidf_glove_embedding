import spacy
from spacy.matcher import Matcher
#import PyPDF2
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pandas as pd
import re
from model import text_preprocessing
from fuzzywuzzy import process
# Load the Spacy English model
nlp = spacy.load('en_core_web_sm')


data = pd.read_csv("job_descriptions.csv")
# Read skills from CSV file
unique_skills = pd.Series(data['skills'].unique())
# Create a Matcher object
matcher = Matcher(nlp.vocab)




def get_tfidf_vectors(data_df, data_input):
    """
    Computes TF-IDF vectors for job skills and user skills.

    Parameters:
    data_df (pandas.DataFrame): The DataFrame containing job skills.
    data_input (pandas.DataFrame): The DataFrame containing user skills.

    Returns:
    tuple: A tuple containing two numpy arrays - (job_tfidf_vectors, user_tfidf_vector).
    """
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(min_df=1, analyzer=text_preprocessing, lowercase=False)

    # Fit the vectorizer on the skills data
    job_tfidf_vectors = vectorizer.fit_transform(data_df['skills_sep'])
    user_tfidf_vector = vectorizer.transform(data_input)

    return job_tfidf_vectors, user_tfidf_vector




def get_glove_embeddings(data_df, data_input, ):
    """
    Computes GloVe embeddings for job skills and user skills.

    Parameters:
    data_df (pandas.DataFrame): The DataFrame containing job skills.
    data_input (pandas.DataFrame): The DataFrame containing user skills.
    glove_file_path (str): The file path to the GloVe embeddings file.

    Returns:
    tuple: A tuple containing two numpy arrays - (job_glove_vectors, user_glove_vectors).
    """

    # Load GloVe vectors from file
    glove_vectors = {}
    with open('glove.6B.100d.txt', 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            glove_vectors[word] = vector
    
    # Define a function to compute average GloVe vector for a text
    def get_average_glove_vector(text):
        words = text.split()
        vectors = [glove_vectors.get(word, np.zeros(len(glove_vectors['a']))) for word in words]
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(len(glove_vectors['a']))

    # Compute GloVe vectors for job skills
    job_glove_vectors = np.array([get_average_glove_vector(skill) for skill in data_df['skills_sep']])

    # Compute GloVe vectors for user skills
    user_glove_vectors = np.array([get_average_glove_vector(skill) for skill in data_input])

    return job_glove_vectors, user_glove_vectors


def create_skill_patterns(paragraph):
    # Process the paragraph with spaCy
    doc = nlp(paragraph)
    # Initialize a list to store skill patterns
    skill_patterns = []
    # Initialize a temporary list to store tokens for a single skill
    current_skill_tokens = []
    # Iterate over tokens in the paragraph
    for token in doc:
        # Check if the token is a comma or "and"
        if token.text == ',' or token.text.lower() == 'and':
            # Add the current skill tokens as a pattern if not empty
            if current_skill_tokens:
                skill_pattern = {"LOWER": {"IN": [token.text.lower() for token in current_skill_tokens]}}
                skill_patterns.append(skill_pattern)
                # Reset the current skill tokens list
                current_skill_tokens = []
        # If the token is not a comma or "and" and is a noun or proper noun, add it to current skill tokens
        elif token.pos_ in ['NOUN', 'PROPN']:
            current_skill_tokens.append(token)
    # Add the last skill pattern if not empty
    if current_skill_tokens:
        skill_pattern = {"LOWER": {"IN": [token.text.lower() for token in current_skill_tokens]}}
        skill_patterns.append(skill_pattern)
    
    return skill_patterns

# Add skill patterns to the matcher
skill_patterns =unique_skills.apply(create_skill_patterns)
for pattern in skill_patterns:
    matcher.add('Skills', [pattern])

def extract_skills(text):
    doc = nlp(text)
    matches = matcher(doc)
    skills = set()
    for match_id, start, end in matches:
        skill = doc[start:end].text
        skills.add(skill)
    return skills

def preprocess_text(input_str, options):
    """
    Finds the element in a list that is most similar to the input.

    Parameters:
    input_str (str): The input string.
    options (list): List of options to search for similarity.

    Returns:
    str: The element from the list that is most similar to the input.
    """
    # Use process.extractOne to find the most similar option
    most_similar, score = process.extractOne(input_str, options)
    
    return most_similar






# def skills_extractor(file_path):
#         # Extract text from PDF
        
#         resume_text = extract_text_from_pdf(file_path)

#         # Extract skills from resume text
#         skills = list(extract_skills(resume_text))
#         return skills

# def gender_extractor(file_path):
#         # Extract text from PDF
        
#         resume_text = extract_text_from_pdf(file_path)

#         # Extract skills from resume text
#         gender = extract_gender(resume_text)
#         return gender
import pandas as pd
from data_preparation import preprocess_data

import parsing
import model

data = pd.read_csv('job_descriptions.csv') 

def get_user_input():
    """
    Get user input for qualification, years of experience, skills, and gender.

    Returns:
    tuple: A tuple containing user inputs - (qualification, experience_years, skills, gender).
    """
    print("Please provide the following information:")
    
    qualification = input("Qualification: ")

    #qualification = 'phd'
    while True:
        try:
            experience_years = int(input("Years of Experience: "))
            #experience_years = 10
            break
        except ValueError:
            print("Please enter a valid number for years of experience.")
    
    skills_input = input("Skills (comma-separated): ")
    #skills_input ='Python, Statistics, Data Handling, Data Visualization,Linear Algebra, Neural Networks, Transfer Learning,Feature Extraction, Deep Learning, Sci-kit Learn, Keras,OpenCV, SQLite, Html, CSS,GUI using Pyqt module, Git GitHub,C, C++, etc.'
    skills = [",".join(skills_input.split(','))]
    gender = input("Gender: ")
    #gender = 'female'
    return qualification, experience_years, skills, gender

qualification, experience_years, skills, gender = get_user_input()

processed_data = preprocess_data(data, qualification, experience_years, gender)

skills = parsing.extract_skills(skills[0])
job_tfidf_vectors, user_tfidf_vector=parsing.get_tfidf_vectors(processed_data,skills)
job_glove_vectors, user_glove_vectors =parsing.get_glove_embeddings(processed_data,skills)



# Calculate cosine similarity for TF-IDF vectors
similarities_tfidf = model.calculate_cosine_similarity( user_tfidf_vector,job_tfidf_vectors)

# Calculate cosine similarity for GloVe vectors
similarities_glove = model.calculate_cosine_similarity( user_glove_vectors,job_glove_vectors)

print("hello")
# Fit KNN model using GloVe vectors
nbrs_glove = model.fit_knn_model(user_glove_vectors)
print("hello 3")

# Get nearest job using KNN with GloVe vectors
distances_glove, indices_glove = model.get_nearest_job_knn(job_glove_vectors, nbrs_glove)

# Fit KNN model using TF-IDF vectors
nbrs_tfidf = model.fit_knn_model(user_tfidf_vector)

# Get nearest job using KNN with TF-IDF vectors
distances_tfidf, indices_tfidf = model.get_nearest_job_knn(job_tfidf_vectors, nbrs_tfidf)

# Process the results
distances_ls = []

# Assuming 'data_df' and 'data' are defined elsewhere
for i, j in enumerate(indices_glove):
    dist = round(distances_glove[i][0], 2)
    temp = [dist]
    distances_ls.append(temp)

matches = pd.DataFrame(distances_ls, columns=['Distance'])

# Merge distances with original data
processed_data['Distance'] = matches['Distance']
sorted_data_knn = processed_data.sort_values(by='Distance')

# Print the top 5 jobs based on KNN model with GloVe vectors
print('The Top 5 Jobs Based on KNN model with GloVe vectors')
print(sorted_data_knn.head(5))

# Process cosine similarity results
scores = pd.DataFrame(similarities_tfidf.flatten(), columns=['similarity_scores'])
processed_data['similarity_scores'] = scores['similarity_scores']
sorted_data_cosine = processed_data.sort_values(by='similarity_scores')

# Print the top 5 jobs based on cosine similarity with TF-IDF vectors
print('The Top 5 Jobs Based on Cosine similarity with TF-IDF vectors')
print(sorted_data_cosine.head(5))

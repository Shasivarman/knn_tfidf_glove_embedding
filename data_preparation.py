import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import parsing

def preprocess_data(data,qual,exp,gender):
    
    unique_qual = list(data['Qualifications'].unique())
    qualification = parsing.preprocess_text(qual,unique_qual)

    print(qualification)
    
    # Extract numerical experience
    data['Min_Experience'] = data['Experience'].apply(lambda x: int(x.split(" to ")[0]))

    def separate_skills(skill_paragraph):
        # Split the skill paragraph using commas and 'and'
        skills = re.split(r',| and ', skill_paragraph)
        
        # Remove leading and trailing spaces from each skill
        skills = ', '.join([skill.strip() for skill in skills if skill.strip()])
        return skills
    
    data['skills_sep']  = data['skills'].apply(separate_skills)

    # Example: Select relevant columns
    selected_columns = ['Company','Min_Experience','Qualifications','Preference','skills_sep'] 
    # Filter data with selected columns
    processed_data = data[selected_columns]
    processed_data = processed_data[processed_data['Qualifications']==qualification]
    processed_data = processed_data[processed_data['Min_Experience'] <= exp]
    processed_data = processed_data[(processed_data['Preference'].str.upper() == gender.upper()) | (processed_data['Preference'].str.upper() == 'BOTH')]

    print(processed_data)
    return processed_data



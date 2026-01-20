import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_csv("fake_job_postings.csv")

# Combine important text columns
df['text'] = df['title'].fillna('') + ' ' + \
             df['company_profile'].fillna('') + ' ' + \
             df['description'].fillna('') + ' ' + \
             df['requirements'].fillna('')

# Text cleaning function
def clean_text(text):
    text = text.lower()                       # lowercase
    text = re.sub(r'http\S+', '', text)       # remove URLs
    text = re.sub(r'[^a-z\s]', '', text)      # remove numbers & symbols
    text = re.sub(r'\s+', ' ', text)          # remove extra spaces
    return text

df['clean_text'] = df['text'].apply(clean_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])

# Target
y = df['fraudulent']

print("Text preprocessing completed")
print("TF-IDF shape:", X.shape)

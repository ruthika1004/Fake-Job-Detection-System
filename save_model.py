import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

url = "https://drive.google.com/uc?id=1fg2wvgk97KTKAcxp61hNiZ34DTJOIoZn"
df = pd.read_csv(url)

df['text'] = df['title'].fillna('') + ' ' + \
             df['company_profile'].fillna('') + ' ' + \
             df['description'].fillna('') + ' ' + \
             df['requirements'].fillna('')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

df['clean_text'] = df['text'].apply(clean_text)

vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['fraudulent']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model and Vectorizer saved!")

import pandas as pd
import re
import shap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("fake_job_postings.csv")

# Combine text columns
df['text'] = df['title'].fillna('') + ' ' + \
             df['company_profile'].fillna('') + ' ' + \
             df['description'].fillna('') + ' ' + \
             df['requirements'].fillna('')

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

df['clean_text'] = df['text'].apply(clean_text)

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['fraudulent']

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# SHAP Explainer
explainer = shap.LinearExplainer(model, X, feature_perturbation="interventional")
shap_values = explainer.shap_values(X[:1])

# Show explanation for first job
feature_names = vectorizer.get_feature_names_out()
important_words = sorted(
    zip(feature_names, shap_values[0]),
    key=lambda x: abs(x[1]),
    reverse=True
)[:10]

print("Top words influencing the prediction:")
for word, score in important_words:
    print(word, ":", score)

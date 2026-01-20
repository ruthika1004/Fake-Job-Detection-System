import pandas as pd
import re

# Load dataset
df = pd.read_csv("fake_job_postings.csv")

# Function to check company email domain
def is_company_email(text):
    if pd.isna(text):
        return False
    free_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']
    for domain in free_domains:
        if domain in text.lower():
            return False
    return True

# Credibility Score Function
def calculate_credibility(row):
    score = 0

    # Email domain check
    if is_company_email(row['company_profile']):
        score += 30

    # Company profile exists
    if pd.notna(row['company_profile']) and len(row['company_profile']) > 50:
        score += 20

    # Salary range present
    if pd.notna(row['salary_range']):
        score += 20

    # Description length check
    if pd.notna(row['description']) and len(row['description']) > 300:
        score += 15

    # Requirements present
    if pd.notna(row['requirements']) and len(row['requirements']) > 100:
        score += 15

    return score

# Apply credibility score
df['credibility_score'] = df.apply(calculate_credibility, axis=1)

# Show sample results
print(df[['title', 'fraudulent', 'credibility_score']].head(10))

import pandas as pd

# Load dataset
df = pd.read_csv("fake_job_postings.csv")

# Show basic information
print("Dataset shape:", df.shape)
print("\nColumns:\n", df.columns)
print("\nFirst 5 rows:\n", df.head())

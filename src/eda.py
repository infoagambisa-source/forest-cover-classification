import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/cover_data.csv")

# View data info
print("Shape of dataset:", df.shape)
print("\nColumn names:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values per column:")
print(df.isnull().sum())

# Target column (Cover_Type)
print("\nTarget class distribution:")
print(df["class"].value_counts())
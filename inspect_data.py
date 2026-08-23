import pandas as pd

# 1. Load dataset from your local data folder
print("Loading dataset...")
df = pd.read_csv("data/real-estate_dataset_addis-ababa_v1.csv")

# 2. Print shape and column names
print("\n=== DATASET SHAPE ===")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n=== COLUMN NAMES ===")
for i, col in enumerate(df.columns):
    print(f"{i+1}. {col}")

print("\n=== FIRST 2 ROWS ===")
print(df.head(2))
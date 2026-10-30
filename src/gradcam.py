import pandas as pd
import os

CSV_PATH = "dataset/HAM10000_metadata.csv"

df = pd.read_csv(CSV_PATH)

print(df.head())
print(df.shape)

print("\nClass distribution:")
print(df["dx"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())
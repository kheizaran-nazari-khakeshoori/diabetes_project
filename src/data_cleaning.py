import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

def load_data(raw_data_path):
    """Load raw data from CSV file"""
    print(f"Loading raw data from {raw_data_path}...")
    df = pd.read_csv(raw_data_path)
    print(f"Raw data loaded. Shape: {df.shape}")
    return df

def clean_data(raw_df):
    """Clean and preprocess the raw dataframe"""
    print("Starting Data Cleaning...")
    
    df = raw_df.copy()
    
    # 1. Encode target column
    le = LabelEncoder()
    df['class'] = le.fit_transform(df['class'])
    print("Target column encoded successfully.")

    # 2. Handle missing values (zeros)
    columns_with_zero = ['plas', 'pres', 'skin', 'insu', 'mass']
    for col in columns_with_zero:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())
    print("Missing values (zeros) handled using Median Imputation.")
    
    print(f"Data cleaning complete. Shape: {df.shape}")
    return df

def save_cleaned_data(cleaned_df, output_path):
    """Save cleaned data to CSV file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
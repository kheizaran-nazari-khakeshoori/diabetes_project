import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def clean_data(raw_data_path, processed_dir_path):
    print("Starting Data Preprocessing...")
    
    os.makedirs(processed_dir_path, exist_ok=True)
    
    # 1. Load dataset
    df = pd.read_csv(raw_data_path)
    print(f"Raw data loaded. Shape: {df.shape}")

    # 2. Encode target column
    le = LabelEncoder()
    df['class'] = le.fit_transform(df['class'])
    print("Target column encoded successfully.")

    # 3. Handle missing values (zeros)
    columns_with_zero = ['plas', 'pres', 'skin', 'insu', 'mass']
    for col in columns_with_zero:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())
    print("Missing values (zeros) handled using Median Imputation.")

    # 4. Separate features and target
    X = df.drop(columns=['class'])
    y = df['class']

    # 5. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 6. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 7. Save processed data
    np.save(os.path.join(processed_dir_path, 'X_train.npy'), X_train_scaled)
    np.save(os.path.join(processed_dir_path, 'X_test.npy'), X_test_scaled)
    np.save(os.path.join(processed_dir_path, 'y_train.npy'), y_train.values)
    np.save(os.path.join(processed_dir_path, 'y_test.npy'), y_test.values)
    df.to_csv(os.path.join(processed_dir_path, 'diabetes_cleaned.csv'), index=False)

    print("Preprocessing complete. Processed files saved in data/processed/")
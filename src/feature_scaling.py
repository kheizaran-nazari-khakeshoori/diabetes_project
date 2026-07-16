from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def scale_features(cleaned_data_path, processed_dir_path):
    print(" Starting Feature Scaling...")

    cleaned_data_path = Path(cleaned_data_path)
    processed_dir_path = Path(processed_dir_path)
    processed_dir_path.mkdir(parents=True, exist_ok=True)

    # Load the cleaned diabetes dataset from the previous step.
    df = pd.read_csv(cleaned_data_path)

    target_column = "outcome" if "outcome" in df.columns else "class"

    # Separate features (X) from the target (y).
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split into 80% train and 20% test while preserving class balance.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize features using statistics learned from the training set only.
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaled arrays for downstream modeling steps.
    np.save(processed_dir_path / "X_train_scaled.npy", X_train_scaled)
    np.save(processed_dir_path / "X_test_scaled.npy", X_test_scaled)
    np.save(processed_dir_path / "y_train.npy", y_train.to_numpy())
    np.save(processed_dir_path / "y_test.npy", y_test.to_numpy())

    print(" Data split into Train/Test and scaled successfully.")
    print(f" Saved scaled arrays in: {processed_dir_path}")


if __name__ == "__main__":
    scale_features("data/processed/diabetes_cleaned.csv", "data/processed/")
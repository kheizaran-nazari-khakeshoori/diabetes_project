from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_data(file_path: str | Path) -> pd.DataFrame:
    """Load the raw diabetes dataset from a CSV file."""
    return pd.read_csv(file_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply a basic cleaning pipeline that works with most tabular CSV datasets."""
    cleaned_df = df.copy()
    cleaned_df.columns = [column.strip().lower().replace(" ", "_") for column in cleaned_df.columns]
    cleaned_df = cleaned_df.drop_duplicates()

    numeric_columns = cleaned_df.select_dtypes(include="number").columns
    categorical_columns = cleaned_df.select_dtypes(exclude="number").columns

    for column in numeric_columns:
        cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].median())

    for column in categorical_columns:
        if not cleaned_df[column].dropna().empty:
            cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mode().iloc[0])

    return cleaned_df


def save_cleaned_data(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save the cleaned dataset to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

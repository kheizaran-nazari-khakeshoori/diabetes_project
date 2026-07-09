from pathlib import Path

from src.data_cleaning import clean_data, load_data, save_cleaned_data


RAW_DATA_PATH = Path("data/raw/diabetes.csv")
CLEAN_DATA_PATH = Path("data/processed/diabetes_cleaned.csv")


def main() -> None:
    raw_df = load_data(RAW_DATA_PATH)
    cleaned_df = clean_data(raw_df)
    save_cleaned_data(cleaned_df, CLEAN_DATA_PATH)
    print(f"Cleaned data saved to {CLEAN_DATA_PATH}")


if __name__ == "__main__":
    main()

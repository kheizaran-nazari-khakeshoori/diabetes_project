import os
from src.data_cleaning import clean_data, load_data, save_cleaned_data
from src.feature_scaling import scale_features
from src.feature_selection import select_features


def main():
    # ۱. تعریف مسیرهای فایل‌ها و پوشه‌ها
    RAW_DATA_PATH = os.path.join("data", "raw", "diabetes.csv")
    PROCESSED_DIR_PATH = os.path.join("data", "processed")
    CLEANED_DATA_PATH = os.path.join(PROCESSED_DIR_PATH, "diabetes_cleaned.csv")
    
    print("🚀 --- Starting Machine Learning Pipeline --- 🚀\n")
    
    # ۲. مرحله اول: تمیزکاری داده‌ها (Data Cleaning)
    raw_df = load_data(RAW_DATA_PATH)
    cleaned_df = clean_data(raw_df)
    save_cleaned_data(cleaned_df, CLEANED_DATA_PATH)
    print("-" * 40)
    
    # ۳. مرحله دوم: مقیاس‌دهی ویژگی‌ها (Feature Scaling)
    scale_features(CLEANED_DATA_PATH, PROCESSED_DIR_PATH)
    print("-" * 40)
    
    # ۴. مرحله سوم: تحلیل و انتخاب ویژگی‌ها (Feature Selection)
    select_features(CLEANED_DATA_PATH)
    
    print("\n🚀 --- Pipeline Executed Successfully! --- 🚀")

if __name__ == "__main__":
    main()
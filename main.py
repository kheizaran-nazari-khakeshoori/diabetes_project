import os
from src.data_cleaning import clean_data, load_data, save_cleaned_data
from src.feature_scaling import scale_features
from src.feature_selection import select_features
from src.model_training import train_models
from src.final_evaluation import evaluate_models
from src.visualization import create_data_visualizations, plot_model_metrics

def main():
    RAW_DATA_PATH = os.path.join("data", "raw", "diabetes.csv")
    PROCESSED_DIR_PATH = os.path.join("data", "processed")
    CLEANED_DATA_PATH = os.path.join(PROCESSED_DIR_PATH, "diabetes_cleaned.csv")
    
    print("--- Starting Machine Learning Pipeline ---\n")
    
    # Step 1: Data Cleaning
    raw_df = load_data(RAW_DATA_PATH)
    cleaned_df = clean_data(raw_df)
    save_cleaned_data(cleaned_df, CLEANED_DATA_PATH)
    create_data_visualizations(cleaned_df, os.path.join("report", "plots"))
    print("-" * 40)
    
    # Step 2: Feature Scaling
    scale_features(CLEANED_DATA_PATH, PROCESSED_DIR_PATH)
    print("-" * 40)
    
    # Step 3: Feature Selection
    select_features(CLEANED_DATA_PATH)
    print("-" * 40)
    
    # Step 4: Model Training
    train_models(PROCESSED_DIR_PATH)
    print("-" * 40)
    
    # Step 5: Model Evaluation
    metrics = evaluate_models(PROCESSED_DIR_PATH)
    plot_model_metrics(metrics, os.path.join("report", "plots"))
    
    print("\n--- Pipeline Executed Successfully! All Done! ---")

if __name__ == "__main__":
    main()
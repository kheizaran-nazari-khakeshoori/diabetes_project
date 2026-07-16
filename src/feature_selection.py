import pandas as pd

def select_features(cleaned_data_path):
    print("Starting Feature Selection & Correlation Analysis...")
    
    # 1. Load cleaned data
    df = pd.read_csv(cleaned_data_path)
    
    # 2. Calculate Pearson correlation matrix
    correlation_matrix = df.corr()
    
    target_column = 'outcome' if 'outcome' in correlation_matrix.columns else 'class'

    # 3. Get correlation with target column
    target_correlation = correlation_matrix[target_column].sort_values(ascending=False)
    
    print(f"\nFeature Correlation with Target ('{target_column}'):")
    print(target_correlation)
    
    # 4. Filter weak features
    threshold = 0.05
    weak_features = target_correlation[abs(target_correlation) < threshold].index.tolist()
    
    if weak_features:
        print(f"\nWeak features found (Correlation < {threshold}): {weak_features}")
    else:
        print(f"\nAll features have a reasonable correlation (> {threshold}) with the target.")
        
    print("Feature analysis complete.")
import numpy as np
import os
import pickle
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def train_models(processed_dir_path):
    print("Starting Model Training...")
    
    # 1. Load scaled training data
    X_train = np.load(os.path.join(processed_dir_path, 'X_train_scaled.npy'))
    y_train = np.load(os.path.join(processed_dir_path, 'y_train.npy'))
    
    # 2. Define baseline model
    baseline_model = DummyClassifier(strategy='most_frequent', random_state=42)
    
    # 3. Define main models
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    svm_model = SVC(kernel='rbf', probability=True, random_state=42)
    
    # 4. Train models
    print("Training Baseline (Dummy) Model...")
    baseline_model.fit(X_train, y_train)
    
    print("Training Random Forest Model...")
    rf_model.fit(X_train, y_train)
    
    print("Training Support Vector Machine (SVM) Model...")
    svm_model.fit(X_train, y_train)
    
    # 5. Save trained models
    models = {
        'baseline': baseline_model,
        'random_forest': rf_model,
        'svm': svm_model
    }
    
    for name, model in models.items():
        model_path = os.path.join(processed_dir_path, f'{name}_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
            
    print("All models trained and saved successfully in data/processed/")
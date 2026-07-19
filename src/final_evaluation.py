import numpy as np
import os
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_models(processed_dir_path):
    print("Starting Model Evaluation...")
    
    # 1. Load test data
    X_test = np.load(os.path.join(processed_dir_path, 'X_test_scaled.npy'))
    y_test = np.load(os.path.join(processed_dir_path, 'y_test.npy'))
    
    model_names = ['baseline', 'random_forest', 'svm']
    metrics = {}
    
    print("\n================ Performance Results ================")
    
    for name in model_names:
        model_path = os.path.join(processed_dir_path, f'{name}_model.pkl')
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        metrics[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
        }
        
        print(f"\nModel: {name.upper()}")
        print(f"   Accuracy:  {acc:.4f}")
        print(f"   Precision: {prec:.4f}")
        print(f"   Recall:    {rec:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        
    print("\n=====================================================")
    print("Model evaluation complete. All metrics calculated!")
    return metrics
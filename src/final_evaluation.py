import numpy as np
import os
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def evaluate_models(processed_dir_path):
    print("📊 Starting Model Evaluation...")
    
    # ۱. بارگذاری داده‌های تست
    X_test = np.load(os.path.join(processed_dir_path, 'X_test_scaled.npy'))
    y_test = np.load(os.path.join(processed_dir_path, 'y_test.npy'))
    
    # ۲. لیست مدل‌هایی که باید ارزیابی شوند
    model_names = ['baseline', 'random_forest', 'svm']
    
    print("\n================ Performance Results ================")
    
    for name in model_names:
        model_path = os.path.join(processed_dir_path, f'{name}_model.pkl')
        
        # بارگذاری فایل مدل
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        # پیش‌بینی روی داده‌های تست
        y_pred = model.predict(X_test)
        
        # محاسبه معیارهای ارزیابی
        # استفاده از zero_division=0 برای جلوگیری از ارور در مدل baseline
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # چاپ نتایج به صورت مرتب
        print(f"\n▶ Model: {name.upper()}")
        print(f"   Accuracy:  {acc:.4f}")
        print(f"   Precision: {prec:.4f}")
        print(f"   Recall:    {rec:.4f}  <-- (مهم‌ترین معیار در پزشکی)")
        print(f"   F1-Score:  {f1:.4f}")
        
    print("\n=====================================================")
    print("[✓] Model evaluation complete. All metrics calculated!")

if __name__ == "__main__":
    # تست اجرای فایل به صورت مستقیم
    evaluate_models("data/processed")
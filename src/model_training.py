import numpy as np
import os
import pickle
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def train_models(processed_dir_path):
    print("🏋️‍♂️ Starting Model Training...")
    
    # ۱. بارگذاری داده‌های اسکیل‌شده و برچسب‌ها
    X_train = np.load(os.path.join(processed_dir_path, 'X_train_scaled.npy'))
    y_train = np.load(os.path.join(processed_dir_path, 'y_train.npy'))
    
    # ۲. تعریف مدل Baseline (طبق مینی‌پروژه: پیش‌بینی رایج‌ترین کلاس)
    baseline_model = DummyClassifier(strategy='most_frequent', random_state=42)
    
    # ۳. تعریف مدل‌های اصلی (Random Forest و SVM)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    svm_model = SVC(kernel='rbf', probability=True, random_state=42) # probability برای محاسبه معیارهای پیشرفته مثل ROC
    
    # ۴. آموزش (Train) مدل‌ها
    print("[~] Training Baseline (Dummy) Model...")
    baseline_model.fit(X_train, y_train)
    
    print("[~] Training Random Forest Model...")
    rf_model.fit(X_train, y_train)
    
    print("[~] Training Support Vector Machine (SVM) Model...")
    svm_model.fit(X_train, y_train)
    
    # ۵. ذخیره کردن مدل‌های آموزش‌دیده برای مرحله ارزیابی
    models = {
        'baseline': baseline_model,
        'random_forest': rf_model,
        'svm': svm_model
    }
    
    for name, model in models.items():
        model_path = os.path.join(processed_dir_path, f'{name}_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
            
    print("[✓] All models trained and saved successfully in data/processed/")

if __name__ == "__main__":
    # تست اجرای فایل به صورت مستقیم
    train_models("data/processed")
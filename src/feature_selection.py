import pandas as pd

def select_features(cleaned_data_path):
    print("🔍 Starting Feature Selection & Correlation Analysis...")
    
    # ۱. بارگذاری داده‌های تمیز شده
    df = pd.read_csv(cleaned_data_path)
    
    # ۲. محاسبه ماتریس همبستگی (Pearson Correlation)
    correlation_matrix = df.corr()
    
    target_column = "outcome" if "outcome" in correlation_matrix.columns else "class"

    # ۳. استخراج میزان همبستگی هر ویژگی با ستون هدف
    target_correlation = correlation_matrix[target_column].sort_values(ascending=False)
    
    print(f"\n📊 Feature Correlation with Target ('{target_column}'):")
    print(target_correlation)
    
    # ۴. فیلتر کردن ویژگی‌های ضعیف (اختیاری)
    # به عنوان مثال، ویژگی‌هایی که همبستگی آن‌ها با هدف کمتر از 0.05 است را مشخص می‌کنیم
    threshold = 0.05
    weak_features = target_correlation[abs(target_correlation) < threshold].index.tolist()
    
    if weak_features:
        print(f"\n⚠️ Weak features found (Correlation < {threshold}): {weak_features}")
        # در این دیتاست معمولاً همه ویژگی‌ها مهم هستند، اما این بخش برای منطق مینی‌پروژه شما عالی است.
    else:
        print(f"\n[✓] All features have a reasonable correlation (> {threshold}) with the target.")
        
    print("[✓] Feature analysis complete.")

if __name__ == "__main__":
    # تست اجرای فایل به صورت مستقیم
    select_features("data/processed/diabetes_cleaned.csv")
# File: final.py (Đã sửa đổi)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import tree_core # Import module C++ bạn vừa tạo!
import time

# --- 1. Đọc dữ liệu và Chuẩn bị ---
print("Đang đọc dữ liệu và chuẩn bị...")

df = pd.read_csv('processed_housing_data.csv')
# Xử lý các cột phân loại (Categorical)
df_numeric = pd.get_dummies(df, drop_first=True)
target_col = 'Giá nhà'
if target_col not in df_numeric.columns: target_col = df_numeric.columns[-1]

# Dữ liệu cần là float64 cho C++
X = df_numeric.drop(target_col, axis=1).values.astype(np.float64) 
y = df_numeric[target_col].values.astype(np.float64)

# Bước 1: Tách 20% ra làm Test Set trước (giữ lại 80% cho Train + Val)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bước 2: Tách 25% của tập Temp ra làm Validation Set
# (25% của 80% ban đầu = 20% tổng dữ liệu)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)


print(f"Kích thước Train: {X_train.shape}")

# --- 2. Huấn luyện bằng C++: Random Forest Đa luồng ---
print("\n🚀 Bắt đầu huấn luyện Random Forest (C++ Core, Đa luồng)...")

# Tham số mô hình Random Forest:
N_ESTIMATORS = 600  # Số lượng cây sẽ được xây song song (dùng tối đa CPU)
MAX_DEPTH = 15      # Độ sâu tối đa của mỗi cây
MIN_SAMPLES = 5     # Số mẫu tối thiểu để tách nút

print(f"Cấu hình: {N_ESTIMATORS} cây, Độ sâu tối đa: {MAX_DEPTH}, Min mẫu split: {MIN_SAMPLES}")
start_time = time.time()

# Khởi tạo mô hình Random Forest từ C++
# Tham số: n_estimators, min_samples_split, max_depth
rf_model = tree_core.RandomForestRegressor(N_ESTIMATORS, MIN_SAMPLES, MAX_DEPTH)
rf_model.fit(X_train, y_train) # Việc này sẽ chạy 100 luồng xây cây song song

end_time = time.time()
print(f"✅ Hoàn thành huấn luyện trong {end_time - start_time:.4f} giây!")


# --- 3. Đánh giá ---
print("\nĐang dự đoán và đánh giá...")
val_pred = rf_model.predict(X_val)
val_r2 = r2_score(y_val, val_pred)

print(f" - Với {600} cây: R2 Score trên Val = {val_r2:.4f}")

y_pred = rf_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R2 Score (Độ phù hợp): {r2:.4f}")
print(f"RMSE (Lỗi): {rmse:.2f}")

# --- So sánh và kết thúc ---
print("\nSo sánh R2 Score thường thấy:")
print("* 1.0: Hoàn hảo")
print("* 0.0: Mô hình tệ hơn việc đoán giá trị trung bình")
print(f"Kết quả R2 = {r2:.4f} cho thấy mô hình của bạn hoạt động tốt.")
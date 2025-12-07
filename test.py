import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. ĐỌC DỮ LIỆU
# Lưu ý: Thay đường dẫn file của bạn vào đây (nhớ dùng r'...' để tránh lỗi đường dẫn)
file_path ='processed_housing_data.csv'

try:
    df = pd.read_csv(file_path)
    print("Đã đọc file thành công!")
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file tại {file_path}. Vui lòng kiểm tra lại đường dẫn.")
    exit()  # Dừng chương trình nếu không đọc được file

# 2. CHUẨN BỊ DỮ LIỆU (X và y)
# Giả sử cột mục tiêu là 'price' (hoặc tên cột giá nhà trong file của bạn)
target_col = 'Giá nhà'

# Xử lý nhanh các biến không phải số (One-Hot Encoding) để Random Forest chạy được
df_numeric = pd.get_dummies(df, drop_first=True)

# Tách X (đặc trưng) và y (nhãn/giá tiền)
if target_col in df_numeric.columns:
    X = df_numeric.drop(target_col, axis=1)
    y = df_numeric[target_col]
else:
    # Nếu không tìm thấy tên cột chính xác, lấy cột cuối cùng làm target (ví dụ)
    print(f"Cảnh báo: Không thấy cột '{target_col}', sử dụng cột cuối cùng làm target.")
    X = df_numeric.iloc[:, :-1]
    y = df_numeric.iloc[:, -1]

# ---------------------------------------------------------
# 3. CHIA DỮ LIỆU THÀNH 3 PHẦN: TRAIN - VAL - TEST
# Tỷ lệ mong muốn: 60% Train - 20% Val - 20% Test
# ---------------------------------------------------------

# Bước 1: Tách 20% ra làm Test Set trước (giữ lại 80% cho Train + Val)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bước 2: Tách 25% của tập Temp ra làm Validation Set
# (25% của 80% ban đầu = 20% tổng dữ liệu)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

print(f"Kích thước tập Train: {X_train.shape} (Dùng để học)")
print(f"Kích thước tập Val:   {X_val.shape}   (Dùng để chỉnh tham số)")
print(f"Kích thước tập Test:  {X_test.shape}  (Dùng để báo cáo kết quả cuối cùng)")
print("-" * 50)

# 4. HUẤN LUYỆN VÀ TINH CHỈNH (TUNING) TRÊN TẬP VALIDATION
# Thử nghiệm các số lượng cây (n_estimators) khác nhau để xem cái nào tốt nhất trên tập Val
n_estimators_list = [50]
best_score = -float('inf')
best_model = None

print("Đang tìm tham số tốt nhất trên tập Validation...")

for n in n_estimators_list:
    model = RandomForestRegressor(n_estimators=n, random_state=42)
    model.fit(X_train, y_train)

    # Kiểm tra trên tập Validation
    val_pred = model.predict(X_val)
    val_r2 = r2_score(y_val, val_pred)

    print(f" - Với {n} cây: R2 Score trên Val = {val_r2:.4f}")

    if val_r2 > best_score:
        best_score = val_r2
        best_model = model

print("-" * 50)
print(f"Mô hình tốt nhất được chọn có: {best_model.n_estimators} cây")

# 5. ĐÁNH GIÁ CUỐI CÙNG TRÊN TẬP TEST (Chỉ chạy 1 lần duy nhất)
print("Đang đánh giá mô hình tốt nhất trên tập Test...")
y_test_pred = best_model.predict(X_test)

# Tính các chỉ số
mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print("\n=== KẾT QUẢ CUỐI CÙNG (FINAL TEST) ===")
print(f"MSE: {mse:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"MAE: {mae:,.2f}")
print(f"R2 Score: {r2:.4f}")
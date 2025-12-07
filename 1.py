import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

## 1. TẢI DỮ LIỆU
df = pd.read_csv('VN_housing_dataset.csv')

# --- SỬA LỖI Ở ĐÂY: Xử lý cột Diện tích ---
# Dữ liệu diện tích có thể chứa ký tự lạ (vd: "50 m2", "50m") khiến nó bị hiểu là string
if df['Diện tích'].dtype == 'object':
    # Loại bỏ chữ " m2", "m2", "m" nếu có (tùy dữ liệu thực tế của bạn)
    df['Diện tích'] = df['Diện tích'].str.replace(' m²', '', regex=False)
    df['Diện tích'] = df['Diện tích'].str.replace('m²', '', regex=False)
    # Chuyển sang số, nếu dòng nào lỗi biến thành NaN (Not a Number)
    df['Diện tích'] = pd.to_numeric(df['Diện tích'], errors='coerce')

# Xử lý cột Giá/m2 (Code cũ của bạn)
if df['Giá/m2'].dtype == 'object':
    df['Giá/m2'] = df['Giá/m2'].str.replace(' triệu/m²', '', regex=False)
    df['Giá/m2'] = df['Giá/m2'].str.replace(',', '.', regex=False)
    df['Giá/m2'] = pd.to_numeric(df['Giá/m2'], errors='coerce')

# --- BÂY GIỜ PHÉP NHÂN MỚI CHẠY ĐƯỢC ---
df['Giá nhà'] = df['Diện tích'] * df['Giá/m2']

# Xóa các dòng không tính được giá (do diện tích hoặc giá/m2 bị NaN)
df = df.dropna(subset=['Giá nhà'])

# Xóa các cột không cần thiết
# Lưu ý: Xóa khỏi df gốc nếu bạn chắc chắn không bao giờ cần nó nữa
df = df.drop(columns=['Giá/m2', 'Ngày', 'STT'], errors='ignore')

# 4. CHUẨN BỊ DỮ LIỆU ĐỂ TÌM OUTLIER
# --- ĐÂY LÀ PHẦN QUAN TRỌNG THEO YÊU CẦU CỦA BẠN ---

# Bước A: Chỉ lấy các cột số để đưa vào Model
# include=[np.number] sẽ tự động lọc lấy int, float, bỏ qua object/string
df_numeric = df.select_dtypes(include=[np.number])

# Bước B: Xử lý missing value cho các cột số (điền trung vị)
df_numeric = df_numeric.fillna(df_numeric.median())

# Bước C: Tách X và Y từ df_numeric
X_train = df_numeric.drop(columns=['Giá nhà'])
Y_train = df_numeric['Giá nhà']

# 5. HUẤN LUYỆN MODEL & TÌM OUTLIER
# Model này chỉ nhìn thấy các con số (Diện tích, Mặt tiền, Số tầng...), không nhìn thấy chữ
model_full = RandomForestRegressor(n_estimators=20, random_state=42)
model_full.fit(X_train, Y_train)

# Tính độ lệch (Residuals)
residuals = np.abs(Y_train - model_full.predict(X_train))

# Thiết lập ngưỡng lọc (Ví dụ: Giữ lại 90% dữ liệu)
threshold = np.percentile(residuals, 50)

# 6. LỌC VÀ LƯU FILE
# Lưu ý: Chúng ta dùng chỉ số (index) của residuals để lọc trên df gốc (có chứa string)
# Điều này giúp file 'quna.csv' vẫn còn nguyên tên Quận, Phường, Hướng nhà...
df_clean = df[residuals <= threshold]

file_name = 'datahouse.csv'
df_clean.to_csv(file_name, index=False)

print(f"Tổng số dòng: {len(df)}")
print(f"Đã xóa {len(df) - len(df_clean)} dòng outlier.")
print("Lưu ý: Model đã bỏ qua toàn bộ cột chữ (String) khi tính toán.")
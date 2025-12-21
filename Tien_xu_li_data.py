import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. TẢI DỮ LIỆU
df = pd.read_csv('datahouse.csv')

# Xóa cột index thừa nếu có
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# 2. HÀM LÀM SẠCH DỮ LIỆU CƠ BẢN
def clean_numeric_col(df, col, remove_str):
    if col in df.columns:
        # Xóa đơn vị, khoảng trắng và chuyển sang số
        df[col] = df[col].astype(str).str.replace(remove_str, '', regex=False).str.strip()
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# Áp dụng làm sạch cho các cột
df = clean_numeric_col(df, 'Diện tích', ' m²')
df = clean_numeric_col(df, 'Số phòng ngủ', ' phòng')
df = clean_numeric_col(df, 'Dài', ' m')
df = clean_numeric_col(df, 'Rộng', ' m')
df['Số tầng'] = pd.to_numeric(df['Số tầng'], errors='coerce') # Ép kiểu số tầng

# df=df[df['Giá nhà']>1000]
# df['Giá nhà'] = df['Giá nhà'] / 1000

# 4. FEATURE ENGINEERING (TẠO ĐẶC TRƯNG MỚI)


# 5. XỬ LÝ GIÁ TRỊ THIẾU (IMPUTATION)
# Với biến số: Điền bằng trung vị (median)
numeric_cols = ['Diện tích', 'Số tầng', 'Số phòng ngủ', 'Dài']
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())
df['Rộng'] = df['Rộng'].fillna(df['Diện tích']/df['Dài'])

# Với biến phân loại: Điền bằng 'Unknown'
categorical_cols = ['Quận', 'Huyện', 'Loại hình nhà ở', 'Giấy tờ pháp lý']
for col in categorical_cols:
    df[col] = df[col].fillna('Unknown')

# 6. MÃ HÓA (ENCODING) & CHUẨN BỊ DỮ LIỆU CUỐI CÙNG
# Xóa cột Địa chỉ vì quá chi tiết, khó dùng cho model đơn giản
df_model = df.drop(columns=['Địa chỉ'])
#xu li du lieu buoc cuoi cung
df_final = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)
def reverse_ohe(row, prefix):
    cols = [c for c in df.columns if c.startswith(prefix)]
    for c in cols:
        if row[c] == 1:
            return c.replace(prefix, '')
    return 'Khác'
# 3. Tạo cột phân loại để hiển thị (Visual)
if 'Quận' not in df.columns:
    df['Quận'] = df.apply(lambda x: reverse_ohe(x, 'Quận_'), axis=1)

if 'Loại nhà' not in df.columns:
    df['Loại nhà'] = df.apply(lambda x: reverse_ohe(x, 'Loại hình nhà ở_'), axis=1)

# One-Hot Encoding cho các biến phân loại

df_final.to_csv('processed_housing_data.csv', index=False)


# 7. KIỂM TRA KẾT QUẢ
print("Kích thước dữ liệu sau xử lý:", df_final.shape)
print("Các cột dữ liệu (5 dòng đầu):")
print(df_final.head())

# Tách ra X (features) và y (target) để sẵn sàng đưa vào model
X = df_final.drop(columns=['Giá nhà'])
y = df_final['Giá nhà']

# Bạn có thể lưu lại file đã xử lý nếu cần

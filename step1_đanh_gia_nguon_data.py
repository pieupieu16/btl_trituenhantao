import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#### đây là bước để xác định xem với nguồn dữ liệu này thì mình cần xét với những mô hình nào
# Mô hình: Vì dữ liệu sau Log khá đẹp (phân phối chuẩn),
# ta có thể bắt đầu thử nghiệm ngay với Linear Regression (Hồi quy tuyến tính) làm mốc chuẩn (baseline),
# sau đó thử Random Forest hoặc XGBoost để bắt các mối quan hệ phi tuyến tính phức tạp hơn.

df = pd.read_csv('processed_housing_data.csv')
print("--- BẮT ĐẦU PHÂN TÍCH DỮ LIỆU ---")

# 1. Kiểm tra tổng quan biến mục tiêu 'Giá nhà'
print(df['Giá nhà'].describe())

# 2. Phân tích Biến mục tiêu (Target Analysis)
plt.figure(figsize=(14, 6))

# Biểu đồ 1: Phân phối Giá gốc
plt.subplot(1, 2, 1)
# Giá nhà có thể rất lớn, ta dùng đơn vị Tỷ cho dễ nhìn trên biểu đồ (chia 1000)
sns.histplot(df['Giá nhà'] / 1000, kde=True, bins=50)
plt.title(f"Phân phối Giá nhà (Tỷ VNĐ)\nSkew: {df['Giá nhà'].skew():.2f}")
plt.xlabel('Giá nhà (Tỷ)')

# Biểu đồ 2: Áp dụng Log Transform
# Logarit giúp giảm độ lệch (skewness) của dữ liệu giá nhà
df['log_gia_nha'] = np.log1p(df['Giá nhà'])

plt.subplot(1, 2, 2)
sns.histplot(df['log_gia_nha'], kde=True, color='green', bins=50)
plt.title(f"Phân phối sau Log Transform\nSkew: {df['log_gia_nha'].skew():.2f}")
plt.xlabel('Log(Giá nhà)')

plt.tight_layout()
plt.show()

# 3. Phân tích Tương quan (Correlation Heatmap)
plt.figure(figsize=(10, 8))

# CHÚ Ý: Chỉ chọn các cột số quan trọng để vẽ cho rõ (tránh vẽ hàng trăm cột One-hot encoding)
cols_to_analyze = ['Diện tích', 'Số tầng', 'Số phòng ngủ', 'Dài', 'Rộng', 'Giá nhà', 'log_gia_nha']

# Lọc ra các cột thực sự tồn tại trong df (đề phòng trường hợp cột bị thiếu)
existing_cols = [col for col in cols_to_analyze if col in df.columns]

corr_matrix = df[existing_cols].corr()

# In ra top tương quan với giá nhà
print("\nĐộ tương quan với Giá nhà:")
print(corr_matrix['Giá nhà'].sort_values(ascending=False))

# Vẽ Heatmap
mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # Tạo mask để che nửa trên biểu đồ cho đỡ rối
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='RdBu_r', center=0, square=True)
plt.title("Ma trận tương quan các chỉ số chính")
plt.show()

# 4. (Tùy chọn) Vẽ Scatter plot cho 2 yếu tố quan trọng nhất: Diện tích vs Giá
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Diện tích', y='Giá nhà', alpha=0.5)
plt.title('Biểu đồ phân tán: Diện tích vs Giá nhà')
plt.xlabel('Diện tích (m2)')
plt.ylabel('Giá nhà (Triệu)')
# Giới hạn trục để loại bỏ outliers quá xa nếu cần
plt.xlim(0, 500)
plt.ylim(0, 50000)
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

HouseDF = pd.read_csv('processed_housing_data.csv')

# Xác định X (Features) và y (Target)
X = HouseDF.drop('Giá nhà', axis=1)
y = HouseDF['Giá nhà']

# Chia dữ liệu: 80% huấn luyện (train), 20% kiểm tra (test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu (Quan trọng cho Lasso và Ridge)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Khởi tạo danh sách các mô hình
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Lasso Regression": Lasso(alpha=0.1,max_iter=10000), # Alpha là tham số điều chuẩn
    "Ridge Regression": Ridge(alpha=1.0)
}

# 3. Huấn luyện và Đánh giá
results = []

for name, model in models.items():
    # Sử dụng dữ liệu đã chuẩn hóa cho các mô hình tuyến tính, dữ liệu gốc cho cây quyết định (tùy chọn)
    # Để đơn giản, ta dùng dữ liệu chuẩn hóa cho tất cả (không ảnh hưởng xấu đến cây quyết định)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # Tính toán các chỉ số
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse) # RMSE là căn bậc hai của MSE
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Lưu kết quả
    results.append({
        "Algorithm": name,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2 Score": r2
    })

# 4. Hiển thị kết quả dưới dạng DataFrame
df_results = pd.DataFrame(results)

# Sắp xếp theo R2 Score giảm dần (Mô hình tốt nhất lên đầu)
df_results = df_results.sort_values(by="R2 Score", ascending=False)

print("--- BẢNG SO SÁNH HIỆU SUẤT MÔ HÌNH ---")
print(df_results.to_string(index=False))

# 5. Vẽ biểu đồ so sánh R2 Score
plt.figure(figsize=(10, 6))
plt.barh(df_results["Algorithm"], df_results["R2 Score"], color='skyblue')
plt.xlabel("R2 Score (Càng gần 1 càng tốt)")
plt.title("So sánh độ chính xác (R2 Score) của các thuật toán")
plt.xlim(0, 1)
plt.gca().invert_yaxis() # Đảo ngược trục Y để mô hình tốt nhất nằm trên cùng
plt.show()
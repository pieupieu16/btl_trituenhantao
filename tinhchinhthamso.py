import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
import time

# --- 1. Load và Chuẩn bị dữ liệu ---
HouseDF = pd.read_csv('processed_housing_data.csv')

X = HouseDF.drop('Giá nhà', axis=1)
y = HouseDF['Giá nhà']

# Chia tập dữ liệu thành Train (80%) và Test (20%)
# GridSearch sẽ tự động chia Train thành Train/Validation nội bộ (Cross-Validation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# --- 2. Thiết lập Lưới tham số (Parameter Grid) ---
# Đây là nơi bạn định nghĩa "tất cả" các tham số muốn thử.
# Cảnh báo: Càng nhiều tham số, thời gian chạy càng lâu (cấp số nhân).
param_grid = {
    # Số lượng cây (càng nhiều càng tốt nhưng chậm)
    'n_estimators': [100, 200, 500],

    # Tốc độ học (thường đi kèm n_estimators, thấp thì cần nhiều cây)
    'learning_rate': [0.01, 0.1, 0.2],

    # Độ sâu tối đa của mỗi cây (kiểm soát độ phức tạp, tránh overfitting)
    'max_depth': [3, 4, 5],

    # Số lượng mẫu tối thiểu để tách nút
    'min_samples_split': [2, 5],

    # Tỷ lệ mẫu dùng để huấn luyện từng cây (Stochastic Gradient Boosting)
    'subsample': [0.8, 1.0]
}

print(f"Đang bắt đầu tìm kiếm lưới (Grid Search)...")
print(
    f"Tổng số tổ hợp sẽ chạy: {3 * 3 * 3 * 2 * 2} tổ hợp x 3 lần kiểm chéo (Cross-Validation) = {3 * 3 * 3 * 2 * 2 * 3} lần huấn luyện.")
start_time = time.time()

# --- 3. Khởi tạo mô hình và GridSearch ---
gbr = GradientBoostingRegressor(random_state=101)

# cv=3: Chia tập train thành 3 phần để kiểm tra chéo (K-Fold Cross Validation)
# n_jobs=-1: Sử dụng tất cả các nhân CPU để chạy nhanh hơn
# verbose=2: Hiển thị chi tiết quá trình chạy
grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid,
                           cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

# Bắt đầu chạy (Fit)
grid_search.fit(X_train, y_train)

end_time = time.time()
print(f"\nHoàn thành trong: {(end_time - start_time) / 60:.2f} phút")

# --- 4. Kết quả tốt nhất ---
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("\n--- BỘ THAM SỐ TỐT NHẤT TÌM ĐƯỢC ---")
print(best_params)

# --- 5. Đánh giá mô hình tối ưu trên tập Test ---
# Dùng mô hình tốt nhất vừa tìm được để dự đoán
final_predictions = best_model.predict(X_test)

mae = metrics.mean_absolute_error(y_test, final_predictions)
mse = metrics.mean_squared_error(y_test, final_predictions)
rmse = np.sqrt(mse)

print("\n--- Kết quả đánh giá trên tập Test (Mô hình tối ưu) ---")
print(f'MAE: {mae:.2f}')
print(f'MSE: {mse:.2f}')
print(f'RMSE: {rmse:.2f}')

# --- 6. Lưu kết quả tham số tốt nhất ra file text (để dùng sau này) ---
with open("best_params_gradient_boosting.txt", "w") as f:
    f.write(str(best_params))
print("\nĐã lưu bộ tham số tốt nhất vào file 'best_params_gradient_boosting.txt'")

# --- 7. Trực quan hóa ---
plt.figure(figsize=(10, 6))
plt.scatter(y_test, final_predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Mô hình tối ưu (Grid Search): Thực tế vs Dự đoán')
plt.xlabel('Giá thực tế')
plt.ylabel('Giá dự đoán')
plt.show()
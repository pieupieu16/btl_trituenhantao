import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- 1. CHUẨN BỊ DỮ LIỆU ---
print(">>> Đang tải dữ liệu đã xử lý...")

# Đọc file dữ liệu (Hãy đảm bảo tên file đúng là file bạn đang có)
try:
    df = pd.read_csv('processed_housing_data.csv')
except FileNotFoundError:
    # Nếu không tìm thấy, thử đọc file raw rồi báo lỗi nếu vẫn không được
    try:
        df = pd.read_csv('processed_housing_data.csvv')
        print("Cảnh báo: Đang đọc file gốc processed_housing_data.csv thay vì file đã xử lý.")
    except:
        print("LỖI: Không tìm thấy file dữ liệu. Hãy kiểm tra tên file.")
        exit()

print(f"Dữ liệu đầu vào: {df.shape[0]} dòng, {df.shape[1]} cột")

# Xác định biến mục tiêu và biến đầu vào
# Dựa trên danh sách cột bạn cung cấp, cột mục tiêu là 'Giá nhà'
target_col = 'Giá nhà'

if target_col not in df.columns:
    print(f"LỖI: Không tìm thấy cột mục tiêu '{target_col}' trong file.")
    exit()

# Tách X và y
# Lưu ý: Cần loại bỏ các cột không dùng để train nếu có (ví dụ ngày tháng gốc nếu chưa xóa)
# Ở đây ta lấy tất cả các cột trừ 'Giá nhà' làm đầu vào
X = df.drop(columns=[target_col])

# Log Transform cho biến mục tiêu (để phân phối chuẩn hơn)
y = np.log1p(df[target_col])

# --- 2. PHÂN LOẠI CỘT ĐỂ XỬ LÝ ---
# Vì dữ liệu đã One-Hot (nhiều cột 0/1), ta chỉ cần Scale các cột số thực
# Các cột 0/1 (Binary) thì giữ nguyên

# Danh sách các cột số thực cần chuẩn hóa (Scale)
# Dựa trên tên cột bạn đưa:
numeric_cols = ['Diện tích', 'Số tầng', 'Số phòng ngủ', 'Dài', 'Rộng', 'Năm', 'Tháng']
# Lọc lại để chắc chắn các cột này có trong X
numeric_cols = [col for col in numeric_cols if col in X.columns]

# Các cột còn lại là Binary (đã one-hot), ta giữ nguyên
binary_cols = [col for col in X.columns if col not in numeric_cols]

# Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. THIẾT LẬP PIPELINE ---
# Chỉ cần Scaler cho cột số, không cần OneHotEncoder nữa
preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), numeric_cols),      # Chuẩn hóa số liệu (diện tích, tầng...)
        ('cat', 'passthrough', binary_cols)         # Giữ nguyên các cột 0/1 (Quận_..., Huyện_...)
    ]
)

# Pipeline: Xử lý -> Mô hình
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=42))
])

# --- 4. TỐI ƯU HÓA (TUNING) ---
print(">>> Đang chạy tối ưu hóa mô hình (RandomizedSearchCV)...")

# Không gian tham số (Giữ nguyên chiến lược cũ)
param_dist = {
    'regressor__n_estimators': [300, 500, 800],
    'regressor__learning_rate': [0.05, 0.1, 0.2],
    'regressor__max_depth': [3, 5, 7],
    'regressor__min_samples_split': [5, 10],
    'regressor__subsample': [0.8, 0.9]
}

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=10,       # Số lần thử
    cv=3,            # Cross-validation
    scoring='r2',    # Tối ưu theo R2 Score
    n_jobs=-1,
    random_state=42,
    verbose=1
)

search.fit(X_train, y_train)
best_model = search.best_estimator_

print("\n" + "="*40)
print("THAM SỐ TỐT NHẤT:")
for param, value in search.best_params_.items():
    print(f" - {param}: {value}")
print("="*40)

# --- 5. ĐÁNH GIÁ KẾT QUẢ ---
y_pred_log = best_model.predict(X_test)
# Chuyển ngược từ Log về tiền thật
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print(f"\nKẾT QUẢ CUỐI CÙNG:")
print(f"RMSE: {rmse:,.2f} Triệu VNĐ")
print(f"MAE:  {mae:,.2f} Triệu VNĐ")
print(f"R2 Score: {r2:.4f}")

# --- 6. VẼ FEATURE IMPORTANCE ---
# Lấy danh sách tên cột sau khi qua pipeline
# Vì ta dùng passthrough cho binary_cols, thứ tự sẽ là [numeric_cols, binary_cols]
feature_names = numeric_cols + binary_cols

importances = best_model.named_steps['regressor'].feature_importances_

# Tạo DataFrame
fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
fi_df = fi_df.sort_values(by='Importance', ascending=False).head(20)

plt.figure(figsize=(12, 10))
sns.barplot(x='Importance', y='Feature', data=fi_df, hue='Feature', legend=False, palette='viridis')
plt.title('TOP 20 YẾU TỐ ẢNH HƯỞNG ĐẾN GIÁ NHÀ', fontsize=15)
plt.xlabel('Mức độ quan trọng')
plt.tight_layout()
plt.show()
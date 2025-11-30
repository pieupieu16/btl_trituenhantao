import pandas as pd
import numpy as np
import joblib  # Thư viện để lưu/tải model
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold # ĐÃ THÊM KFold
from sklearn.impute import SimpleImputer

# --- ĐỊNH NGHĨA THÀNH PHẦN THIẾU ---
# Định nghĩa KFold cho Cross-Validation (ví dụ: 5 folds)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# ------------------------------------


# --- 1. HUẤN LUYỆN MÔ HÌNH LẦN CUỐI (TRÊN TOÀN BỘ DỮ LIỆU) ---
print(">>> Đang tải và xử lý toàn bộ dữ liệu...")

# Đọc dữ liệu (Vui lòng đảm bảo file 'processed_housing_data.csv' đã tồn tại)
try:
    df = pd.read_csv('processed_housing_data.csv')
except:
    df = pd.read_csv('processed_housing_data.csv') # Lỗi này vẫn có thể xảy ra nếu file không tồn tại!


target_col = 'Giá nhà'
# Bỏ cột mục tiêu và cột giá/m2 (nếu có)
X = df.drop(columns=[target_col, 'Giá/m2'], errors='ignore')
y = np.log1p(df[target_col])

# Xác định cột số và cột chữ (Đã bao gồm các cột One-Hot đã được xử lý)
numeric_cols = ['Diện tích', 'Số tầng', 'Số phòng ngủ', 'Dài', 'Rộng']
# Lọc chỉ các cột thực sự có trong X
numeric_cols = [c for c in numeric_cols if c in X.columns]
binary_cols = [c for c in X.columns if c not in numeric_cols]

# Pipeline Xử lý (preprocessor)
# Giả định: Các cột số cần RobustScaling, các cột One-Hot (binary_cols) giữ nguyên.
# Lưu ý: Các bước Imputation nên được thực hiện TRƯỚC BƯỚC NÀY nếu cần thiết,
# hoặc thêm Imputer vào bước 'num' nếu dữ liệu đầu vào vẫn còn NaN.
preprocessor = ColumnTransformer(
    transformers=[
        # Chỉ áp dụng RobustScaler cho các cột số
        ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                ('scaler', RobustScaler())]), numeric_cols),
        # Giữ nguyên các cột nhị phân (One-Hot)
        ('cat', 'passthrough', binary_cols)
    ],
    remainder='drop'
)


final_model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=7,
    min_samples_split=10,
    subsample=0.9,
    random_state=42
)

# Xây dựng Pipeline cuối cùng
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor), # Xử lý, chuẩn hóa dữ liệu
    ('model', final_model)          # Huấn luyện mô hình
])

print(">>> Đang huấn luyện mô hình cuối cùng (Final Training)...")
# Huấn luyện mô hình trên toàn bộ tập dữ liệu (X, y)
pipeline.fit(X, y)

# Đánh giá bằng Cross-Validation (Kiểm định chéo)
print(">>> Đang đánh giá hiệu suất mô hình bằng Cross-Validation...")
cv_results = cross_val_score(
    pipeline,
    X,
    y,
    cv=kf, # Sử dụng KFold đã định nghĩa
    scoring='neg_root_mean_squared_error',
    n_jobs=-1 # Tăng tốc độ tính toán CV
)

# Đổi dấu để lấy RMSE dương
rmse_mean = -cv_results.mean()
rmse_std = cv_results.std()

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score

# Giả định: 'pipeline', 'X', 'y', và 'kf' (KFold) đã được định nghĩa và huấn luyện trước đó.

print("\n>>> Đang tính toán các chỉ số độ chính xác khác...")

# --- 1. ĐỊNH NGHĨA CUSTOM SCORERS (Tính toán trên thang Log) ---
# Cross-validation mặc định tính trên thang giá trị mà mô hình dự đoán (Log scale)

# Scorer cho RMSE (đã có)
# Scorer cho MAE
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
# Scorer cho R2
r2_scorer = make_scorer(r2_score)


# --- 2. TÍNH TOÁN CROSS-VALIDATION CHO TẤT CẢ CÁC CHỈ SỐ ---

# 2.1. Tính RMSE (Negative RMSE)
cv_rmse = cross_val_score(
    pipeline, X, y, cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1
)
rmse_mean = -cv_rmse.mean()
rmse_std = cv_rmse.std()

# 2.2. Tính MAE (Negative MAE)
cv_mae = cross_val_score(
    pipeline, X, y, cv=kf, scoring=mae_scorer, n_jobs=-1
)
mae_mean = -cv_mae.mean()
mae_std = cv_mae.std()

# 2.3. Tính R2 Score
cv_r2 = cross_val_score(
    pipeline, X, y, cv=kf, scoring=r2_scorer, n_jobs=-1
)
r2_mean = cv_r2.mean()
r2_std = cv_r2.std()


# --- 3. IN KẾT QUẢ ĐẦY ĐỦ ---

print("\n--- Kết quả Đánh giá Độ Chính xác Đầy đủ (Log Scale) ---")
print("Chỉ số | RMSE TB | MAE TB | R2 Score")
print("-------------------------------------------------------")
print(f"Mean  | {rmse_mean:.4f}  | {mae_mean:.4f} | {r2_mean:.4f}")
print(f"Std   | {rmse_std:.4f}  | {mae_std:.4f} | {r2_std:.4f}")

print("\n--- Diễn giải (Giới thiệu) ---")
print(f"* MAE (Mean Absolute Error): {mae_mean:.4f}")
print("  > Sai số tuyệt đối trung bình (ít nhạy cảm với outliers hơn RMSE).")
print(f"* R2 Score: {r2_mean:.4f}")
print("  > Cho biết tỷ lệ phần trăm phương sai của biến mục tiêu được giải thích bởi mô hình (gần 1 là tốt nhất).")


# # --- 2. LƯU MÔ HÌNH RA FILE ---
# model_filename = 'house_price_model.pkl'
# joblib.dump(pipeline, model_filename)
# # Lưu thêm danh sách cột để lúc dự đoán biết đường tạo DataFrame
# joblib.dump(X.columns.tolist(), 'model_columns.pkl')
#
# print(f">>> Đã lưu mô hình vào '{model_filename}' thành công!")
#
#
# # --- 3. HÀM DỰ ĐOÁN (DÙNG ĐỂ CHẠY THỬ) ---
# def du_doan_gia_nha(dien_tich, so_tang, so_phong, dai, rong, quan_huyen_dict):
#     """
#     Hàm này nhận thông tin nhà và trả về giá dự đoán
#     quan_huyen_dict: Là dictionary chứa thông tin One-Hot (VD: {'Quận_Cầu Giấy': 1, ...})
#     """
#     # 1. Tải thông tin cấu trúc cột
#     model_cols = joblib.load('model_columns.pkl')
#
#     # 2. Tạo DataFrame rỗng đúng chuẩn
#     input_data = pd.DataFrame(0, index=[0], columns=model_cols)
#
#     # 3. Điền thông tin người dùng nhập
#     input_data['Diện tích'] = dien_tich
#     input_data['Số tầng'] = so_tang
#     input_data['Số phòng ngủ'] = so_phong
#     input_data['Dài'] = dai
#     input_data['Rộng'] = rong
#
#     # Điền thông tin Quận/Huyện (Set = 1 nếu tìm thấy cột tương ứng)
#     for col_name in model_cols:
#         # Nếu tên cột trùng với thông tin quận huyện user nhập
#         if col_name in quan_huyen_dict and quan_huyen_dict[col_name] == 1:
#             input_data[col_name] = 1
#
#     # 4. Tải model và dự đoán
#     loaded_model = joblib.load(model_filename)
#     pred_log = loaded_model.predict(input_data)[0]
#     price = np.expm1(pred_log)  # Chuyển log về tiền thật
#
#     return price
#
#
# # --- 4. CHẠY THỬ DỰ ĐOÁN ---
# print("\n" + "=" * 30)
# print("DEMO DỰ ĐOÁN GIÁ NHÀ")
# print("=" * 30)

# Ví dụ: Nhà 50m2, 4 tầng, ở Quận Cầu Giấy (Giả sử data có cột 'Quận_Quận Cầu Giấy')
# # Bạn cần mở file data lên xem tên cột chính xác là gì nhé (VD: 'Quận_Quận Cầu Giấy' hay 'Quận_Cầu Giấy')
# sample_house_input = {
#     'Quận_Quận Cầu Giấy': 1,  # Thay đổi tên này cho đúng với file cleaned của bạn
#     # Các quận khác tự động là 0
# }
#
# # Dự đoán thử
# gia_du_doan = du_doan_gia_nha(
#     dien_tich=50,
#     so_tang=4,
#     so_phong=4,
#     dai=10,
#     rong=5,
#     quan_huyen_dict=sample_house_input
# )
#
# print(f"Căn nhà ví dụ (50m2, 4 tầng, Cầu Giấy):")
# print(f">>> Giá dự đoán: {gia_du_doan:,.2f} Triệu VNĐ (~ {gia_du_doan / 1000:.2f} Tỷ)")
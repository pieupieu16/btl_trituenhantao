import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import tree_core # Import module C++ bạn vừa tạo!
import time

# 1. Đọc dữ liệu
print("Đang đọc dữ liệu...")

df = pd.read_csv('processed_housing_data.csv')
df_numeric = pd.get_dummies(df, drop_first=True)
target_col = 'Giá nhà'
if target_col not in df_numeric.columns: target_col = df_numeric.columns[-1]

X = df_numeric.drop(target_col, axis=1).values.astype(np.float64) # C++ cần float64
y = df_numeric[target_col].values.astype(np.float64)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Kích thước Train: {X_train.shape}")

# 2. Huấn luyện bằng C++
print("\n🚀 Bắt đầu huấn luyện mô hình (C++ Core)...")
start_time = time.time()

# Khởi tạo cây từ C++ (Nhanh hơn Python thuần rất nhiều)
# Cấu hình mạnh: Depth=20, Min Split=2
model = tree_core.DecisionTreeRegressor(2, 20) 
model.fit(X_train, y_train)

end_time = time.time()
print(f"✅ Hoàn thành trong {end_time - start_time:.4f} giây!")

# 3. Đánh giá
y_pred = model.predict(X_test)
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
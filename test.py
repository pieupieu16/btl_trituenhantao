import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

file_name = "processed_housing_data.csv"
df = pd.read_csv(file_name)
# (Xử lý sơ bộ: boolean -> int, fillna...)
for col in df.select_dtypes(include=['bool']).columns:
    df[col] = df[col].astype(int)
df = df.fillna(df.median())

X = df.drop(columns=['Giá nhà'])
Y = df['Giá nhà']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model_final = GradientBoostingRegressor(n_estimators=200, random_state=42)
model_final.fit(X_train, Y_train)

# Đánh giá
r2 = r2_score(Y_test, model_final.predict(X_test))
print(f"Độ chính xác R-squared mới: {r2:.2%}")
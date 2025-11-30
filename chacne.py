import pandas as pd

# Đọc file CSV hiện tại
df = pd.read_csv('augmented_housing_data.csv')

# Lưu sang Excel
# index=False để không lưu cột số thứ tự (0, 1, 2...) thừa thãi
df.to_excel('ket_qua_nha.xlsx', index=False, sheet_name='Du_Lieu_Sach')

print("Đã chuyển sang file 'ket_qua_nha.xlsx' thành công!")
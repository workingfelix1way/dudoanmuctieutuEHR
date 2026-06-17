import pandas as pd
import numpy as np

print("--- 🔍 BẮT ĐẦU KHÁM PHÁ DỮ LIỆU EHR THỰC TẾ ---")

df = pd.read_csv("diabetic_data.csv")

n_rows, n_cols = df.shape
print(f"📌 Tổng số ca bệnh (Dòng): {n_rows:,}")
print(f"📌 Tổng số thông tin lâm sàng (Cột): {n_cols}")

print("\n--- 📋 ĐÁNH GIÁ CÁC MỤC TIÊU DỰ ĐOÁN (TARGETS) ---")

print("\n1. Phân phối số ngày nằm viện (Length of Stay):")
print(df['time_in_hospital'].describe())

print("\n2. Tỷ lệ các nhãn trong cột Tái nhập viện (Readmitted):")
print(df['readmitted'].value_counts(normalize=True) * 100)

print("\n--- ⚠️ KIỂM TRA DỮ LIỆU KHUYẾT THIẾU (MISSING VALUES) ---")
for col in df.columns:
    missing_count = (df[col] == '?').sum()
    if missing_count > 0:
        percentage = (missing_count / n_rows) * 100
        print(f"• Cột [{col}] bị thiếu: {missing_count:,} dòng ({percentage:.2f}%)")

print("\n--- 🧠 GỢI Ý THIẾT KẾ BÀI TOÁN MULTI-TASK ---")
print("Đầu vào (Features): Tuổi (age), Giới tính (gender), Số lượng thuốc (num_medications), Số phòng khám trước đó (number_emergency), v.v.")
print("Đầu ra 1 (Classification): Dự đoán tái nhập viện trong <30 ngày (Cột 'readmitted')")
print("Đầu ra 2 (Regression): Dự đoán số ngày nằm viện thực tế (Cột 'time_in_hospital')")
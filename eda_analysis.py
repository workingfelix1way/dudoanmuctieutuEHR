import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Đọc dữ liệu (Đảm bảo file diabetic_data.csv nằm cùng thư mục)
# Thay đổi đường dẫn nếu cần thiết
df = pd.read_csv('diabetic_data.csv')

# Thiết lập style chuẩn học thuật cho biểu đồ
sns.set_theme(style="whitegrid")

# ==========================================
# BIỂU ĐỒ 1: PIE CHART - Phân phối Tái nhập viện
# ==========================================
plt.figure(figsize=(8, 8))

# Ánh xạ nhãn nhị phân giống như quy trình tiền xử lý trong đồ án của bạn:
# "<30" là Nhóm Rủi ro cao, còn lại là Nhóm An toàn
df['readmit_binary'] = df['readmitted'].apply(
    lambda x: 'Tái nhập viện <30 ngày\n(Rủi ro cao)' if x == '<30' else 'Không / Tái nhập viện muộn\n(An toàn)'
)

# Đếm số lượng mỗi nhóm
readmit_counts = df['readmit_binary'].value_counts()

# Thiết lập màu sắc và hiệu ứng (Cắt miếng rủi ro ra cho nổi bật)
colors = ['#4CAF50', '#F44336'] # Xanh lá cho an toàn, Đỏ cho rủi ro
explode = (0, 0.1)  

# Vẽ biểu đồ
plt.pie(readmit_counts, labels=readmit_counts.index, colors=colors, 
        autopct='%1.1f%%', startangle=140, explode=explode, 
        shadow=True, textprops={'fontsize': 14, 'fontweight': 'bold'})

plt.title('Phân phối mất cân bằng của biến mục tiêu Tái nhập viện', 
          fontsize=16, fontweight='bold', pad=20)

# Tự động lưu ảnh chất lượng cao để chèn vào Word
plt.savefig('eda_pie_chart.png', dpi=300, bbox_inches='tight')
plt.show()

# ==========================================
# BIỂU ĐỒ 2: HEATMAP - Ma trận tương quan (Pearson Correlation)
# ==========================================
plt.figure(figsize=(12, 9))

# Trích xuất các biến số học cốt lõi cần phân tích
numeric_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 
                'num_medications', 'number_outpatient', 'number_emergency', 
                'number_inpatient', 'number_diagnoses']

# Tính toán ma trận tương quan
corr_matrix = df[numeric_cols].corr()

# Đổi tên cột sang tiếng Việt để biểu đồ trông chỉn chu khi đưa vào đồ án
vietnamese_labels = ['Số ngày nằm viện', 'Số xét nghiệm', 'Số thủ thuật', 
                     'Số lượng thuốc', 'Khám ngoại trú', 'Khám cấp cứu', 
                     'Nhập viện nội trú', 'Số chẩn đoán']

# Vẽ Heatmap
ax = sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', fmt=".2f", 
                 xticklabels=vietnamese_labels, yticklabels=vietnamese_labels,
                 linewidths=.5, vmin=-0.2, vmax=1.0, 
                 cbar_kws={"shrink": .8, 'label': 'Hệ số tương quan Pearson'})

plt.title('Ma trận tương quan giữa các đặc trưng lâm sàng số học', 
          fontsize=16, fontweight='bold', pad=20)

# Chỉnh font chữ trục tọa độ
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)

# Tự động lưu ảnh chất lượng cao
plt.savefig('eda_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
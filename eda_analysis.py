import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('diabetic_data.csv')

sns.set_theme(style="whitegrid")

plt.figure(figsize=(8, 8))

df['readmit_binary'] = df['readmitted'].apply(
    lambda x: 'Tái nhập viện <30 ngày\n(Rủi ro cao)' if x == '<30' else 'Không / Tái nhập viện muộn\n(An toàn)'
)

readmit_counts = df['readmit_binary'].value_counts()

colors = ['#4CAF50', '#F44336']
explode = (0, 0.1)  

plt.pie(readmit_counts, labels=readmit_counts.index, colors=colors, 
        autopct='%1.1f%%', startangle=140, explode=explode, 
        shadow=True, textprops={'fontsize': 14, 'fontweight': 'bold'})

plt.title('Phân phối mất cân bằng của biến mục tiêu Tái nhập viện', 
          fontsize=16, fontweight='bold', pad=20)

plt.savefig('eda_pie_chart.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(12, 9))

numeric_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 
                'num_medications', 'number_outpatient', 'number_emergency', 
                'number_inpatient', 'number_diagnoses']

corr_matrix = df[numeric_cols].corr()

vietnamese_labels = ['Số ngày nằm viện', 'Số xét nghiệm', 'Số thủ thuật', 
                     'Số lượng thuốc', 'Khám ngoại trú', 'Khám cấp cứu', 
                     'Nhập viện nội trú', 'Số chẩn đoán']

ax = sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', fmt=".2f", 
                 xticklabels=vietnamese_labels, yticklabels=vietnamese_labels,
                 linewidths=.5, vmin=-0.2, vmax=1.0, 
                 cbar_kws={"shrink": .8, 'label': 'Hệ số tương quan Pearson'})

plt.title('Ma trận tương quan giữa các đặc trưng lâm sàng số học', 
          fontsize=16, fontweight='bold', pad=20)

plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)

plt.savefig('eda_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
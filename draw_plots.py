import torch
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

history_loss = [3.4632, 2.9641, 2.9238, 2.9124, 2.8973, 2.8854, 2.8715, 2.8669, 2.8501, 2.8365]

print("1. 🎨 Đang vẽ biểu đồ Training Loss...")
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), history_loss, marker='o', color='green', linewidth=2)
plt.title('Quá trình hội tụ của mạng Deep Learning (Training Loss)')
plt.xlabel('Số vòng lặp (Epoch)')
plt.ylabel('Giá trị Sai số (Total Loss)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('loss_curve.png', dpi=300)

print("2. ⏳ Đang nạp dữ liệu và mô hình PyTorch...")
df = pd.read_csv("diabetic_data.csv")
df.replace('?', np.nan, inplace=True)
df.drop(columns=['weight', 'encounter_id', 'patient_nbr'], inplace=True, errors='ignore')
df['Target_Readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
df['Target_Length_of_Stay'] = df['time_in_hospital']

artifacts = joblib.load("ehr_pytorch_artifacts.pkl")
feature_cols, scaler, input_dim, encoders = artifacts["feature_cols"], artifacts["scaler"], artifacts["input_dim"], artifacts["label_encoders"]

for col in ['race', 'payer_code', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3']:
    df[col] = df[col].fillna(df[col].mode()[0] if col in ['race', 'diag_1', 'diag_2', 'diag_3'] else 'Missing')

X = df[feature_cols].copy()
for col in X.columns:
    if col in encoders:
        le = encoders[col]
        X[col] = X[col].apply(lambda val: val if val in le.classes_ else le.classes_[0])
        X[col] = le.transform(X[col].astype(str))

_, X_test, _, y_test_cls, _, y_test_reg = train_test_split(
    X.values, df['Target_Readmitted'].values, df['Target_Length_of_Stay'].values, 
    test_size=0.2, random_state=42, stratify=df['Target_Readmitted'].values
)
X_test_scaled = scaler.transform(X_test)

class MultiTaskNet(torch.nn.Module):
    def __init__(self, dim):
        super(MultiTaskNet, self).__init__()
        self.shared_layers = torch.nn.Sequential(
            torch.nn.Linear(dim, 128), torch.nn.BatchNorm1d(128), torch.nn.ReLU(), torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64), torch.nn.BatchNorm1d(64), torch.nn.ReLU(), torch.nn.Dropout(0.2)
        )
        self.class_head = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1), torch.nn.Sigmoid())
        self.reg_head = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    def forward(self, x):
        s = self.shared_layers(x)
        return self.class_head(s), self.reg_head(s)

model = MultiTaskNet(input_dim)
model.load_state_dict(torch.load("ehr_pytorch_model.pth", map_location=torch.device('cpu')))
model.eval()

with torch.no_grad():
    pred_cls_raw, pred_reg_raw = model(torch.tensor(X_test_scaled, dtype=torch.float32))
    pred_cls = pred_cls_raw.numpy().flatten()
    pred_reg = pred_reg_raw.numpy().flatten()

print("3. 🎨 Đang vẽ 3 biểu đồ đánh giá...")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test_cls, pred_cls)
plt.figure(figsize=(8, 6)); plt.plot(fpr, tpr, color='darkorange', label=f'AUC = {auc(fpr, tpr):.3f}')
plt.plot([0, 1], [0, 1], linestyle='--'); plt.title('ROC Curve'); plt.legend(); plt.savefig('roc_curve.png', dpi=300)

# Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test_cls, np.round(pred_cls)), annot=True, fmt='d', cmap='Blues')
plt.title('Ma trận nhầm lẫn'); plt.savefig('confusion_matrix.png', dpi=300)

# Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test_reg[:500], pred_reg[:500], alpha=0.5)
plt.plot([0, 15], [0, 15], color='red', linestyle='--')
plt.title('Dự đoán vs Thực tế (LOS)'); plt.savefig('scatter_plot.png', dpi=300)

print("✅ Đã hoàn thành bộ sưu tập 4 hình ảnh!")
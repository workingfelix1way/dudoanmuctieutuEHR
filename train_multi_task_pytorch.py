import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, mean_absolute_error
import joblib

print("--- 🔥 KHỞI TẠO MẠNG MULTI-TASK LEARNING BẰNG PYTORCH ---")

df = pd.read_csv("diabetic_data.csv")
df.replace('?', np.nan, inplace=True)
df.drop(columns=['weight', 'encounter_id', 'patient_nbr'], inplace=True, errors='ignore')

for col in ['race', 'payer_code', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3']:
    df[col].fillna(df[col].mode()[0] if col in ['race', 'diag_1', 'diag_2', 'diag_3'] else 'Missing', inplace=True)

df['Target_Readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
df['Target_Length_of_Stay'] = df['time_in_hospital']

feature_cols = [
    'race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id',
    'admission_source_id', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'diag_1', 'diag_2', 'diag_3', 'number_diagnoses',
    'max_glu_serum', 'A1Cresult', 'metformin', 'insulin', 'change', 'diabetesMed'
]
X = df[feature_cols].copy()

label_encoders = {}
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

X_train, X_test, y_train_cls, y_test_cls, y_train_reg, y_test_reg = train_test_split(
    X.values, df['Target_Readmitted'].values, df['Target_Length_of_Stay'].values, 
    test_size=0.2, random_state=42, stratify=df['Target_Readmitted'].values
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train_cls, dtype=torch.float32).unsqueeze(1),
    torch.tensor(y_train_reg, dtype=torch.float32).unsqueeze(1)
)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

class MultiTaskNet(nn.Module):
    def __init__(self, input_dim):
        super(MultiTaskNet, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.class_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.reg_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        shared_features = self.shared_layers(x)
        out_cls = self.class_head(shared_features)
        out_reg = self.reg_head(shared_features)
        return out_cls, out_reg

model = MultiTaskNet(X_train.shape[1])
criterion_cls = nn.BCELoss()
criterion_reg = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\n--- 🚀 ĐANG HUẤN LUYỆN MẠNG MULTI-TASK BẰNG PYTORCH (10 EPOCHS) ---")
model.train()
for epoch in range(10):
    total_loss = 0
    for batch_x, batch_y_cls, batch_y_reg in train_loader:
        optimizer.zero_grad()
        pred_cls, pred_reg = model(batch_x)
    
        loss_cls = criterion_cls(pred_cls, batch_y_cls)
        loss_reg = criterion_reg(pred_reg, batch_y_reg)
        loss = (loss_cls * 5.0) + (loss_reg * 0.2)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/10 - Loss tổng hợp: {total_loss/len(train_loader):.4f}")

model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    test_pred_cls, test_pred_reg = model(X_test_tensor)
    
    auc = roc_auc_score(y_test_cls, test_pred_cls.numpy())
    mae = mean_absolute_error(y_test_reg, test_pred_reg.numpy())

print("\n--- 📈 KẾT QUẢ ĐÁNH GIÁ MẠNG PYTORCH DL ---")
print(f"✨ Task 1 (Tái nhập viện) -> ROC-AUC: {auc:.4f}")
print(f"✨ Task 2 (Số ngày nằm viện) -> Sai số (MAE): {mae:.2f} ngày")

torch.save(model.state_dict(), "ehr_pytorch_model.pth")
artifacts = {
    "scaler": scaler,
    "label_encoders": label_encoders,
    "feature_cols": feature_cols,
    "input_dim": X_train.shape[1]
}
joblib.dump(artifacts, "ehr_pytorch_artifacts.pkl")
print("\n💾 Đã lưu trọng số mô hình PyTorch thành công!")
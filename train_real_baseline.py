import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, roc_auc_score, mean_absolute_error, r2_score
import joblib

print("--- ⚙️ BẮT ĐẦU TIỀN XỬ LÝ DỮ LIỆU THỰC TẾ ---")

df = pd.read_csv("diabetic_data.csv")
df.replace('?', np.nan, inplace=True)

cols_to_drop = ['weight', 'encounter_id', 'patient_nbr']
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

df['race'].fillna(df['race'].mode()[0], inplace=True)
df['payer_code'].fillna('Missing', inplace=True)
df['medical_specialty'].fillna('Missing', inplace=True)
df['diag_1'].fillna(df['diag_1'].mode()[0], inplace=True)
df['diag_2'].fillna(df['diag_2'].mode()[0], inplace=True)
df['diag_3'].fillna(df['diag_3'].mode()[0], inplace=True)

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

y_cls = df['Target_Readmitted']
y_reg = df['Target_Length_of_Stay']

X_train, X_test, y_train_cls, y_test_cls, y_train_reg, y_test_reg = train_test_split(
    X, y_cls, y_reg, test_size=0.2, random_state=42, stratify=y_cls
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Tiền xử lý xong! Đang huấn luyện Baseline bằng Random Forest...")

model_cls = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, class_weight='balanced', n_jobs=-1)
model_cls.fit(X_train_scaled, y_train_cls)
preds_cls = model_cls.predict(X_test_scaled)
probs_cls = model_cls.predict_proba(X_test_scaled)[:, 1]

print("\n📊 KẾT QUẢ TASK 1 (DỰ ĐOÁN TÁI NHẬP VIỆN):")
print(f"ROC-AUC Score: {roc_auc_score(y_test_cls, probs_cls):.4f}")
print(classification_report(y_test_cls, preds_cls, target_names=['Không/Sau 30 ngày', 'Trong 30 ngày']))

model_reg = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
model_reg.fit(X_train_scaled, y_train_reg)
preds_reg = model_reg.predict(X_test_scaled)

print("\n📊 KẾT QUẢ TASK 2 (DỰ ĐOÁN SỐ NGÀY NẰM VIỆN):")
print(f"Sai số tuyệt đối trung bình (MAE): {mean_absolute_error(y_test_reg, preds_reg):.2f} ngày")
print(f"Chỉ số R2 Score: {r2_score(y_test_reg, preds_reg):.4f}")

artifacts = {
    "scaler": scaler,
    "label_encoders": label_encoders,
    "model_classification": model_cls,
    "model_regression": model_reg,
    "feature_cols": feature_cols
}
joblib.dump(artifacts, "ehr_real_baseline_models.pkl")
print("\n💾 Đã lưu thành công các mô hình thật vào file 'ehr_real_baseline_models.pkl'")
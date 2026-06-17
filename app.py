import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import os

st.set_page_config(page_title="Hệ thống EHR Đa Mục Tiêu", page_icon="🏥", layout="wide")

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True


def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

st.sidebar.title("⚙️ Cài đặt hiển thị")
st.sidebar.button("🌓 Đổi Giao Diện (Sáng/Tối)", on_click=toggle_theme)

if st.session_state.dark_mode:
    dark_css = """
    <style>
        [data-testid="stAppViewContainer"] { background-color: #0a0a0a; }
        [data-testid="stSidebar"] { background-color: #1a1a1a; border-right: 1px solid #333; }
        [data-testid="stHeader"] { background-color: rgba(10, 10, 10, 0); }
        
        /* Chữ màu sáng */
        .stApp, .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp label, .stApp span, .stApp div[data-testid="stMetricValue"], .stApp div[data-testid="stMetricLabel"] { 
            color: #f5f5f5 !important; 
        }
        
        /* Ô nhập liệu tối màu */
        input, div[data-baseweb="select"] > div, div[data-baseweb="base-input"], div[data-baseweb="input"] {
            background-color: #2b2b2b !important;
            color: #f5f5f5 !important;
            border: 1px solid #444 !important;
        }

        /* Nút + và - trong Number Input */
        div[data-testid="stNumberInput"] button {
            background-color: #2b2b2b !important;
        }
        div[data-testid="stNumberInput"] button svg {
            fill: #f5f5f5 !important;
        }

        /* NÚT ĐỔI GIAO DIỆN TRONG SIDEBAR (CHẾ ĐỘ TỐI) */
        [data-testid="stSidebar"] button {
            background-color: #2b2b2b !important;
            color: #f5f5f5 !important;
            border: 1px solid #444 !important;
        }
    </style>
    """
    st.markdown(dark_css, unsafe_allow_html=True)
else:
    light_css = """
    <style>
        [data-testid="stAppViewContainer"] { background-color: #ffffff; }
        [data-testid="stSidebar"] { background-color: #f0f2f6; border-right: 1px solid #ddd; }
        [data-testid="stHeader"] { background-color: rgba(255, 255, 255, 0); }
        
        /* Chữ màu tối */
        .stApp, .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp label, .stApp span, .stApp div[data-testid="stMetricValue"], .stApp div[data-testid="stMetricLabel"] { 
            color: #333333 !important; 
        }
        
        /* Ô nhập liệu trắng */
        input, div[data-baseweb="select"] > div, div[data-baseweb="base-input"], div[data-baseweb="input"] {
            background-color: #ffffff !important;
            color: #333333 !important;
            border: 1px solid #cccccc !important;
        }

        /* Nút + và - trong Number Input */
        div[data-testid="stNumberInput"] button {
            background-color: #f0f2f6 !important;
        }
        div[data-testid="stNumberInput"] button svg {
            fill: #333333 !important;
        }

        /* NÚT ĐỔI GIAO DIỆN TRONG SIDEBAR (CHẾ ĐỘ SÁNG) */
        [data-testid="stSidebar"] button {
            background-color: #ffffff !important;
            color: #333333 !important;
            border: 1px solid #cccccc !important;
        }
    </style>
    """
    st.markdown(light_css, unsafe_allow_html=True)
    
st.title("🏥 Hệ Thống Dự Đoán Đa Mục Tiêu Từ Bệnh Án Điện Tử (EHR)")
st.markdown("**Created by Felix1way, ThaoNguyen**")
st.write("---")

class MultiTaskNet(nn.Module):
    def __init__(self, input_dim):
        super(MultiTaskNet, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2)
        )
        self.class_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
        self.reg_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        shared_features = self.shared_layers(x)
        return self.class_head(shared_features), self.reg_head(shared_features)

tab1, tab2 = st.tabs(["🚀 DỰ ĐOÁN LÂM SÀNG", "📊 BÁO CÁO HIỆU NĂNG MÔ HÌNH"])

with tab1:
    MODEL_PATH = "ehr_pytorch_model.pth"
    ARTIFACTS_PATH = "ehr_pytorch_artifacts.pkl"

    if not os.path.exists(MODEL_PATH) or not os.path.exists(ARTIFACTS_PATH):
        st.error("❌ KHÔNG TÌM THẤY FILE MÔ HÌNH PYTORCH!")
    else:
        artifacts = joblib.load(ARTIFACTS_PATH)
        scaler, encoders, feature_cols, input_dim = artifacts["scaler"], artifacts["label_encoders"], artifacts["feature_cols"], artifacts["input_dim"]

        @st.cache_resource
        def load_pytorch_model(weights_path, dim):
            net = MultiTaskNet(dim)
            net.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
            net.eval()
            return net

        model = load_pytorch_model(MODEL_PATH, input_dim)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📋 Thông tin hành chính & Tiền sử")
            race = st.selectbox("Chủng tộc (Race):", ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other", "Missing"])
            gender = st.selectbox("Giới tính (Gender):", ["Female", "Male"])
            age = st.selectbox("Nhóm tuổi (Age):", ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"])
            num_outpatient = st.number_input("Số lần khám ngoại trú:", min_value=0, max_value=50, value=0)
            num_emergency = st.number_input("Số lần khám cấp cứu:", min_value=0, max_value=50, value=0)
            num_inpatient = st.number_input("Số lần nhập viện trước đây:", min_value=0, max_value=50, value=0)

        with col2:
            st.subheader("🔬 Chỉ số xét nghiệm & Điều trị")
            num_lab_procedures = st.number_input("Số lượng xét nghiệm:", min_value=1, max_value=150, value=40)
            num_procedures = st.number_input("Số lượng thủ thuật:", min_value=0, max_value=10, value=1)
            num_medications = st.number_input("Số lượng thuốc:", min_value=1, max_value=100, value=15)
            number_diagnoses = st.number_input("Số lượng chẩn đoán:", min_value=1, max_value=20, value=5)
            insulin = st.selectbox("Sử dụng Insulin:", ["No", "Steady", "Up", "Down"])
            metformin = st.selectbox("Sử dụng Metformin:", ["No", "Steady", "Up", "Down"])
            diabetesMed = st.selectbox("Kê đơn thuốc tiểu đường?", ["Yes", "No"])
            change = st.selectbox("Thay đổi liều lượng thuốc?", ["No", "Ch"])

        st.write("---")
        if st.button("🚀 KÍCH HOẠT MẠNG ĐA NHIỆM PHÂN TÍCH", type="primary"):
            input_dict = {
                'race': race, 'gender': gender, 'age': age, 'admission_type_id': 1, 'discharge_disposition_id': 1, 'admission_source_id': 7,
                'num_lab_procedures': num_lab_procedures, 'num_procedures': num_procedures, 'num_medications': num_medications, 
                'number_outpatient': num_outpatient, 'number_emergency': num_emergency, 'number_inpatient': num_inpatient,
                'diag_1': '428', 'diag_2': '250', 'diag_3': '250', 'number_diagnoses': number_diagnoses,
                'max_glu_serum': 'None', 'A1Cresult': 'None', 'metformin': metformin, 'insulin': insulin, 'change': change, 'diabetesMed': diabetesMed
            }
            input_df = pd.DataFrame([input_dict])
            for col in input_df.columns:
                if col in encoders:
                    le = encoders[col]
                    input_df[col] = le.transform(input_df[col].astype(str)) if input_df[col].iloc[0] in le.classes_ else le.transform([le.classes_[0]])
            
            input_scaled = scaler.transform(input_df[feature_cols].values)
            
            with torch.no_grad():
                prob_readmit_tensor, pred_los_tensor = model(torch.tensor(input_scaled, dtype=torch.float32))
                prob_readmit, pred_los = prob_readmit_tensor.item(), pred_los_tensor.item()

            st.header("📊 Kết Quả Đánh Giá Lâm Sàng Đồng Thời:")
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.subheader("1. Nguy cơ tái nhập viện (<30 ngày)")
                st.metric(label="Xác suất rủi ro", value=f"{prob_readmit*100:.1f}%")
                if prob_readmit >= 0.5: st.error("⚠️ RỦI RO CAO")
                else: st.success("✅ RỦI RO THẤP")
            with res_col2:
                st.subheader("2. Thời gian nằm viện dự kiến")
                st.metric(label="Số ngày điều trị", value=f"{pred_los:.1f} ngày")


with tab2:
    st.header("📈 Dashboard Đánh Giá Hiệu Năng (4 Biểu Đồ Toàn Diện)")
    

    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.subheader("1. Quá trình hội tụ (Loss)")
        if os.path.exists("loss_curve.png"):
            st.image("loss_curve.png", caption="Training Loss qua 10 Epochs", use_container_width=True)
        else: st.warning("Thiếu loss_curve.png")

    with row1_col2:
        st.subheader("2. Độ nhạy mô hình (ROC)")
        if os.path.exists("roc_curve.png"):
            st.image("roc_curve.png", caption="Đường cong ROC-AUC", use_container_width=True)
        else: st.warning("Thiếu roc_curve.png")

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        st.subheader("3. Kiểm soát sai số (Confusion)")
        if os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png", caption="Ma trận nhầm lẫn", use_container_width=True)
        else: st.warning("Thiếu confusion_matrix.png")

    with row2_col2:
        st.subheader("4. Độ chính xác số ngày (Scatter)")
        if os.path.exists("scatter_plot.png"):
            st.image("scatter_plot.png", caption="Thực tế vs Dự đoán số ngày", use_container_width=True)
        else: st.warning("Thiếu scatter_plot.png")
            
    
# python -m streamlit run app.py
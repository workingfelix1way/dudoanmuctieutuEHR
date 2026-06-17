import os
import kagglehub
import shutil

os.environ['KAGGLE_API_TOKEN'] = "KGAT_e05f2c473e620d41977cd256aefd2145"

print("1. 🔥 Bắt đầu quá trình kết nối tới Kaggle...")

try:
    print("2. ⏳ Đang tải bộ dữ liệu 'brandao/diabetes' từ Kagglehub (Vui lòng đợi vài giây)...")
    path = kagglehub.dataset_download("brandao/diabetes")
    print(f"3. 🎉 Đã tải xong! Thư mục lưu tạm của hệ thống: {path}")
    
    print("4. 📂 Đang sao chép file .csv vào thư mục Proposal của bạn...")
    files_moved = 0
    for file in os.listdir(path):
        if file.endswith('.csv'):
            shutil.copy(os.path.join(path, file), '.')
            print(f"   -> Đã copy file: {file}")
            files_moved += 1
            
    if files_moved > 0:
        print("\n✅ HOÀN THÀNH XUẤT SẮC! Dữ liệu thật đã sẵn sàng trong thư mục hiện tại.")
    else:
        print("\n⚠️ Cảnh báo: Tải xong nhưng không tìm thấy file .csv nào.")
            
except Exception as e:
    print(f"\n❌ Đã xảy ra lỗi: {e}")
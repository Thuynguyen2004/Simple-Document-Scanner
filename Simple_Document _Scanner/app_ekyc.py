import streamlit as st
import cv2
import numpy as np
import easyocr
import ssl
import pandas as pd
from datetime import datetime
import os
import re

# --- 1. SỬA LỖI SSL ---
ssl._create_default_https_context = ssl._create_unverified_context

# --- 2. CẤU HÌNH TRANG ---
st.set_page_config(page_title="eKYC Pro", page_icon="🆔", layout="wide")
st.title("🆔 Hệ thống eKYC & Quản lý Lịch sử")
st.markdown("---")

HISTORY_FILE = 'ekyc_history.csv'

# --- HÀM XỬ LÝ LỊCH SỬ ---
def load_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame(columns=["Thời gian", "Số CCCD", "Họ và tên", "Ngày sinh"])

def save_to_history(cccd_id, name, dob):
    new_data = {
        "Thời gian": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Số CCCD": cccd_id,
        "Họ và tên": name,
        "Ngày sinh": dob
    }
    df = load_history()
    # Đưa dòng mới lên đầu (để dễ thấy nhất)
    df = pd.concat([pd.DataFrame([new_data]), df], ignore_index=True)
    df.to_csv(HISTORY_FILE, index=False)
    return df

# Hàm xóa lịch sử
def clear_history_file():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)

def delete_last_entry():
    df = load_history()
    if not df.empty:
        # Bỏ dòng đầu tiên (dòng mới nhất)
        df = df.iloc[1:]
        df.to_csv(HISTORY_FILE, index=False)

# --- 3. TẢI MODEL ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['vi', 'en'], gpu=False)

try:
    reader = load_ocr()
except Exception as e:
    st.error(f"Lỗi tải model: {e}")

# --- 4. GIAO DIỆN CHÍNH ---
col_upload, col_display = st.columns([1, 2])

with col_upload:
    st.subheader("1. Tải ảnh lên")
    uploaded_file = st.file_uploader("Chọn ảnh CCCD", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Ảnh gốc", use_container_width=True)

# Nút bấm xử lý
if uploaded_file is not None and col_upload.button("🚀 Phân tích & Lưu", type="primary"):
    
    with col_display:
        st.subheader("2. Kết quả phân tích")
        with st.spinner("Đang trích xuất và lưu dữ liệu..."):
            
            # --- XỬ LÝ CẮT MẶT ---
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            face_found = False
            face_img_display = None
            if len(faces) > 0:
                faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
                (x, y, w, h) = faces[0]
                face_img = image[max(0, y-30):y+h+30, max(0, x-30):x+w+30]
                if face_img.size == 0: face_img = image[y:y+h, x:x+w]
                face_img_display = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                face_found = True

            # --- XỬ LÝ ĐỌC CHỮ ---
            results = reader.readtext(image)
            found_id = "Không rõ"
            found_name = "Không rõ"
            found_dob = "Không rõ"
            black_list = ["CỘNG HÒA", "XÃ HỘI", "VIỆT NAM", "ĐỘC LẬP", "HẠNH PHÚC", 
                          "CĂN CƯỚC", "CÔNG DÂN", "SỐ", "FULL NAME", "DATE OF BIRTH",
                          "QUÊ QUÁN", "THƯỜNG TRÚ", "CỤC TRƯỞNG", "CÓ GIÁ TRỊ"]
            current_year = datetime.now().year 

            for (bbox, text, prob) in results:
                if prob > 0.50 and len(text) > 2:
                    text_upper = text.upper()
                    if text.isdigit() and len(text) == 12: found_id = text
                    date_match = re.search(r'\d{2}/\d{2}/\d{4}', text)
                    if date_match:
                        date_str = date_match.group(0)
                        try:
                            year = int(date_str.split('/')[-1])
                            if 1900 < year < current_year and found_dob == "Không rõ":
                                found_dob = date_str
                        except: pass
                    if text.isupper() and len(text) > 3 and not any(c.isdigit() for c in text):
                        is_clean = True
                        for bad in black_list:
                            if bad in text_upper: is_clean = False; break
                        if is_clean and len(text) > len(found_name) and len(text) < 30:
                            found_name = text

            # --- HIỂN THỊ KẾT QUẢ ---
            st.success("✅ Đã lưu vào hệ thống!")
            st.write("---")
            c1, c2 = st.columns([1, 2])
            with c1:
                if face_found: st.image(face_img_display, width=160, caption="Ảnh chân dung")
                else: st.warning("⚠️ Không cắt được mặt")
            with c2:
                st.info(f"**🔢 Số CCCD:** {found_id}")
                st.success(f"**👤 Họ và tên:** {found_name}")
                st.warning(f"**🎂 Ngày sinh:** {found_dob}")
            
            if found_id != "Không rõ" or found_name != "Không rõ":
                save_to_history(found_id, found_name, found_dob)

# --- 5. HIỂN THỊ BẢNG LỊCH SỬ ---
st.markdown("---")
c_hist, c_btn = st.columns([3, 1])

with c_hist:
    st.subheader("📜 Lịch sử quét gần đây")

with c_btn:
    # Nút xóa nằm bên phải tiêu đề
    if st.button("🗑️ Xóa toàn bộ lịch sử"):
        clear_history_file()
        st.rerun()

df_history = load_history()

if not df_history.empty:
    st.dataframe(df_history, use_container_width=True)
    
    col_dl, col_del_one = st.columns(2)
    with col_dl:
        csv = df_history.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Tải về Excel (CSV)", csv, 'ekyc_history.csv', 'text/csv')
    with col_del_one:
        if st.button("❌ Xóa dòng mới nhất"):
            delete_last_entry()
            st.rerun()
else:
    st.info("Chưa có dữ liệu lịch sử. Hãy thử quét một ảnh!")
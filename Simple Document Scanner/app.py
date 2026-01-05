import streamlit as st
import cv2
import numpy as np
from PIL import Image

# 1. Thiết kế giao diện (Tiêu đề, hướng dẫn)
st.set_page_config(page_title="AI Document Scanner", page_icon="📄")
st.title("📄 Ứng dụng Quét Tài Liệu Thông Minh")
st.write("Tải ảnh hóa đơn hoặc tài liệu lên để hệ thống tự động nhận diện.")

# 2. Widget tải ảnh lên
uploaded_file = st.file_uploader("Chọn ảnh từ máy của bạn...", type=['jpg', 'png', 'jpeg'])

# 3. Xử lý khi có ảnh được tải lên
if uploaded_file is not None:
    # Đọc file ảnh từ Streamlit và chuyển sang định dạng OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Tạo 2 cột để hiển thị ảnh Trước và Sau
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ảnh gốc")
        # Streamlit dùng màu RGB, OpenCV dùng BGR nên phải chuyển đổi để hiển thị đúng màu
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

    # --- PHẦN XỬ LÝ ẢNH (Giống hệt code cũ) ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 75, 200)
    
    # Tìm contour
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    screenCnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break
            
    # Vẽ kết quả
    if screenCnt is not None:
        # Vẽ đường viền màu xanh lá, độ dày 5 cho dễ nhìn
        cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 5)
        st.success("✅ Đã tìm thấy tài liệu thành công!")
    else:
        st.warning("⚠️ Không tìm thấy khung hình chữ nhật rõ ràng. Đang hiển thị ảnh gốc.")

    with col2:
        st.subheader("Kết quả nhận diện")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
        
    # Thêm tùy chọn xem các bước trung gian (Debug)
    if st.checkbox("Xem ảnh đen trắng (Edges)"):
        st.image(edged, caption="Canny Edge Detection")
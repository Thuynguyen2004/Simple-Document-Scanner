import cv2
import numpy as np

# Đọc ảnh
try:
    image = cv2.imread('document.jpg')
    # Resize để xử lý nhanh hơn
    ratio = image.shape[0] / 600.0
    orig = image.copy()
    image = cv2.resize(image, (600, int(image.shape[0] / ratio)))
except AttributeError:
    print("Lỗi: Không đọc được ảnh. Kiểm tra tên file!")
    exit()

# Xử lý ảnh
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# Tìm contour
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

print(f"Đã tìm thấy {len(contours)} đường bao tiềm năng.")

# Lặp qua các contour để kiểm tra
found = False
for i, c in enumerate(contours):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    
    # In ra số góc máy tính nhìn thấy (Debug)
    print(f" - Hình thứ {i+1} có {len(approx)} góc (Diện tích: {cv2.contourArea(c)})")

    # Nếu có 4 góc thì chọn luôn
    if len(approx) == 4:
        screenCnt = approx
        found = True
        print("=> ĐÃ TÌM THẤY TÀI LIỆU (4 GÓC)!")
        break
    
    # [MỚI] Nếu không tìm thấy 4 góc, ta tạm lấy hình to nhất (dù nhiều góc)
    if i == 0:
        screenCnt = approx # Lưu tạm hình to nhất

# Hiển thị kết quả
# Vẽ đường bao (Màu xanh lá cây)
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)

# Vẽ các điểm góc (Màu đỏ) để bạn dễ nhìn
for point in screenCnt:
    cv2.circle(image, tuple(point[0]), 5, (0, 0, 255), -1)

if not found:
    print("\nCảnh báo: Không tìm thấy hình chữ nhật hoàn hảo.")
    print("Đang hiển thị hình to nhất tìm được (có thể bị méo do giấy hồng).")

cv2.imshow("Ket qua (Nhan phim bat ky de thoat)", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
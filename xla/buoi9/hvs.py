import cv2
import numpy as np

# Chương trình con callback cho trackbar (không cần làm gì cả)
def get_hsv_values(val):
    pass

# Tạo cửa sổ hiển thị ảnh
cv2.namedWindow('image')

# Load ảnh
image = cv2.imread('bien_so_4.png')
if image is None:
    print("Không tìm thấy ảnh!")
    exit()

# Tạo trackbar cho giá trị Hue (H: 0-179)
cv2.createTrackbar('H min', 'image', 78, 255, get_hsv_values)
cv2.createTrackbar('H max', 'image', 255, 255, get_hsv_values)

# Tạo trackbar cho giá trị Saturation (S: 0-255)
cv2.createTrackbar('S min', 'image', 0, 255, get_hsv_values)
cv2.createTrackbar('S max', 'image', 255, 255, get_hsv_values)

# Tạo trackbar cho giá trị Value (V: 0-255)
cv2.createTrackbar('V min', 'image', 0, 255, get_hsv_values)
cv2.createTrackbar('V max', 'image', 255, 255, get_hsv_values)

while True:
    # Chuyển sang ảnh HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Lấy giá trị trackbar
    h_min = cv2.getTrackbarPos('H min', 'image')
    h_max = cv2.getTrackbarPos('H max', 'image')
    s_min = cv2.getTrackbarPos('S min', 'image')
    s_max = cv2.getTrackbarPos('S max', 'image')
    v_min = cv2.getTrackbarPos('V min', 'image')
    v_max = cv2.getTrackbarPos('V max', 'image')

    # Tạo giới hạn dưới và trên (dtype uint8 để tránh lỗi inRange)
    lower_limit = np.array([h_min, s_min, v_min], dtype=np.uint8)
    upper_limit = np.array([h_max, s_max, v_max], dtype=np.uint8)

    # Tạo mask lọc màu
    mask = cv2.inRange(hsv_image, lower_limit, upper_limit)

    # Áp mask lên ảnh gốc
    result = cv2.bitwise_and(image, image, mask=mask)

    # Hiển thị ảnh
    cv2.imshow("Original", image)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)

    # In giá trị HSV ra console
    print(f"H min: {h_min}, H max: {h_max}, S min: {s_min}, S max: {s_max}, V min: {v_min}, V max: {v_max}", end='\r')

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

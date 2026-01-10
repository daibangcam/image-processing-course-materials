import cv2
import numpy as np

# Đọc ảnh gốc
image = cv2.imread("09.png")
if image is None:
    print("Không tìm thấy ảnh 09.png")
    exit()

# Chuyển sang HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Khoảng HSV bạn muốn giữ
h_min, h_max = 12, 255
s_min, s_max = 0, 255
v_min, v_max = 161, 255

lower_bound = np.array([h_min, s_min, v_min])
upper_bound = np.array([h_max, s_max, v_max])

# Tạo mask: pixel trong khoảng HSV
mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

# Áp mask lên ảnh gốc
result = cv2.bitwise_and(image, image, mask=mask)

# Chuyển ảnh sang grayscale
gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

# Lưu ảnh grayscale
cv2.imwrite("output09_gray.png", gray_result)

# Hiển thị kết quả
cv2.imshow("Original 09", image)
cv2.imshow("Gray Output", gray_result)
cv2.waitKey(0)
cv2.destroyAllWindows()

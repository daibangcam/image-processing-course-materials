import cv2
import numpy as np

# Đọc ảnh gốc
image = cv2.imread("09.png")
if image is None:
    print("Không tìm thấy ảnh 09.png")
    exit()

# Chuyển sang HSV và lọc màu
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h_min, h_max = 19, 113
s_min, s_max = 0, 98
v_min, v_max = 137, 255
lower_bound = np.array([h_min, s_min, v_min])
upper_bound = np.array([h_max, s_max, v_max])
mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
filtered = cv2.bitwise_and(image, image, mask=mask)

# Khử nhiễu và chuyển sang grayscale
denoised = cv2.GaussianBlur(filtered, (5, 5), 0)
gray_result = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)

# Callback khi trackbar thay đổi
def update_thresh(_):
    white_thresh = cv2.getTrackbarPos("White Threshold", "White/Black")
    black_thresh = cv2.getTrackbarPos("Black Threshold", "White/Black")

    # Tạo mask trắng
    white_mask = cv2.threshold(gray_result, white_thresh, 255, cv2.THRESH_BINARY)[1]
    # Tạo mask đen
    black_mask = cv2.threshold(gray_result, black_thresh, 255, cv2.THRESH_BINARY_INV)[1]

    # Kết hợp hai mask
    combined = cv2.bitwise_or(white_mask, black_mask)

    cv2.imshow("White/Black", combined)

# Tạo cửa sổ và trackbar
cv2.imshow("Original", image)
cv2.imshow("Gray", gray_result)
cv2.namedWindow("White/Black")
cv2.createTrackbar("White Threshold", "White/Black", 180, 255, update_thresh)
cv2.createTrackbar("Black Threshold", "White/Black", 50, 255, update_thresh)

# Khởi tạo hiển thị lần đầu
update_thresh(0)

cv2.waitKey(0)
cv2.destroyAllWindows()

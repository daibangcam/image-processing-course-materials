# import thư viện sử dụng
import cv2
import numpy as np

# tạo chương trình con để lấy giá trị H, S, V từ trackbar
def get_hsv_values(val):
    pass

# tạo 1 cửa sổ hiển thị ảnh
cv2.namedWindow('image')

# load ảnh
image = cv2.imread('09.jpg')

# tạo trackbar cho giá trị Hue
cv2.createTrackbar('H min', 'image', 0, 255, get_hsv_values)
cv2.createTrackbar('H max', 'image', 0, 255, get_hsv_values)

# tạo trackbar cho giá trị Saturation
cv2.createTrackbar('S min', 'image', 0, 255, get_hsv_values)
cv2.createTrackbar('S max', 'image', 0, 255, get_hsv_values)

# tạo trackbar cho giá trị Value
cv2.createTrackbar('V min', 'image', 0, 255, get_hsv_values)
cv2.createTrackbar('V max', 'image', 0, 255, get_hsv_values)

# tạo vòng lặp để mỗi khi kéo thanh trackbar thì giá trị H, S, V được update hiển thị trên ảnh kết quả
# điều kiện vòng lặp while luôn đúng
while True:
    # chuyển sang ảnh màu HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # lấy giá trị min, max của H, S, V từ 3 thanh trackbar
    h_min = cv2.getTrackbarPos('H min', 'image')
    h_max = cv2.getTrackbarPos('H max', 'image')
    s_min = cv2.getTrackbarPos('S min', 'image')
    s_max = cv2.getTrackbarPos('S max', 'image')
    v_min = cv2.getTrackbarPos('V min', 'image')
    v_max = cv2.getTrackbarPos('V max', 'image')

    # gán giá trị min, max của H, S, V vào 2 biến giới hạn trên và giới hạn dưới
    lower_limit = np.array([h_min, s_min, v_min])
    upper_limit = np.array([h_max, s_max, v_max])

    # tạo 1 mặt nạ lọc màu phân loại
    mask = cv2.inRange(hsv_image, lower_limit, upper_limit)

    # áp mặt nạ lọc màu ở trên vào ảnh gốc
    result = cv2.bitwise_and(image, image, mask=mask)

    # hiển thị các ảnh kết quả
    cv2.imshow("orginal", image)
    cv2.imshow("mask", mask)
    cv2.imshow('result', result)

    # nếu phím Q được nhấn thì thoát chương trình
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # in giá trị giới hạn trên-dưới của H, S, V
    print('H min: {}, H max: {}, S min: {}, S max: {}, V min: {}, V max: {}'.format(
        h_min, h_max, s_min, s_max, v_min, v_max))

# đóng chương trình
cv2.destroyAllWindows()
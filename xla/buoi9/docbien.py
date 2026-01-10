# import thư viện
from PIL import Image
import pytesseract
import numpy as np
import cv2

# khai báo đường dẫn đến tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# PSM 6: coi ảnh như 1 khối văn bản đơn cột; giữ khoảng trắng giữa các từ
cfg = r"--oem 3 --psm 6 -c preserve_interword_spaces=1 -c user_defined_dpi=300"

# load ảnh cần OCR
img_1 = np.array(Image.open("bien_so_1.png"))

# OCR ảnh ko có tiền xử lý
text_1 = pytesseract.image_to_string(img_1, lang="eng", config=cfg)
# in kết quả OCR
print("Result OCR without image_processing: ", text_1)

# show ảnh cần OCR
cv2.imshow("OCR", img_1)

# chuẩn hóa ảnh
norm_img = np.zeros((img_1.shape[0], img_1.shape[1]))
img_2 = cv2.normalize(img_1, norm_img, 0, 255, cv2.NORM_MINMAX)
cv2.imshow("Normalize", img_2)

# lấy ngưỡng
img_3 = cv2.threshold(img_2, 180, 255, cv2.THRESH_BINARY)[1]
cv2.imshow("Threshold", img_3)

# làm mờ
img_4 = cv2.GaussianBlur(img_3, (1, 1), 0)
cv2.imshow("GaussianBlur", img_4)

# OCR ảnh sau tiền xử lý
text_2 = pytesseract.image_to_string(img_4, lang="eng", config=cfg)
# in kết quả OCR
print("Result OCR with image_processing: ", text_2)

# chờ nhấn phím rồi thoát
cv2.waitKey(0)
cv2.destroyAllWindows()
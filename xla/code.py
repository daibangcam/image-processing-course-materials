# import thư viện
import cv2
import qrcode
# đọc ảnh qrcode
img = cv2.imread('qrcode.png')
# show ảnh qrcode
cv2.imshow('QR Code', img)
# tạo 1 bộ giải mã qrcode tên là detector
detector = cv2.QRCodeDetector()
# giải mã QR code
val, b, c = detector.detectAndDecode(img)
# in nội dung QR code
print(val)
# chờ nhấn phím rồi thoát
cv2.waitKey(0)
cv2.destroyAllWindows()
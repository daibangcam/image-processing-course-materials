from PIL import Image
import pytesseract
import numpy as np
import cv2
import math
import glob
import os


# --- Đường dẫn ảnh ---
image_path = "11.jpg"
output_dir = "output_regions"
os.makedirs(output_dir, exist_ok=True)

# --- Đọc ảnh ---
image = cv2.imread(image_path)





# Hiển thị ảnh

if image is None:
    print("❌ Không tìm thấy ảnh:", image_path)
    exit()

# --- B1: Chuyển sang HSV và lọc màu ---
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([17, 0, 147])
upper = np.array([114, 95, 226])
mask = cv2.inRange(hsv, lower, upper)
filtered = cv2.bitwise_and(image, image, mask=mask)

# --- B2: Làm mượt, chuyển sang grayscale ---
blur = cv2.GaussianBlur(filtered, (5, 5), 0)
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

# --- B3: Tăng độ nét ---
gauss = cv2.GaussianBlur(gray, (9, 9), 10.0)
sharpened = cv2.addWeighted(gray, 1.5, gauss, -0.5, 0)

# --- B4: Chuyển sang trắng đen ---
_, binary = cv2.threshold(sharpened, 130, 255, cv2.THRESH_BINARY)

# --- B5: Tìm contours ---
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

output = image.copy()
count = 0
cropped_first = None

for contour in contours:
    area = cv2.contourArea(contour)
    if area < 2200 or area >3200 :
        continue

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    (cx, cy), (w, h), angle = rect

    # --- Chỉnh lại nếu bị lộn ngang/dọc ---
    if w < h:
        angle = angle - 90
        w, h = h, w

    # --- Tính tỉ lệ dài/rộng ---
    ratio = w / h if h != 0 else 0

    # --- Bỏ qua nếu tỉ lệ > 2 hoặc < 0.5 ---
    if ratio > 1.8 or ratio < 1.1 :
        print(f"⚠️ Bỏ qua vùng có tỉ lệ {ratio:.2f} (w={w:.1f}, h={h:.1f})")
        continue

        # --- Vẽ khung ---

    cv2.drawContours(output, [box], 0, (0, 255, 0), 3)
    count += 1

    # --- Xoay ảnh quanh tâm ---
    rotation_matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

    # --- Cắt phần biển số ---
    x, y = int(cx - w / 2), int(cy - h / 2)
    cropped = rotated[int(y):int(y + h), int(x):int(x + w)]

    # --- Bỏ qua nếu vùng rỗng ---
    if cropped.size == 0:
        continue

    # --- Resize để dễ OCR ---
    cropped_resized = cv2.resize(cropped, (300, 120))

    # --- Lưu ảnh ---
    filename = f"{output_dir}/bien_so_{count}.png"
    cv2.imwrite(filename, cropped_resized)
    print(f"✅ Đã cắt & lưu biển số {count}: {filename}")

    # --- Giữ lại biển đầu tiên để OCR ---
    if cropped_first is None:
        cropped_first = cropped_resized

    # --- Đánh dấu điểm góc ---
    for (px, py) in box:
        cv2.circle(output, (px, py), 5, (0, 0, 255), -1)

print(f"\n🔹 Tổng số biển số hợp lệ đã cắt: {count}")
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# === 1. ĐỌC ẢNH BIỂN SỐ ===
folder_path = "output_regions"  # Thư mục chứa ảnh đã cắt
image_files = sorted(glob.glob(os.path.join(folder_path, "*.png")))

if not image_files:
    raise FileNotFoundError(f"Không tìm thấy ảnh trong thư mục: {folder_path}")

print(f"==> Tổng số ảnh được tìm thấy: {len(image_files)}")

final_text = ""
# === 2. VÒNG LẶP QUA TỪNG ẢNH ===
for idx, file_path in enumerate(image_files):
   # print(f"\n=== ẢNH {idx+1}/{len(image_files)}: {os.path.basename(file_path)} ===")

    img = cv2.imread(file_path)
    if img is None:
       # print(f"⚠ Không đọc được ảnh: {file_path}")
        continue
    img = cv2.convertScaleAbs(img, alpha=1, beta=10)
    # === 2. TIỀN XỬ LÝ ===
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (1, 1), 1)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 45, 15
    )
  #  cv2.imshow(f"Threshold_{idx+1}", thresh)

    # === 3. TÌM KHUNG KÝ TỰ ===
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    char_regions = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = h / w
        if 15 < h < 70 and 15 < w < 120 and 0.2 < aspect_ratio < 4.0:
            if 50 < w < 120:
                # Chia khung thành 2 phần
                half_w = w // 2
                char_regions.append((x, y, half_w, h))  # phần trái
                char_regions.append((x + half_w, y, w - half_w, h))  # phần phải
            else:
                char_regions.append((x, y, w, h))

   # Sắp xếp theo dòng và cột
    char_regions = sorted(char_regions, key=lambda b: (b[1] // 50, b[0]))
   # print(f"===> Tổng số ô ký tự phát hiện: {len(char_regions)}")

    # === 4. VẼ KHUNG KÝ TỰ ===
    for (x, y, w, h) in char_regions:
        pad = 7
        if 60 < w < 120:
            # Chia làm 2 khung
            half_w = w // 2
            # Vẽ khung bên trái
            cv2.rectangle(img, (x - pad, y - pad), (x + half_w + pad, y + h + pad), (0, 255, 0), 2)
            # Vẽ khung bên phải
            cv2.rectangle(img, (x + half_w - pad, y - pad), (x + w + pad, y + h + pad), (0, 255, 0), 2)
        else:
            # Vẽ khung bình thường
            cv2.rectangle(img, (x - pad, y - pad), (x + w + pad, y + h + pad), (0, 255, 0), 2)
 #   cv2.imshow(f"Detected_{idx+1}", img)

    # === 5. NHẬN DIỆN BAN ĐẦU ===
    char_info_list = []
    for i, (x, y, w, h) in enumerate(char_regions):
        pad = 7
        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + w + pad, thresh.shape[1])
        y2 = min(y + h + pad, thresh.shape[0])
        char_img = thresh[y1:y2, x1:x2]

        #cv2.imshow(f"Char_{idx+1}_{i+1}_Raw", char_img)
      #  cv2.waitKey(200)

        black_ratio = np.sum(char_img > 0) / (char_img.size)
        if black_ratio < 0.1:
        #    print(f"⚠ Char {i+1}: bỏ qua (vùng đen chỉ {black_ratio*100:.1f}%)")
            continue

        if idx < 2:
            whitelist = "023456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # 2 ký tự đầu có thể là số hoặc chữ
        elif idx == 2:
            whitelist = "FBHYE"
        elif idx == 4:
            whitelist = "FBHYEBGC9EFWALCTD0123457"
        elif idx == 8:
            whitelist = "FBHYEBGC9EFWALCTD0123457"
        else:
            whitelist = "FBHYEBGC9EFWALCD012634578"

        cfg = f"--oem 3 --psm 7 -c tessedit_char_whitelist={whitelist}"
        char1 = pytesseract.image_to_string(char_img, lang='eng', config=cfg).strip()
        char1 = char1.replace("\n", "").replace("\f", "")
        if len(char1) > 1:
            char1 = char1[-1]

      #  print(f"Lần 1 - Char {i+1}: '{char1}'")

        char_info_list.append({
            'img': char_img,
            'whitelist': whitelist,
            'raw': char1,
            'confirmed': len(char1) == 1
        })

    # === 6. XOAY KÝ TỰ ===
    for i, info in enumerate(char_info_list):
        char_img = info['img']
        whitelist = info['whitelist']
        char1 = info['raw']

        # Nếu là 2 ký tự đầu mà nhận ra CHỮ → ép xoay lại để thử ra SỐ
        force_rotate = False
        if i < 2 and (not char1.isdigit()):
            force_rotate = True

        if info['confirmed'] and not force_rotate:
     #       print(f"→ Char {i+1}: '{char1}' (đã nhận diện chính xác, không xoay)")
            final_text += char1
            continue

        coords = np.column_stack(np.where(char_img > 0))
        if len(coords) < 10:
            final_text += char1 if len(char1) == 1 else ""
            continue

        vx, vy, _, _ = cv2.fitLine(coords, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy = float(vx), float(vy)
        angle = np.degrees(np.arctan2(vy, vx))
    #    print(f"Char {i+1}: nghiêng {angle:.2f}°, thử xoay thuận và nghịch...")

        h_c, w_c = char_img.shape[:2]
        best_char, best_angle, best_img = "", None, None

        # Xoay cả thuận và nghịch
        for direction in [1, -1]:
            for delta in np.arange(-8, 8, 2):
                test_angle = direction * angle + delta
                M = cv2.getRotationMatrix2D((w_c // 2, h_c // 2), test_angle, 1)
                rotated = cv2.warpAffine(char_img, M, (w_c, h_c),
                                         flags=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(0, 0, 0))

                test_img = cv2.copyMakeBorder(rotated, 10, 10, 10, 10,
                                              cv2.BORDER_CONSTANT, value=[0, 0, 0])
                test_img = cv2.bitwise_not(test_img)
                test_img = cv2.resize(test_img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                _, test_img = cv2.threshold(test_img, 0, 255,
                                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                cfg2 = f"--oem 3 --psm 10 -c tessedit_char_whitelist={whitelist}"
                detected = pytesseract.image_to_string(test_img, lang='eng', config=cfg2).strip()
                detected = detected.replace("\n", "").replace("\f", "")
                if len(detected) > 1:
                    detected = detected[-1]

                # === 👇 Hiển thị có ghi ký tự lên tiêu đề cửa sổ
             #   window_name = f"Char_{idx+1}_{i+1}_Angle_{test_angle:+.1f}_→_{detected}"
              #  cv2.imshow(window_name, test_img)
           #     cv2.waitKey(100)

                if len(detected) == 1 and detected in whitelist:
                    if i < 2 and not detected.isdigit():
                        continue
               #     print(f"   ✅ Góc {test_angle:+.1f}° → '{detected}'")
                    best_char, best_angle, best_img = detected, test_angle, test_img.copy()
                    break
               # else:
                    print(f"   ❌ Góc {test_angle:+.1f}° → '{detected}'")
#
            if best_char:
                break

        if best_char:
       #     print(f"→ Char {i+1}: '{best_char}' (OK ở góc {best_angle:+.1f}°)")
            final_text += best_char
         #   cv2.imshow(f"Char_{idx+1}_{i+1}_Best_{best_char} ({best_angle:+.1f}°)", best_img)
        else:
          #  if i < 2:
             #   print(f"⚠ Char {i+1}: không nhận ra số, bỏ qua.")
            #else:
                final_text += char1 if len(char1) == 1 else ""
          #  cv2.imshow(f"Char_{idx+1}_{i+1}_Fail_{char1}", char_img)


    def fix_common_errors(text):
        rep = {'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '4',
               'Z': '4', 'S': '5', 'B': '8', 'F': '3', 'W': '7',
                'A': '4', 'C': '9', 'G': '9', 'E': '3','T':'Y'}
        fixed = ""
        first_three = list(text[:3])
        if len(first_three) >= 3:
            char3 = first_three[2]
        else:
            char3 = ''
        for i in range(min(2, len(first_three))):
            ch = first_three[i]
            if ch not in "0123456789":
                ch = rep.get(ch, '0')
            first_three[i] = ch
        fixed += ''.join(first_three[:2])
        if char3:
            fixed += char3
        for ch in text[3:]:
            ch = rep.get(ch, ch)
            fixed += ch
        return fixed
    fixed_text = fix_common_errors(final_text)
   # print("\n===> RESULT (ĐÃ CHỈNH):", fixed_text)
    print(f"bien_so_{idx+1}", fixed_text)
    cv2.imshow(f"{fixed_text}", thresh)
    final_text = ""
    cv2.waitKey(500)



# --- Hiển thị ---
cv2.imshow("ANH_GOC", image)
cv2.imshow("SELECT BIEN", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

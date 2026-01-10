from PIL import Image
import pytesseract
import numpy as np
import cv2
import math
import glob
import os

# === 0. ĐƯỜNG DẪN TESSERACT ===
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
    print(f"\n=== ẢNH {idx+1}/{len(image_files)}: {os.path.basename(file_path)} ===")

    img = cv2.imread(file_path)
    if img is None:
        print(f"⚠ Không đọc được ảnh: {file_path}")
        continue

    # === 2. TIỀN XỬ LÝ ===
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (1, 1), 1)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 45, 20
    )
  #  cv2.imshow(f"Threshold_{idx+1}", thresh)

    # === 3. TÌM KHUNG KÝ TỰ ===
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    char_regions = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = h / w
        if 20 < h < 150 and 10 < w < 100 and 1.0 < aspect_ratio < 6.0:
            char_regions.append((x, y, w, h))

    char_regions = sorted(char_regions, key=lambda b: (b[1] // 50, b[0]))
    print(f"===> Tổng số ô ký tự phát hiện: {len(char_regions)}")

    # === 4. VẼ KHUNG KÝ TỰ ===
    for (x, y, w, h) in char_regions:
        pad = 16
        cv2.rectangle(img, (x - pad, y - pad), (x + w + pad, y + h + pad), (0, 255, 0), 2)
 #   cv2.imshow(f"Detected_{idx+1}", img)

    # === 5. NHẬN DIỆN BAN ĐẦU ===
    char_info_list = []
    for i, (x, y, w, h) in enumerate(char_regions):
        pad = 9
        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + w + pad, thresh.shape[1])
        y2 = min(y + h + pad, thresh.shape[0])
        char_img = thresh[y1:y2, x1:x2]

        #cv2.imshow(f"Char_{idx+1}_{i+1}_Raw", char_img)
      #  cv2.waitKey(200)

        black_ratio = np.sum(char_img > 0) / (char_img.size)
        if black_ratio < 0.15:
            print(f"⚠ Char {i+1}: bỏ qua (vùng đen chỉ {black_ratio*100:.1f}%)")
            continue

        if i < 2:
            whitelist = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        elif i == 2:
            whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        else:
            whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

        cfg = f"--oem 3 --psm 10 -c tessedit_char_whitelist={whitelist}"
        char1 = pytesseract.image_to_string(char_img, lang='eng', config=cfg).strip()
        char1 = char1.replace("\n", "").replace("\f", "")
        if len(char1) > 1:
            char1 = char1[-1]

        print(f"Lần 1 - Char {i+1}: '{char1}'")

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
            print(f"→ Char {i+1}: '{char1}' (đã nhận diện chính xác, không xoay)")
            final_text += char1
            continue

        coords = np.column_stack(np.where(char_img > 0))
        if len(coords) < 10:
            final_text += char1 if len(char1) == 1 else ""
            continue

        vx, vy, _, _ = cv2.fitLine(coords, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy = float(vx), float(vy)
        angle = np.degrees(np.arctan2(vy, vx))
        print(f"Char {i+1}: nghiêng {angle:.2f}°, thử xoay thuận và nghịch...")

        h_c, w_c = char_img.shape[:2]
        best_char, best_angle, best_img = "", None, None

        # Xoay cả thuận và nghịch
        for direction in [1, -1]:
            for delta in np.arange(-8, 9, 2):
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
                    print(f"   ✅ Góc {test_angle:+.1f}° → '{detected}'")
                    best_char, best_angle, best_img = detected, test_angle, test_img.copy()
                    break
                else:
                    print(f"   ❌ Góc {test_angle:+.1f}° → '{detected}'")

            if best_char:
                break

        if best_char:
            print(f"→ Char {i+1}: '{best_char}' (OK ở góc {best_angle:+.1f}°)")
            final_text += best_char
         #   cv2.imshow(f"Char_{idx+1}_{i+1}_Best_{best_char} ({best_angle:+.1f}°)", best_img)
        else:
            if i < 2:
                print(f"⚠ Char {i+1}: không nhận ra số, bỏ qua.")
            else:
                final_text += char1 if len(char1) == 1 else ""
          #  cv2.imshow(f"Char_{idx+1}_{i+1}_Fail_{char1}", char_img)


    def fix_common_errors(text):
        rep = {'O': '0', 'Q': '0', 'I': '1', 'L': '4',
               'Z': '4', 'S': '5', 'B': '8', 'F': '3',
               'T': '1', 'A': '2', 'G': '9', 'E': '3'}
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
    cv2.imshow(f"{fixed_text}", thresh)
    final_text = ""
    cv2.waitKey(500)
# === 7. SỬA LỖI NHẬN DẠNG THƯỜNG GẶP ===



cv2.waitKey(0)
cv2.destroyAllWindows()

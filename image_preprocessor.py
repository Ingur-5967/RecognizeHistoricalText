import cv2
import numpy as np
import os
import glob

def local_deskew(line_img):
    _, line_bin = cv2.threshold(line_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(line_bin > 0))

    if len(coords) > 50:
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) > 1.0:
            (lh, lw) = line_img.shape[:2]
            center_line = (lw // 2, lh // 2)
            M_line = cv2.getRotationMatrix2D(center_line, angle, 1.0)

            cos_l = np.abs(M_line[0, 0])
            sin_l = np.abs(M_line[0, 1])
            new_lw = int((lh * sin_l) + (lw * cos_l))
            new_lh = int((lh * cos_l) + (lw * sin_l))

            M_line[0, 2] += (new_lw / 2) - center_line[0]
            M_line[1, 2] += (new_lh / 2) - center_line[1]

            line_img = cv2.warpAffine(line_img, M_line, (new_lw, new_lh),
                                      borderValue=255, flags=cv2.INTER_CUBIC)
    return line_img


def preprocess_and_segment(image_path, output_dir, page_id):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Не удалось загрузить: {image_path}")
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    denoised = cv2.fastNlMeansDenoising(gray, None, h=15, templateWindowSize=7, searchWindowSize=21)

    bin_img_inv = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 15, 10)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 2))
    detected_lines = cv2.morphologyEx(bin_img_inv, cv2.MORPH_CLOSE, horizontal_kernel, iterations=2)

    cnts, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[1])

    os.makedirs(output_dir, exist_ok=True)

    saved_lines = []

    for i, cnt in enumerate(cnts):
        x, y, w, h = cv2.boundingRect(cnt)

        if w < 50 or h < 10:
            continue

        pad = 15
        y1 = max(0, y - pad)
        y2 = min(denoised.shape[0], y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(denoised.shape[1], x + w + pad)

        line_img = denoised[y1:y2, x1:x2]

        line_img = local_deskew(line_img)

        line_filename = f"{page_id}_line_{i:03d}.png"
        cv2.imwrite(os.path.join(output_dir, line_filename), line_img)
        saved_lines.append(line_filename)

    return saved_lines


if __name__ == "__main__":
    input_folder = "raw_photos/"
    output_folder = "processed_lines/"

    os.makedirs(input_folder, exist_ok=True)

    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")
    all_images = []
    for ext in image_extensions:
        all_images.extend(glob.glob(os.path.join(input_folder, ext)))

    if not all_images:
        print(f"В папке {input_folder} не найдено изображений. Положите туда фото рукописей.")
    else:
        for img_path in all_images:
            page_id = os.path.splitext(os.path.basename(img_path))[0]
            print(f"Обрабатываю страницу: {page_id}...")
            lines = preprocess_and_segment(img_path, output_folder, page_id)
            print(f"  -> Успешно нарезано и выровнено {len(lines)} строк.")

        print("\nГотово! Проверьте папку:", output_folder)
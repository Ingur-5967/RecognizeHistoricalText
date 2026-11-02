
import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import logging
import json
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

FONT_PATH = os.getenv("FONT_PATH", "./font/gogol_regular.otf")
GT_FILE = os.getenv("GT_FILE", "prereform_words.txt")
OUTPUT_DIR = os.getenv("IMG_DIR", "C:/ml-output")

FONT_SIZE = int(os.getenv("FONT_SIZE", "64"))
CANVAS_WIDTH = int(os.getenv("CANVAS_WIDTH", "1200"))
CANVAS_HEIGHT = int(os.getenv("CANVAS_HEIGHT", "200"))
PADDING = int(os.getenv("PADDING", "25"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler("generation_log.txt", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RealismEffects:
    @staticmethod
    def elastic_transform(image: np.ndarray, alpha: float = 15.0, sigma: float = 3.5) -> np.ndarray:
        shape = image.shape[:2]
        h, w = shape
        dx = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1).astype(np.float32), ksize=(0, 0), sigmaX=sigma) * alpha
        dy = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1).astype(np.float32), ksize=(0, 0), sigmaX=sigma) * alpha
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    @staticmethod
    def simulate_ink_spread(image: np.ndarray, intensity: float = 0.25) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image.copy()
        inverted = 255 - gray
        kernel_size = max(2, int(2.5 * intensity))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        spread = cv2.dilate(inverted, kernel, iterations=1)
        blur_size = max(3, int(3.5 * intensity))
        if blur_size % 2 == 0: blur_size += 1
        spread = cv2.GaussianBlur(spread, (blur_size, blur_size), 0)
        result = 255 - spread
        return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB) if len(image.shape) == 3 else result

    @staticmethod
    def add_paper_texture(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        h, w = image.shape[:2]
        base_color = np.array([240, 235, 220], dtype=np.float32)
        paper = np.ones((h, w, 3), dtype=np.float32) * base_color
        paper += np.random.normal(0, 12, (h, w, 3))

        for _ in range(random.randint(4, 8)):
            cx, cy = random.randint(0, w), random.randint(0, h)
            radius = random.randint(30, 100)
            y_grid, x_grid = np.ogrid[:h, :w]
            mask = np.exp(-((x_grid - cx)**2 + (y_grid - cy)**2) / (2 * radius**2))
            mask_3d = mask[:, :, np.newaxis]
            stain_color = np.array([base_color[0]*0.7, base_color[1]*0.7, base_color[2]*0.7], dtype=np.float32)
            paper = np.clip(paper * (1 - mask_3d * 0.25) + stain_color * mask_3d * 0.25, 0, 255)

        result = image.astype(np.float32) * (paper / 255.0)
        sepia = intensity * 0.15
        result[:, :, 0] = np.clip(result[:, :, 0] * (1 + sepia), 0, 255)
        result[:, :, 1] = np.clip(result[:, :, 1] * (1 + sepia * 0.4), 0, 255)
        result[:, :, 2] = np.clip(result[:, :, 2] * (1 - sepia * 0.2), 0, 255)

        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def simulate_low_resolution(image: np.ndarray, probability: float = 0.4) -> np.ndarray:
        if random.random() > probability:
            return image
        h, w = image.shape[:2]
        scale = random.uniform(0.3, 0.6)
        small_w = max(1, int(w * scale))
        small_h = max(1, int(h * scale))
        small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
        degraded = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        degraded = cv2.GaussianBlur(degraded, (3, 3), 0)
        return degraded


DESCENDER_WIDE_CHARS = {'д', 'у', 'ц', 'щ', 'ъ', 'ь', 'р', 'й'}
NORMAL_CURSIVE_CHARS = {'а', 'б', 'в', 'г', 'е', 'ж', 'з', 'и', 'к', 'л', 'м', 'н', 'о', 'п', 'с', 'т', 'ф', 'х', 'ч', 'ш', 'э', 'ю', 'я', 'ѣ', 'і'}

class PrereformGenerator:
    def __init__(self, font_path, font_size):
        if not Path(font_path).exists():
            raise FileNotFoundError(f"Шрифт не найден: {font_path}")
        self.font = ImageFont.truetype(font_path, font_size)
        self.effects = RealismEffects()

    def render_text(self, text: str) -> tuple[np.ndarray, str]:
        img = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        x_pos = 50
        baseline_y = CANVAS_HEIGHT // 2 + 20

        gray_level = random.randint(150, 175)
        base_ink_color = (gray_level, gray_level - 5, gray_level - 10)

        barely_legible_count = 0

        for char_idx, char in enumerate(text):
            if char == '\n':
                x_pos = 50
                baseline_y += int(FONT_SIZE * 1.6)
                continue

            is_letter = char.isalpha()
            is_digit = char.isdigit()

            if is_letter or is_digit:
                is_faint = random.random() < 0.15
                if is_faint:
                    barely_legible_count += 1

                size_var = random.uniform(0.90, 1.05)
                varied_size = int(FONT_SIZE * size_var)
                varied_font = ImageFont.truetype(self.font.path, varied_size)

                size_diff = FONT_SIZE - varied_size
                baseline_offset = int(size_diff * 0.2)
                draw_y = baseline_y + baseline_offset

                if is_faint:
                    alpha = random.randint(130, 180)
                    faint_gray = random.randint(160, 190)
                    faint_color = (faint_gray, faint_gray - 5, faint_gray - 10, alpha)

                    temp_size = int(varied_size * 1.3)
                    temp_img = Image.new("RGBA", (temp_size, temp_size), (255, 255, 255, 0))
                    temp_draw = ImageDraw.Draw(temp_img)
                    temp_draw.text((0, 0), char, fill=faint_color, font=varied_font)
                    temp_img = temp_img.filter(ImageFilter.GaussianBlur(radius=0.4))

                    img.paste(temp_img, (int(x_pos), int(draw_y)), temp_img)
                else:
                    draw.text((int(x_pos), int(draw_y)), char, fill=base_ink_color, font=varied_font)

                base_advance = self.font.getlength(char)

                if char in DESCENDER_WIDE_CHARS:
                    squeeze_factor = random.uniform(0.50, 0.65)
                elif char in NORMAL_CURSIVE_CHARS:
                    squeeze_factor = random.uniform(0.70, 0.85)
                else:
                    squeeze_factor = random.uniform(0.75, 0.90)

                actual_advance = int(base_advance * squeeze_factor)
                actual_advance = max(actual_advance, int(FONT_SIZE * 0.25))

                x_pos += actual_advance

            else:
                punct_advance = self.font.getlength(char)
                if char == ' ':
                    x_pos += int(self.font.getlength('о') * random.uniform(1.5, 2.5))
                elif char in {',', '.', ';', ':', '!', '?', '-', '—'}:
                    x_pos += int(punct_advance * 1.2)
                else:
                    x_pos += int(punct_advance)

        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        text_mask = gray < 245
        coords = np.column_stack(np.where(text_mask))

        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            y_min = max(0, y_min - PADDING)
            x_min = max(0, x_min - PADDING)
            y_max = min(img_np.shape[0] - 1, y_max + PADDING)
            x_max = min(img_np.shape[1] - 1, x_max + PADDING)

            cropped = img_np[y_min:y_max+1, x_min:x_max+1]
            img = Image.fromarray(cropped)

        img = self.post_process_text(img)
        effects_str = f"Слитные, {barely_legible_count} шт. с эффектом недопрокраса"
        return np.array(img), effects_str

    def post_process_text(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img)
        img_pil = Image.fromarray(img_np).filter(ImageFilter.GaussianBlur(radius=0.4))
        img_np = np.array(img_pil)

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        inverted = 255 - gray

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        closed = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel, iterations=1)
        connected_gray = 255 - closed
        connected_rgb = cv2.cvtColor(connected_gray, cv2.COLOR_GRAY2RGB)

        text_mask = (connected_gray < 225).astype(np.float32)
        noise = np.random.normal(0, 10, connected_rgb.shape)
        background_mask = 1.0 - text_mask[:, :, np.newaxis]
        noisy_bg = connected_rgb.astype(np.float32) + noise * background_mask

        return Image.fromarray(np.clip(noisy_bg, 0, 255).astype(np.uint8))

    def generate_image(self, text: str, save_path: str):
        filename = Path(save_path).name
        img, char_effects_log = self.render_text(text)

        alpha = random.uniform(14.0, 17.5)
        sigma = random.uniform(3.5, 5.0)
        img = self.effects.elastic_transform(img, alpha=alpha, sigma=sigma)

        ink_intensity = random.uniform(0.24, 0.35)
        img = self.effects.simulate_ink_spread(img, intensity=ink_intensity)

        img = self.effects.simulate_low_resolution(img, probability=0.4)

        paper_intensity = random.uniform(0.5, 0.7)
        img = self.effects.add_paper_texture(img, intensity=paper_intensity)

        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        save_success = cv2.imwrite(save_path, bgr_img)
        if not save_success:
            raise IOError(f"cv2.imwrite не смог сохранить файл! Проверьте путь: {save_path}")

        height, width = img.shape[:2]

        return {
            "filename": filename,
            "text": text,
            "effects": char_effects_log,
            "width": width,
            "height": height
        }


def read_word_list(file_path: str) -> list:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Файл со словами не найден: {file_path}")

    entries = []

    for encoding in ['utf-8', 'utf-8-sig', 'cp1251']:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                lines = f.readlines()

            for line in lines:
                word = line.strip()
                if not word:
                    continue

                filename = f"gen_{len(entries):06d}.png"
                entries.append((filename, word))

            logger.info(f"✅ Файл прочитан (кодировка: {encoding})")
            logger.info(f"   Всего слов для генерации: {len(entries)}")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(f"Ошибка при чтении файла: {e}")
            return []
    else:
        logger.error("Не удалось определить кодировку файла. Пожалуйста, сохраните его в UTF-8.")
        return []

    return entries


def main():
    logger.info("="*80)
    logger.info("ГЕНЕРАЦИЯ ИЗОБРАЖЕНИЙ ИЗ СПИСКА СЛОВ (ПО ОДНОМУ СЛОВУ НА СТРОКУ)")
    logger.info(f"Файл со словами: {GT_FILE}")
    logger.info(f"Папка вывода: {OUTPUT_DIR}")
    logger.info(f"Шрифт: {FONT_PATH} (Размер: {FONT_SIZE})")
    logger.info(f"Холст: {CANVAS_WIDTH}x{CANVAS_HEIGHT}, Padding: {PADDING}")
    logger.info("="*80)

    entries = read_word_list(GT_FILE)
    if not entries:
        logger.error("Не удалось прочитать файл. Завершение.")
        return

    char_counter = Counter()
    for filename, word in entries:
        for char in word:
            if char.isalpha():
                char_counter[char.lower()] += 1

    logger.info(f"\nУникальных слов: {len(set(w for _, w in entries))}")
    logger.info(f"\nЧАСТОТНОСТЬ БУКВ ДО ГЕНЕРАЦИИ:")
    logger.info(f"{'Буква':<8} {'Встречается':<15} {'Статус':<20}")
    logger.info("-" * 50)

    for char in sorted(char_counter.keys()):
        count = char_counter[char]
        status = "✓ отлично" if count >= 300 else "⚠ достаточно" if count >= 200 else "✗ МАЛО"
        logger.info(f"  {char:<6} {count:>10}       {status}")

    try:
        generator = PrereformGenerator(FONT_PATH, FONT_SIZE)
    except FileNotFoundError as e:
        logger.error(f"ОШИБКА: {e}")
        return

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    logger.info(f"\nНачало генерации {len(entries)} изображений...")
    logger.info("-" * 80)

    metadata_path = Path(OUTPUT_DIR) / "metadata.json"
    final_char_counter = Counter()
    total_images = 0
    errors = 0

    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.write('[\n')

        for i, (filename, word) in enumerate(entries):
            filepath = str(Path(OUTPUT_DIR) / filename)

            try:
                meta = generator.generate_image(word, filepath)
                if meta is None:
                    raise RuntimeError("generate_image вернул None")

                if total_images > 0:
                    f.write(',\n')

                json_str = json.dumps(meta, ensure_ascii=False, indent=2)
                f.write(f"  {json_str}")
                f.flush()

                for char in meta['text']:
                    if char.isalpha():
                        final_char_counter[char.lower()] += 1

                total_images += 1

                if (i + 1) % 1000 == 0:
                    logger.info(f"Сгенерировано {i + 1}/{len(entries)} изображений")

            except Exception as e:
                logger.warning(f"❌ Ошибка генерации для слова '{word}': {e}")
                errors += 1
                continue

        f.write('\n]')

    logger.info("\n" + "="*80)
    logger.info(f"ГОТОВО!")
    logger.info(f"Успешно создано изображений: {total_images}")
    logger.info(f"Ошибок: {errors}")
    logger.info(f"\nФИНАЛЬНАЯ СТАТИСТИКА ПО БУКВАМ:")
    logger.info(f"{'Буква':<8} {'Всего вхождений':<20} {'Статус':<20}")
    logger.info("-" * 50)

    for char in sorted(final_char_counter.keys()):
        count = final_char_counter[char]
        status = "✓" if count >= 300 else "⚠" if count >= 200 else "✗"
        logger.info(f"  {char:<6} {count:>12}       {status}")

    logger.info(f"\nМетаданные сохранены (потоковая запись): {metadata_path.resolve()}")
    logger.info(f"Изображения сохранены в: {Path(OUTPUT_DIR).resolve()}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
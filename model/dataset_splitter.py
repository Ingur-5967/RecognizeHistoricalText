import os
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()

METADATA_PATH = Path(os.getenv("METADATA_PATH", "C:/ml-output/metadata.json"))
TRAIN_GT = Path(os.getenv("TRAIN_GT", "C:/ml-output/dataset/train.txt"))
VAL_GT = Path(os.getenv("VAL_GT", "C:/ml-output/dataset/val.txt"))


def split_dataset():
    print(f"Чтение исходного файла {METADATA_PATH.name}...")

    if not METADATA_PATH.exists():
        print(f"Ошибка: Файл метаданных не найден: {METADATA_PATH}")
        return

    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    print(f"Всего записей в metadata.json: {len(metadata)}")

    word_to_images = defaultdict(list)
    skipped_empty = 0

    for item in metadata:
        filename = item.get('filename', '').strip()
        text = item.get('text', '').strip()

        if not filename or not text:
            skipped_empty += 1
            continue

        word_to_images[text].append(f"{filename},{text}\n")

    print(f"Найдено {len(word_to_images)} УНИКАЛЬНЫХ слов.")
    if skipped_empty > 0:
        print(f"Пропущено {skipped_empty} пустых/некорректных записей.")

    unique_words = list(word_to_images.keys())
    train_words, val_words = train_test_split(unique_words, test_size=0.2, random_state=42)

    train_lines = []
    for word in train_words:
        train_lines.extend(word_to_images[word])

    val_lines = []
    for word in val_words:
        val_lines.extend(word_to_images[word])

    TRAIN_GT.parent.mkdir(parents=True, exist_ok=True)
    VAL_GT.parent.mkdir(parents=True, exist_ok=True)

    with open(TRAIN_GT, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)

    with open(VAL_GT, 'w', encoding='utf-8') as f:
        f.writelines(val_lines)

    print("\nДанные успешно разделены БЕЗ УТЕЧКИ (по уникальным словам):")
    print(f"   - Обучение: {len(train_words):,} уник. слов ({len(train_lines):,} картинок) -> {TRAIN_GT}")
    print(f"   - Валидация: {len(val_words):,} уник. слов ({len(val_lines):,} картинок) -> {VAL_GT}")

    print("\nПримеры из train.txt:")
    for line in train_lines[:3]:
        print(f"   {line.strip()}")

if __name__ == "__main__":
    split_dataset()
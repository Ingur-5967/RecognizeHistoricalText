"""
Скрипт для автоматического построения словаря (vocab.json)
на основе текстов из train.txt.
Конфигурация путей загружается из файла .env
"""
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# 🔥 Загрузка переменных окружения из файла .env
load_dotenv()

# Чтение путей из .env с fallback-значениями
TRAIN_GT = Path(os.getenv("TRAIN_GT", "C:/ml-output/dataset/train.txt"))
VOCAB_PATH = Path(os.getenv("VOCAB_PATH"))

def build_vocab():
    if not TRAIN_GT.exists():
        print(f"❌ Файл {TRAIN_GT} не найден! Проверьте путь в .env")
        return

    print(f"🔨 Построение словаря из {TRAIN_GT}...")
    unique_chars = set()
    lines_processed = 0

    with open(TRAIN_GT, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Разделяем по ЗАПЯТОЙ (формат: filename,text)
            parts = line.split(',', 1)
            if len(parts) == 2:
                text = parts[1].strip()
                unique_chars.update(text)
                lines_processed += 1

    print(f"   Обработано строк с текстом: {lines_processed}")

    # Сортируем для детерминированности
    sorted_chars = sorted(list(unique_chars))

    # 🔥 ИСПРАВЛЕНИЕ: <BLANK> должен быть ПЕРВЫМ (индекс 0) для корректной работы CTCLoss
    vocab = ['<BLANK>', '<PAD>'] + sorted_chars

    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {str(idx): char for idx, char in enumerate(vocab)}

    vocab_data = {
        'char2idx': char2idx,
        'idx2char': idx2char,
        'vocab_size': len(vocab)
    }

    # Гарантируем создание директории для словаря (например, папки 'extended')
    VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(VOCAB_PATH, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Словарь успешно создан и сохранен в: {VOCAB_PATH}")
    print(f"   Размер словаря: {len(vocab)} уникальных символов")
    print(f"   Примеры символов: {sorted_chars[:40]}")

if __name__ == "__main__":
    build_vocab()
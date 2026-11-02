import gc
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from dotenv import load_dotenv
from torch.nn import CTCLoss
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

load_dotenv()

from metrics.metrics import decode_ctc_greedy, calculate_cer
from model.model import CRNN

TRAIN_GT = Path(os.getenv("TRAIN_GT", "C:/ml-output/dataset/train.txt"))
VAL_GT = Path(os.getenv("VAL_GT", "C:/ml-output/dataset/val.txt"))
METADATA_PATH = Path(os.getenv("METADATA_PATH", "C:/ml-output/metadata.json"))
VOCAB_PATH = Path(os.getenv("VOCAB_PATH", "C:/ml-output/extended/vocab_extended.json"))
IMG_DIR = Path(os.getenv("IMG_DIR", "C:/ml-output"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "C:/ml-output/artefact-model"))
MODEL_BEST_VERSION_PATH = Path(os.getenv("MODEL_BEST_VERSION_PATH", str(MODEL_DIR / "best_model.pth")))
MODEL_BEST_PARAMS_PATH = Path(os.getenv("MODEL_BEST_PARAMS_PATH", str(MODEL_DIR / "model_best_params.json")))

MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_WORKERS = 0

DEFAULT_PARAMS = {
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.104,
    'learning_rate': 8.35e-05,
    'batch_size': 128
}

FINAL_EPOCHS = int(os.getenv("FINAL_EPOCHS", "20"))
IMG_HEIGHT = int(os.getenv("IMG_HEIGHT", "32"))
MAX_WIDTH = int(os.getenv("MAX_WIDTH", "512"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "50000"))

if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0: return f"{h}ч {m}м {s}с"
    if m > 0: return f"{m}м {s}с"
    return f"{s}с"

def load_training_params(params_path: Path, default_params: dict) -> dict:
    if params_path.exists():
        try:
            with open(params_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            if isinstance(loaded, dict) and loaded:
                merged_params = default_params.copy()
                merged_params.update(loaded)
                print(f"Параметры модели загружены из: {params_path.name}")
                return merged_params
        except (json.JSONDecodeError, Exception) as e:
            print(f"Ошибка чтения {params_path.name} ({e}). Используются значения по умолчанию.")
    return default_params.copy()

def load_vocab() -> Tuple[Dict[str, int], List[str]]:
    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    char2idx = vocab_data['char2idx']
    idx2char = vocab_data['idx2char']
    vocab = [idx2char[str(i)] for i in range(len(idx2char))]
    return char2idx, vocab

def parse_line(line: str):
    line = line.strip()
    if not line: return None, None
    if ',' in line: parts = line.split(',', 1)
    else: parts = line.split(' ', 1)
    if len(parts) != 2: return None, None

    img_path_str = parts[0].strip().replace('\\', '/')
    text = parts[1].strip()
    img_name = img_path_str.split('img/', 1)[-1] if img_path_str.startswith('img/') else img_path_str
    return img_name, text

def load_chunk_into_ram(lines: List[str], img_dir: Path, char2idx: Dict[str, int]):
    chunk_data = []
    skipped = 0
    for line in tqdm(lines, desc="   Загрузка чанка", leave=False):
        img_name, text = parse_line(line)
        if img_name is None or text is None:
            skipped += 1; continue
        img_path = img_dir / img_name
        if not img_path.exists():
            skipped += 1; continue
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None or img.shape[0] == 0:
            skipped += 1; continue

        h, w = img.shape[:2]
        new_w = max(1, int(w * (IMG_HEIGHT / float(h))))
        if new_w > MAX_WIDTH:
            img = cv2.resize(img, (MAX_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
        else:
            img = cv2.resize(img, (new_w, IMG_HEIGHT), interpolation=cv2.INTER_AREA)

        img_float = img.astype(np.float32) / 255.0
        img_array = (img_float - 0.5) / 0.5

        unk_idx = char2idx.get('<UNK>', 0)
        target_indices = [char2idx.get(c, unk_idx) for c in text]
        chunk_data.append((img_array, target_indices, len(target_indices)))

    if skipped > 0: print(f"      ⚠Пропущено: {skipped}")
    return chunk_data

def collate_fn_chunk(batch):
    imgs_np, targets, target_lens = zip(*batch)
    max_w = max(img.shape[1] for img in imgs_np)
    batch_size = len(imgs_np)

    batch_imgs = torch.zeros(batch_size, 1, IMG_HEIGHT, max_w)
    for i, img_np in enumerate(imgs_np):
        h, w = img_np.shape
        batch_imgs[i, 0, :h, :w] = torch.from_numpy(img_np)

    flat_targets = []
    for t in targets:
        flat_targets.extend(t)

    return batch_imgs, torch.tensor(flat_targets, dtype=torch.long), torch.tensor(target_lens, dtype=torch.long)

class TempChunkDataset(Dataset):
    def __init__(self, chunk_data): self.data = chunk_data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def validate(model, val_loader, vocab, device, blank_idx):
    model.eval()
    total_cer, count = 0, 0
    with torch.no_grad():
        for imgs, targets, target_lens in tqdm(val_loader, desc="Val", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            output = model(imgs)
            batch_size = imgs.shape[0]
            for i in range(batch_size):
                target_start = (target_lens[:i].sum()).item()
                target_end = target_start + target_lens[i].item()
                target_indices = targets[target_start:target_end].cpu().tolist()
                pred_indices = decode_ctc_greedy(output[:, i, :].cpu(), blank_idx)

                predicted_text = ''.join([vocab[idx] for idx in pred_indices if idx < len(vocab)])
                target_text = ''.join([vocab[idx] for idx in target_indices if idx < len(vocab)])
                total_cer += calculate_cer(predicted_text, target_text)
                count += 1
    return total_cer / max(count, 1)

def main():
    print("ФИНАЛЬНОЕ ОБУЧЕНИЕ")

    print(f"\nCUDA доступен: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Устройство: {torch.cuda.get_device_name(0)}")
        print(f"Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} ГБ")
        print(f"Потоки загрузки данных: {NUM_WORKERS}")
    else:
        print("PyTorch не видит видеокарту! Обучение пойдет на CPU")
        print("Решение: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

    print(f"\nАктивные пути и параметры из .env:")
    print(f"   Epochs      : {FINAL_EPOCHS}")
    print(f"   Img Height  : {IMG_HEIGHT} | Max Width : {MAX_WIDTH}")
    print(f"   Chunk Size  : {CHUNK_SIZE:,}")
    print(f"   Model Dir   : {MODEL_DIR}")

    if not IMG_DIR.exists():
        print(f"Не найдена папка изображений: {IMG_DIR}"); return

    if not VOCAB_PATH.exists():
        print(f"\nСловарь не найден по пути: {VOCAB_PATH}")
        print("Попытка автоматического создания словаря через model_vocab.py...")
        try:
            from model_vocab import build_vocab
            build_vocab()
            print("Словарь успешно создан.")
        except ImportError:
            print("Ошибка: Не удалось импортировать build_vocab из model_vocab.py.")
            print("   Убедитесь, что файл model_vocab.py находится в корневой папке проекта.")
            return
        except Exception as e:
            print(f"Ошибка при выполнении build_vocab: {e}")
            return

        if not VOCAB_PATH.exists():
            print("Критическая ошибка: Файл словаря так и не был создан.")
            return

    if not TRAIN_GT.exists() or not VAL_GT.exists():
        print("\nФайлы train.txt или val.txt не найдены. Попытка авто-сплита...")
        try:
            from model.dataset_splitter import split_dataset
            split_dataset()
            print("Разделение данных успешно завершено.")
        except Exception as e:
            print(f"Ошибка: {e}\n   Запустите dataset_splitter.py вручную."); return

    active_params = load_training_params(MODEL_BEST_PARAMS_PATH, DEFAULT_PARAMS)
    print(f"\nГиперпараметры модели:")
    for k, v in active_params.items(): print(f"   {k}: {v}")

    char2idx, vocab = load_vocab()
    blank_idx = vocab.index('<BLANK>') if '<BLANK>' in vocab else 0
    print(f"\nСловарь: {len(vocab)} символов | <BLANK> индекс: {blank_idx}")

    with open(TRAIN_GT, 'r', encoding='utf-8') as f:
        all_train_lines = [line.strip() for line in f if line.strip()]
    print(f"   Всего train строк: {len(all_train_lines):,}")

    num_chunks = (len(all_train_lines) + CHUNK_SIZE - 1) // CHUNK_SIZE
    chunks_lines = [all_train_lines[i:i + CHUNK_SIZE] for i in range(0, len(all_train_lines), CHUNK_SIZE)]
    print(f"   Разбито на {num_chunks} чанков")

    with open(VAL_GT, 'r', encoding='utf-8') as f:
        val_lines = [line.strip() for line in f if line.strip()]
    val_data = load_chunk_into_ram(val_lines, IMG_DIR, char2idx)

    val_dataset = TempChunkDataset(val_data)
    val_loader = DataLoader(val_dataset, batch_size=active_params['batch_size'], shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=False, collate_fn=collate_fn_chunk)

    model = CRNN(
        vocab_size=len(vocab),
        hidden_size=active_params['hidden_size'],
        num_layers=active_params['num_layers'],
        dropout=active_params['dropout']
    ).to(DEVICE)
    print(f"   Параметров модели: {sum(p.numel() for p in model.parameters()):,}")

    criterion = CTCLoss(blank=blank_idx, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=active_params['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

    gc.disable()
    best_val_cer = float('inf')
    final_start = time.time()

    print(f"\n🏋️ НАЧАЛО ОБУЧЕНИЯ ({FINAL_EPOCHS} эпох × {num_chunks} чанков)")
    for epoch in range(FINAL_EPOCHS):
        model.train()
        epoch_loss, valid_batches = 0, 0
        epoch_start = time.time()

        for chunk_idx, chunk_lines in enumerate(chunks_lines):
            chunk_data = load_chunk_into_ram(chunk_lines, IMG_DIR, char2idx)
            chunk_dataset = TempChunkDataset(chunk_data)

            chunk_loader = DataLoader(chunk_dataset, batch_size=active_params['batch_size'], shuffle=True,
                                      num_workers=NUM_WORKERS, pin_memory=True, drop_last=True, collate_fn=collate_fn_chunk)

            for imgs, targets, target_lens in tqdm(chunk_loader, desc=f"   Ep {epoch+1} Chunk {chunk_idx+1}", leave=False, unit="b"):
                imgs, target_lens = imgs.to(DEVICE, non_blocking=True), target_lens.to(DEVICE, non_blocking=True)
                optimizer.zero_grad()

                if DEVICE.type == 'cuda':
                    with torch.amp.autocast(device_type='cuda'): output = model(imgs)
                    output = output.float()
                else:
                    output = model(imgs)

                input_lens = torch.full((imgs.shape[0],), output.shape[0], dtype=torch.long, device=DEVICE)
                valid_mask = input_lens > target_lens

                if not valid_mask.all():
                    valid_indices = valid_mask.nonzero(as_tuple=True)[0]
                    if len(valid_indices) == 0: continue
                    imgs, output = imgs[valid_indices], output[:, valid_indices, :]
                    input_lens, target_lens = input_lens[valid_indices], target_lens[valid_indices]
                    targets = torch.cat([targets[(target_lens[:i].sum()).item():(target_lens[:i].sum() + target_lens[i]).item()] for i in valid_indices.cpu().tolist()])
                else:
                    targets = targets

                loss = criterion(output, targets, input_lens, target_lens)
                if not torch.isfinite(loss): continue

                if DEVICE.type == 'cuda':
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer); scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

                epoch_loss += loss.item()
                valid_batches += 1

            del chunk_data, chunk_dataset, chunk_loader
            gc.collect()

        if valid_batches > 0:
            val_cer = validate(model, val_loader, vocab, DEVICE, blank_idx)
            scheduler.step(val_cer)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"   Ep {epoch+1} | Loss: {epoch_loss/valid_batches:.4f} | CER: {val_cer:.4f} | LR: {current_lr:.2e} | ⏱️ {format_time(time.time() - epoch_start)}")

            if val_cer < best_val_cer:
                best_val_cer = val_cer
                torch.save(model.state_dict(), MODEL_BEST_VERSION_PATH)

                params_to_save = {
                    **active_params,
                    'vocab_size': len(vocab),
                    'img_height': IMG_HEIGHT,
                    'max_width': MAX_WIDTH,
                    'best_cer': best_val_cer,
                    'epoch_saved': epoch + 1
                }
                with open(MODEL_BEST_PARAMS_PATH, 'w', encoding='utf-8') as f:
                    json.dump(params_to_save, f, indent=2, ensure_ascii=False)

                print(f"   НОВЫЙ РЕКОРД! CER: {best_val_cer:.4f} (Веса и параметры сохранены)")

    gc.enable()
    print(f"\nОбучение завершено за {format_time(time.time() - final_start)} | Лучший CER: {best_val_cer:.4f}")
    print(f"Модель: {MODEL_BEST_VERSION_PATH}")
    print(f"Параметры: {MODEL_BEST_PARAMS_PATH}")

if __name__ == "__main__":
    main()
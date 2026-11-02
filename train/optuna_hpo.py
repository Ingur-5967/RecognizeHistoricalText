import os

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

import gc
import time
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CTCLoss
from torch.utils.data import Dataset, DataLoader
import optuna
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

from model.model import CRNN
from metrics.metrics import decode_ctc_greedy, calculate_cer

TRAIN_GT = Path(os.getenv("TRAIN_GT"))
VAL_GT = Path(os.getenv("VAL_GT"))
VOCAB_PATH = Path(os.getenv("VOCAB_PATH"))
IMG_DIR = Path(os.getenv("IMG_DIR"))

N_TRIALS = int(os.getenv("N_TRIALS"))
N_JOBS = int(os.getenv("N_JOBS"))
EPOCHS_PER_TRIAL = int(os.getenv("EPOCHS_PER_TRIAL"))
IMG_HEIGHT = int(os.getenv("IMG_HEIGHT"))
MAX_WIDTH = int(os.getenv("MAX_WIDTH"))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} ГБ")
else:
    torch.backends.mkldnn.enabled = True
    torch.set_num_threads(8)
    print(f"⚙️  CPU: {os.cpu_count()} ядер")

if DEVICE.type == 'cuda':
    torch.cuda.empty_cache()

global_train_lines = []
global_val_lines = []


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0: return f"{h}ч {m}м {s}с"
    if m > 0: return f"{m}м {s}с"
    return f"{s}с"


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
    parts = line.split(',', 1) if ',' in line else line.split(' ', 1)
    if len(parts) != 2: return None, None

    img_path_str = parts[0].strip().replace('\\', '/')
    text = parts[1].strip()
    img_name = img_path_str.split('img/', 1)[-1] if img_path_str.startswith('img/') else img_path_str
    return img_name, text


class HTRDiskDataset(Dataset):
    def __init__(self, lines: List[str], img_dir: Path, char2idx: Dict[str, int]):
        self.samples = []
        self.img_dir = img_dir
        self.char2idx = char2idx
        self.unk_idx = char2idx.get('<UNK>', 0)

        for line in lines:
            img_name, text = parse_line(line)
            if img_name and text:
                self.samples.append((img_name, text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, text = self.samples[idx]
        img_path = self.img_dir / img_name

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None or img.shape[0] == 0:
            img = np.zeros((IMG_HEIGHT, 32), dtype=np.uint8)

        h, w = img.shape[:2]
        new_w = max(1, int(w * (IMG_HEIGHT / float(h))))
        if new_w > MAX_WIDTH:
            img = cv2.resize(img, (MAX_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
        else:
            img = cv2.resize(img, (new_w, IMG_HEIGHT), interpolation=cv2.INTER_AREA)

        img_float = img.astype(np.float32) / 255.0
        img_array = (img_float - 0.5) / 0.5

        target_indices = [self.char2idx.get(c, self.unk_idx) for c in text]
        return img_array, target_indices, len(target_indices)


def collate_fn_disk(batch):
    imgs_np, targets, target_lens = zip(*batch)
    max_w = max(img.shape[1] for img in imgs_np)
    batch_size = len(imgs_np)

    batch_imgs = torch.zeros(batch_size, 1, IMG_HEIGHT, max_w)
    for i, img_np in enumerate(imgs_np):
        h, w = img_np.shape
        batch_imgs[i, 0, :h, :w] = torch.from_numpy(img_np)

    flat_targets = [item for t in targets for item in t]
    return (
        batch_imgs,
        torch.tensor(flat_targets, dtype=torch.long),
        torch.tensor(target_lens, dtype=torch.long)
    )


def objective(trial: optuna.Trial) -> float:
    hidden_size = trial.suggest_categorical('hidden_size', [128, 256])
    num_layers = trial.suggest_int('num_layers', 1, 2)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-4, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])

    print(f"\n{'='*70}")
    print(f"🔬 TRIAL #{trial.number} | hidden={hidden_size}, layers={num_layers}, "
          f"bs={batch_size}, lr={learning_rate:.2e}, drop={dropout:.2f}")
    print(f"{'='*70}")

    char2idx, vocab = load_vocab()
    vocab_size = len(vocab)
    blank_idx = vocab.index('<BLANK>') if '<BLANK>' in vocab else 0

    train_dataset = HTRDiskDataset(global_train_lines, IMG_DIR, char2idx)
    val_dataset = HTRDiskDataset(global_val_lines, IMG_DIR, char2idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False, collate_fn=collate_fn_disk, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False, collate_fn=collate_fn_disk)

    model = CRNN(vocab_size=vocab_size, hidden_size=hidden_size,
                 num_layers=num_layers, dropout=dropout).to(DEVICE)

    criterion = CTCLoss(blank=blank_idx, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

    best_val_cer = float('inf')
    trial_start = time.time()

    try:
        for epoch in range(EPOCHS_PER_TRIAL):
            model.train()
            epoch_loss = 0
            valid_batches = 0
            epoch_start = time.time()

            pbar = tqdm(train_loader, desc=f"   Ep {epoch+1}/{EPOCHS_PER_TRIAL}", leave=False, unit="b", ncols=100)

            for batch_idx, (imgs, targets, target_lens) in enumerate(pbar):
                imgs = imgs.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)
                target_lens = target_lens.to(DEVICE, non_blocking=True)

                optimizer.zero_grad()

                if DEVICE.type == 'cuda':
                    with torch.amp.autocast(device_type='cuda'):
                        output = model(imgs)
                    output = output.float()
                else:
                    output = model(imgs)

                input_lens = torch.full((imgs.shape[0],), output.shape[0], dtype=torch.long, device=DEVICE)

                valid_mask = input_lens > target_lens
                if not valid_mask.all():
                    valid_indices = valid_mask.nonzero(as_tuple=True)[0]
                    if len(valid_indices) == 0:
                        continue
                    continue

                loss = criterion(output, targets, input_lens, target_lens)

                if not torch.isfinite(loss):
                    continue

                if DEVICE.type == 'cuda':
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

                epoch_loss += loss.item()
                valid_batches += 1
                pbar.set_postfix(loss=f"{loss.item():.3f}")

            if valid_batches == 0:
                print(f"   ⚠Эпоха {epoch+1}: все батчи были некорректными!")
                return float('inf')

            avg_loss = epoch_loss / valid_batches
            val_cer = validate(model, val_loader, vocab, DEVICE, blank_idx)
            scheduler.step(val_cer)

            epoch_duration = time.time() - epoch_start
            print(f"   Ep {epoch+1} | Loss: {avg_loss:.4f} | CER: {val_cer:.4f} | {format_time(epoch_duration)}")

            if val_cer < best_val_cer:
                best_val_cer = val_cer

        trial_duration = time.time() - trial_start
        print(f"   Trial #{trial.number} завершен за {format_time(trial_duration)} | Лучший CER: {best_val_cer:.4f}")
        return best_val_cer

    finally:
        del model, optimizer, scheduler, train_loader, val_loader, train_dataset, val_dataset
        gc.collect()
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()


def validate(model: nn.Module, val_loader, vocab: List[str], device, blank_idx: int) -> float:
    model.eval()
    total_cer, count = 0, 0

    with torch.no_grad():
        for imgs, targets, target_lens in tqdm(val_loader, desc="      Val", leave=False, unit="b"):
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
    global global_train_lines, global_val_lines

    print("ЗАПУСК ПОДБОРА ГИПЕРПАРАМЕТРОВ")

    print(f"\nАктивные пути из .env:")
    print(f"   Train GT : {TRAIN_GT}")
    print(f"   Val GT   : {VAL_GT}")
    print(f"   Vocab    : {VOCAB_PATH}")

    if not TRAIN_GT.exists() or not VAL_GT.exists():
        print("Файлы train.txt или val.txt не найдены!")
        print("   Пожалуйста, сначала запустите dataset_splitter.py")
        return

    print(f"\n[1/3] Загрузка словаря...")
    char2idx, vocab = load_vocab()
    print(f"   Размер словаря: {len(vocab)} символов")

    print(f"\n[2/3] Чтение путей из txt файлов в RAM (единожды)...")
    with open(TRAIN_GT, 'r', encoding='utf-8') as f:
        global_train_lines = [line.strip() for line in f if line.strip()]
    with open(VAL_GT, 'r', encoding='utf-8') as f:
        global_val_lines = [line.strip() for line in f if line.strip()]

    print(f"   Train строк: {len(global_train_lines):,} | Val строк: {len(global_val_lines):,}")

    print(f"\n[3/3] Запуск Optuna: {N_TRIALS} trials")
    print(f"   Режим: {'GPU' if DEVICE.type == 'cuda' else 'CPU'} | Потоков: {N_JOBS}")
    print(f"   Epochs per trial: {EPOCHS_PER_TRIAL}")

    study = optuna.create_study(direction='minimize', study_name='htr_crnn_disk_env')
    start_time = time.time()

    study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS, show_progress_bar=True)

    optuna_duration = time.time() - start_time
    print(f"\nОптимизация завершена за {format_time(optuna_duration)}")
    print(f"   Лучший CER: {study.best_value:.4f}")
    print(f"   ⚙Лучшие параметры: {study.best_params}")

    results_path = IMG_DIR / "optuna_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'best_cer': study.best_value,
            'best_params': study.best_params,
            'architecture': 'VGG_style_CNN_BiLSTM',
            'normalization': '[-1, 1]'
        }, f, indent=2, ensure_ascii=False)

    print(f"\nРезультаты успешно сохранены в: {results_path}")

if __name__ == "__main__":
    main()
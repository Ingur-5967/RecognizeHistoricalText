"""
Класс предсказания для ДОУЧЕННОЙ модели.
ОПТИМИЗИРОВАН ТОЛЬКО ДЛЯ РАСПОЗНАВАНИЯ ОДНОГО ИЗОБРАЖЕНИЯ.
Параметры модели автоматически загружаются из JSON, пути берутся из .env.
"""
import os
import json
import cv2
import numpy as np
import torch
from pathlib import Path
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

from model.model import CRNN
from metrics.metrics import decode_ctc_greedy

class FineTunedPredictor:
    """
    Класс для распознавания текста дообученной моделью.
    Автоматически подтягивает пути и гиперпараметры из .env и JSON.
    """

    def __init__(
            self,
            model_path: str = None,
            vocab_path: str = None,
            device: str = None,
            # Если передать значения здесь, они перезапишут значения из JSON
            hidden_size: int = None,
            num_layers: int = None,
            dropout: float = None,
            img_height: int = 32,
            max_width: int = 2048,
            min_width: int = 320,
    ):
        # 1. Определение путей (из аргументов или из .env, или дефолтные)
        self.model_path = Path(model_path or os.getenv("MODEL_BEST_VERSION_PATH", "C:/ml-output/artefact-model/best_model.pth"))
        self.vocab_path = Path(vocab_path or os.getenv("VOCAB_PATH", "C:/ml-output/extended/vocab_extended.json"))
        params_path = Path(os.getenv("MODEL_BEST_PARAMS_PATH", "C:/ml-output/artefact-model/model_best_params.json"))

        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))

        # Параметры предобработки (фиксированные)
        self.img_height = img_height
        self.max_width = max_width
        self.min_width = min_width

        # 2. Загрузка гиперпараметров с fallback на дефолтные значения
        default_params = {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.104}
        loaded_params = self._load_params_safely(params_path, default_params)

        # 3. Присвоение параметров (приоритет: аргументы > JSON > дефолт)
        self.hidden_size = hidden_size if hidden_size is not None else loaded_params['hidden_size']
        self.num_layers = num_layers if num_layers is not None else loaded_params['num_layers']
        self.dropout = dropout if dropout is not None else loaded_params['dropout']

        # 4. Валидация путей
        if not self.model_path.exists():
            raise FileNotFoundError(f"❌ Модель не найдена: {self.model_path}\nПроверьте путь в .env (MODEL_BEST_VERSION_PATH)")
        if not self.vocab_path.exists():
            raise FileNotFoundError(f"❌ Словарь не найден: {self.vocab_path}\nПроверьте путь в .env (VOCAB_PATH)")

        # 5. Инициализация компонентов
        self.char2idx, self.vocab = self._load_vocab()
        self.blank_idx = self.vocab.index('<BLANK>') if '<BLANK>' in self.vocab else 0
        self.model = self._load_model()

        print(f"✅ Предсказатель инициализирован")
        print(f"   🖥️  Устройство: {self.device}")
        print(f"    Словарь: {len(self.vocab)} симв. (BLANK={self.blank_idx})")

        # 🔥 ПРОВЕРКА: загружена ли модель в eval mode?
        print(f"    Model training: {self.model.training}")  # Должно быть False!

        # 🔥 ПРОВЕРКА: количество параметров
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"   ⚙️  Параметров: {total_params:,} (ожидалось: 5,572,675)")

    def _load_params_safely(self, params_path: Path, defaults: dict) -> dict:
        """Безопасная загрузка параметров из JSON с обработкой ошибок и пустых файлов."""
        if not params_path.exists():
            print(f"⚠️ Файл параметров не найден ({params_path}). Используются значения по умолчанию.")
            return defaults.copy()

        try:
            with open(params_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Если файл пустой или не словарь
            if not isinstance(data, dict) or not data:
                print(f"⚠️ Файл параметров пуст. Используются значения по умолчанию.")
                return defaults.copy()

            # Поддержка как плоской структуры {"hidden_size": 128},
            # так и структуры от Optuna {"best_params": {"hidden_size": 128}, "best_cer": 0.05}
            params = data.get('best_params', data)

            # Объединяем с дефолтными значениями на случай, если в JSON не хватает какого-то ключа
            final_params = defaults.copy()
            final_params.update(params)

            print(f"✅ Гиперпараметры успешно загружены из: {params_path.name}")
            return final_params

        except json.JSONDecodeError:
            print(f"⚠️ Ошибка чтения JSON ({params_path}). Файл поврежден. Используются значения по умолчанию.")
            return defaults.copy()
        except Exception as e:
            print(f"⚠️ Неожиданная ошибка при чтении параметров: {e}. Используются значения по умолчанию.")
            return defaults.copy()

    def _load_vocab(self):
        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        char2idx = data['char2idx']
        idx2char = data['idx2char']
        vocab = [idx2char[str(i)] for i in range(len(idx2char))]
        return char2idx, vocab

    def _load_model(self) -> CRNN:
        model = CRNN(
            vocab_size=len(self.vocab),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)

        # weights_only=True для безопасности
        state = torch.load(self.model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return model

    def preprocess_image(self, img_path: str) -> torch.Tensor:
        img_path = Path(img_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Изображение не найдено: {img_path}")

        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

        if len(img.shape) == 3:
            if img.shape[2] == 4:
                alpha = img[:, :, 3] / 255.0
                rgb = img[:, :, :3].astype(np.float32)
                bg = np.ones_like(rgb) * 255
                img = (rgb * alpha[:, :, np.newaxis] + bg * (1 - alpha[:, :, np.newaxis])).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        original_h, original_w = img.shape[:2]
        print(f"   📐 Исходный размер: {original_w}x{original_h}")

        # Апскейл для маленьких изображений
        if original_h < 100:
            target_h = 160
        elif original_h < 150:
            target_h = 128
        else:
            target_h = original_h

        if target_h != original_h:
            scale_factor = target_h / float(original_h)
            new_w = max(1, int(original_w * scale_factor))
            img = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            print(f"   🔍 Апскейл до: {new_w}x{target_h} (x{scale_factor:.2f})")

        # ❌ УБИРАЕМ бинаризацию и обрезку!
        # Только легкое повышение контраста
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

        # Ресайз до высоты модели
        h, w = img.shape[:2]
        new_w = max(1, int(w * (self.img_height / float(h))))
        new_w = min(new_w, self.max_width)
        img = cv2.resize(img, (new_w, self.img_height), interpolation=cv2.INTER_AREA)

        # Минимальный паддинг
        if img.shape[1] < 32:
            padded = np.zeros((img.shape[0], 32), dtype=img.dtype)
            padded[:, :img.shape[1]] = img
            img = padded

        # Нормализация [-1, 1]
        img_float = img.astype(np.float32) / 255.0
        img_norm = (img_float - 0.5) / 0.5

        return torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(self.device)
    @torch.no_grad()
    def predict(self, img_path: str) -> str:
        """Распознавание текста на одном изображении."""
        tensor = self.preprocess_image(img_path)
        output = self.model(tensor)

        # Greedy decoding для первого (и единственного) элемента батча
        pred_indices = decode_ctc_greedy(output[:, 0, :].cpu(), self.blank_idx)
        text = ''.join([self.vocab[idx] for idx in pred_indices if idx < len(self.vocab)])

        return text


# ======================== ПРИМЕР ИСПОЛЬЗОВАНИЯ ========================
if __name__ == '__main__':
    # 1. Инициализация БЕЗ явного указания путей (они возьмутся из .env и JSON автоматически!)
    # Если нужно переопределить, просто передайте аргументы: predictor = FineTunedPredictor(model_path="...")
    try:
        predictor = FineTunedPredictor()
    except FileNotFoundError as e:
        print(e)
        print("💡 Убедитесь, что файл .env настроен правильно и файлы существуют.")
        exit(1)

    # 2. Путь к картинке (ЗАМЕНИТЕ НА СВОЙ реальный путь для теста)
    test_img_path = "C:/Pisya/test_dataset/images/img_5.png"

    # 3. Предсказание
    if Path(test_img_path).exists():
        recognized_text = predictor.predict(test_img_path)
        print("\n" + "="*50)
        print(f"📄 Файл: {Path(test_img_path).name}")
        print(f"🔤 Распознанный текст: '{recognized_text}'")
        print("="*50)
    else:
        print(f"❌ Тестовый файл не найден: {test_img_path}")
        print("💡 Создайте тестовое изображение или измените путь в коде.")
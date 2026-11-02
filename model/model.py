import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network для распознавания текста.
    Архитектура: VGG-style CNN + BiLSTM + CTC Head
    """
    def __init__(self, vocab_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(CRNN, self).__init__()

        # CNN backbone, адаптированный для высоты 32.
        # Сжимает высоту в 32 раза (до 1), а ширину только в 4 раза.
        # Для W=512 на выходе будет W=128, что позволяет предсказывать тексты до ~120 символов.
        self.cnn = nn.Sequential(
            # Блок 1: H/2, W/2
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            # Блок 2: H/4, W/4
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            # Блок 3: H/4, W/4 (без пулинга для сохранения признаков)
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # Блок 4: H/8, W/4 (пулинг только по высоте)
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),

            # Блок 5: H/16, W/4 (пулинг только по высоте)
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),

            # Блок 6: H/32, W/4 (пулинг только по высоте, итоговая высота = 1)
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
        )

        # RNN: двунаправленный LSTM
        self.rnn = nn.LSTM(
            input_size=512,  # Выходной канал последнего слоя CNN
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=False,
            dropout=dropout if num_layers > 1 else 0
        )

        # Классификатор
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, vocab_size)  # *2 для bidirectional
        )

    def forward(self, x):
        """
        Args:
            x: тензор формы (Batch, 1, H, W)
        Returns:
            output: тензор формы (Time, Batch, Vocab)
        """
        # CNN feature extraction: (B, 1, H, W) -> (B, 512, 1, W//4)
        features = self.cnn(x)

        # Убираем измерение высоты (оно равно 1)
        # (B, 512, 1, W//4) -> (B, 512, W//4)
        features = features.squeeze(2)

        # Меняем оси для LSTM: (Time, Batch, Features)
        # (B, 512, W//4) -> (W//4, B, 512)
        features = features.permute(2, 0, 1)

        # RNN
        rnn_out, _ = self.rnn(features)  # (W//4, B, hidden*2)

        # Классификатор
        output = self.classifier(rnn_out)  # (W//4, B, vocab)

        # Log softmax для CTC
        output = F.log_softmax(output, dim=2)

        return output


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    model = CRNN(vocab_size=62, hidden_size=128, num_layers=2, dropout=0.3)
    total, trainable = count_parameters(model)
    print(f"Всего параметров: {total:,}")
    print(f"Обучаемых: {trainable:,}")

    # Тест forward pass для изображения 32x512
    x = torch.randn(4, 1, 32, 512)
    out = model(x)
    print(f"Вход:  {x.shape}")
    print(f"Выход: {out.shape} (Time=128, Batch=4, Vocab=62)")
from typing import List

import torch
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix


def decode_ctc_greedy(output: torch.Tensor, blank_idx: int) -> List[int]:
    pred_classes = torch.argmax(output, dim=1)  # (Time,)

    decoded = []
    prev_idx = -1

    for idx in pred_classes.cpu().tolist():
        if idx != blank_idx and idx != prev_idx:
            decoded.append(idx)
        prev_idx = idx

    return decoded

def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def calculate_cer(predicted: str, target: str) -> float:
    if len(target) == 0:
        return 0.0 if len(predicted) == 0 else 1.0
    distance = levenshtein_distance(predicted, target)
    return distance / len(target)


def calculate_wer(predicted: str, target: str) -> float:
    pred_words = predicted.split()
    target_words = target.split()
    if len(target_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    distance = levenshtein_distance(pred_words, target_words)
    return distance / len(target_words)


def calculate_f1_score(predicted_chars: list, target_chars: list, vocab: list) -> float:
    all_predicted = []
    all_target = []
    for pred, targ in zip(predicted_chars, target_chars):
        all_predicted.extend(list(pred))
        all_target.extend(list(targ))

    max_len = max(len(all_predicted), len(all_target))
    all_predicted.extend([''] * (max_len - len(all_predicted)))
    all_target.extend([''] * (max_len - len(all_target)))

    char2idx = {char: idx for idx, char in enumerate(vocab)}
    pred_indices = [char2idx.get(c, len(vocab)) for c in all_predicted]
    target_indices = [char2idx.get(c, len(vocab)) for c in all_target]

    return f1_score(target_indices, pred_indices, average='macro', zero_division=0)


def calculate_confusion_matrix(predicted_chars: list, target_chars: list, vocab: list) -> np.ndarray:
    all_predicted = []
    all_target = []
    for pred, targ in zip(predicted_chars, target_chars):
        all_predicted.extend(list(pred))
        all_target.extend(list(targ))

    max_len = max(len(all_predicted), len(all_target))
    all_predicted.extend([''] * (max_len - len(all_predicted)))
    all_target.extend([''] * (max_len - len(all_target)))

    char2idx = {char: idx for idx, char in enumerate(vocab)}
    pred_indices = [char2idx.get(c, len(vocab)) for c in all_predicted]
    target_indices = [char2idx.get(c, len(vocab)) for c in all_target]

    return confusion_matrix(target_indices, pred_indices, labels=range(len(vocab) + 1))


def calculate_r2_accuracy(predicted: str, target: str) -> float:
    if len(target) == 0:
        return 1.0 if len(predicted) == 0 else 0.0
    correct = sum(1 for p, t in zip(predicted, target) if p == t)
    return correct / len(target)
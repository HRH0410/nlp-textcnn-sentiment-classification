import ast
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch.utils.data import Dataset


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


LABEL_NAMES = {
    0: "very negative",
    1: "negative",
    2: "neutral",
    3: "positive",
    4: "very positive",
}


def load_vocab(tokens2id_path: str | Path) -> Dict[str, int]:
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    with open(tokens2id_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            token = row[0]
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def read_examples(csv_path: str | Path) -> List[Tuple[List[str], int]]:
    examples: List[Tuple[List[str], int]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tokens = ast.literal_eval(row["sentences"])
            label = int(row["label"])
            examples.append((tokens, label))
    return examples


def count_labels(examples: List[Tuple[List[str], int]], num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float)
    for _, label in examples:
        counts[label] += 1
    return counts


def make_class_weights(examples: List[Tuple[List[str], int]], num_classes: int) -> torch.Tensor:
    counts = count_labels(examples, num_classes)
    weights = counts.sum() / (num_classes * counts.clamp(min=1.0))
    return weights / weights.mean()


def encode_tokens(tokens: Iterable[str], vocab: Dict[str, int], max_len: int) -> Tuple[List[int], int]:
    ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokens]
    length = min(len(ids), max_len)
    ids = ids[:max_len]
    if len(ids) < max_len:
        ids.extend([vocab[PAD_TOKEN]] * (max_len - len(ids)))
    return ids, length


class SentimentDataset(Dataset):
    def __init__(self, csv_path: str | Path, vocab: Dict[str, int], max_len: int):
        self.examples = read_examples(csv_path)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        tokens, label = self.examples[index]
        input_ids, length = encode_tokens(tokens, self.vocab, self.max_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }

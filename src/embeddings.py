from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from src.data import PAD_TOKEN, UNK_TOKEN


def load_pretrained_embeddings(
    embedding_path: str | Path,
    vocab: Dict[str, int],
    embedding_dim: int,
    seed: int,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(loc=0.0, scale=0.05, size=(len(vocab), embedding_dim)).astype(np.float32)
    matrix[vocab[PAD_TOKEN]] = 0.0

    found = 0
    total = 0
    embedding_path = Path(embedding_path)
    with open(embedding_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) != embedding_dim + 1:
                continue
            token = parts[0]
            index = vocab.get(token)
            if index is None:
                continue
            vector = np.asarray(parts[1:], dtype=np.float32)
            matrix[index] = vector
            found += 1
            total += 1

    if vocab[UNK_TOKEN] == 1:
        known_indices = [idx for token, idx in vocab.items() if token not in {PAD_TOKEN, UNK_TOKEN}]
        if known_indices:
            matrix[vocab[UNK_TOKEN]] = matrix[known_indices].mean(axis=0)

    stats = {
        "vocab_size": len(vocab),
        "matched_tokens": found,
        "coverage": found / max(len(vocab) - 2, 1),
        "embedding_dim": embedding_dim,
        "embedding_path": str(embedding_path),
    }
    return torch.tensor(matrix, dtype=torch.float), stats

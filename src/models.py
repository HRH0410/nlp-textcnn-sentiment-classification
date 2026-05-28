from typing import Iterable, Sequence

import torch
from torch import nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """Kim-style CNN for sentence classification."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int,
        num_filters: int,
        filter_sizes: Sequence[int],
        dropout: float,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.convs = nn.ModuleList(
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=filter_size,
            )
            for filter_size in filter_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids).transpose(1, 2)
        pooled_outputs = []
        for conv in self.convs:
            features = F.relu(conv(embedded))
            pooled = F.max_pool1d(features, kernel_size=features.size(2)).squeeze(2)
            pooled_outputs.append(pooled)
        sentence_features = torch.cat(pooled_outputs, dim=1)
        return self.classifier(self.dropout(sentence_features))


class FastTextClassifier(nn.Module):
    """Embedding average baseline."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int,
        dropout: float,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = input_ids.ne(self.padding_idx).unsqueeze(-1)
        embedded = self.embedding(input_ids) * mask
        lengths = mask.sum(dim=1).clamp(min=1)
        averaged = embedded.sum(dim=1) / lengths
        return self.classifier(self.dropout(averaged))


def build_model(config: dict, vocab_size: int) -> nn.Module:
    model_name = config["model"]["name"].lower()
    common = {
        "vocab_size": vocab_size,
        "num_classes": config["data"]["num_classes"],
        "embedding_dim": config["model"]["embedding_dim"],
        "dropout": config["model"]["dropout"],
        "padding_idx": 0,
    }
    if model_name == "textcnn":
        return TextCNN(
            **common,
            num_filters=config["model"]["num_filters"],
            filter_sizes=config["model"]["filter_sizes"],
        )
    if model_name == "fasttext":
        return FastTextClassifier(**common)
    raise ValueError(f"Unsupported model: {config['model']['name']}")

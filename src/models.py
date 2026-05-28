from typing import Sequence

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
        pretrained_embeddings: torch.Tensor | None = None,
        freeze_embeddings: bool = False,
        use_static_channel: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        self.embedding.weight.requires_grad = not freeze_embeddings

        self.static_embedding = None
        input_dim = embedding_dim
        if use_static_channel:
            self.static_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
            if pretrained_embeddings is not None:
                self.static_embedding.weight.data.copy_(pretrained_embeddings)
            self.static_embedding.weight.requires_grad = False
            input_dim = embedding_dim * 2

        self.convs = nn.ModuleList(
            nn.Conv1d(
                in_channels=input_dim,
                out_channels=num_filters,
                kernel_size=filter_size,
            )
            for filter_size in filter_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        if self.static_embedding is not None:
            static_embedded = self.static_embedding(input_ids)
            embedded = torch.cat([embedded, static_embedded], dim=-1)
        embedded = embedded.transpose(1, 2)
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
        pretrained_embeddings: torch.Tensor | None = None,
        freeze_embeddings: bool = False,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        self.embedding.weight.requires_grad = not freeze_embeddings
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = input_ids.ne(self.padding_idx).unsqueeze(-1)
        embedded = self.embedding(input_ids) * mask
        lengths = mask.sum(dim=1).clamp(min=1)
        averaged = embedded.sum(dim=1) / lengths
        return self.classifier(self.dropout(averaged))


def build_model(
    config: dict,
    vocab_size: int,
    pretrained_embeddings: torch.Tensor | None = None,
) -> nn.Module:
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
            pretrained_embeddings=pretrained_embeddings,
            freeze_embeddings=config["model"].get("freeze_embeddings", False),
            use_static_channel=config["model"].get("use_static_channel", False),
        )
    if model_name == "fasttext":
        return FastTextClassifier(
            **common,
            pretrained_embeddings=pretrained_embeddings,
            freeze_embeddings=config["model"].get("freeze_embeddings", False),
        )
    raise ValueError(f"Unsupported model: {config['model']['name']}")

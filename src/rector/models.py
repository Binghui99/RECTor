"""Neural components described in the RECTor paper."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class GRUWindowEncoder(nn.Module):
    def __init__(self, hidden_size: int = 64, layers: int = 2, bidirectional: bool = False):
        super().__init__()
        self.bidirectional = bidirectional
        self.gru = nn.GRU(1, hidden_size, layers, batch_first=True, bidirectional=bidirectional)

    @property
    def output_size(self) -> int:
        return self.gru.hidden_size * (2 if self.bidirectional else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        if self.bidirectional:
            return torch.cat((hidden[-2], hidden[-1]), dim=-1)
        return hidden[-1]


class RECTorEncoder(nn.Module):
    """Shared GRU encoder with attention-based multiple-instance aggregation."""

    def __init__(self, hidden_size: int = 64, layers: int = 2, bidirectional: bool = False):
        super().__init__()
        self.window_encoder = GRUWindowEncoder(hidden_size, layers, bidirectional)
        width = self.window_encoder.output_size
        self.attention = nn.Sequential(nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))
        self.projection = nn.Linear(width, width)

    def forward(self, flows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, windows, length, channels = flows.shape
        if channels != 1:
            raise ValueError(f"expected one feature channel, received {channels}")
        encoded = self.window_encoder(flows.reshape(batch * windows, length, channels))
        encoded = encoded.reshape(batch, windows, -1)
        weights = torch.softmax(self.attention(encoded), dim=1)
        embedding = self.projection(torch.sum(weights * encoded, dim=1))
        return F.normalize(embedding, dim=-1), weights


def cosine_triplet_loss(
    anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float = 0.1
) -> torch.Tensor:
    positive_similarity = F.cosine_similarity(anchor, positive)
    negative_similarity = F.cosine_similarity(anchor, negative)
    return F.relu(negative_similarity - positive_similarity + margin).mean()


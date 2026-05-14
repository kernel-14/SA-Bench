import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(name: str) -> nn.Module:
    activations = {
        "relu": nn.ReLU(),
        "elu": nn.ELU(),
        "tanh": nn.Tanh(),
        "silu": nn.SiLU(),
        "gelu": nn.GELU(),
        "leaky_relu": nn.LeakyReLU(),
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name]


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: Tuple[int, ...],
        activation: str = "relu",
        output_activation: Optional[str] = None,
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(get_activation(activation))
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        if output_activation is not None:
            layers.append(get_activation(output_activation))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianHead(nn.Module):
    """MLP head predicting mean and log-std of a Gaussian distribution."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int,
        activation: str = "relu",
        min_std: float = 1e-4,
        max_std: float = 10.0,
    ):
        super().__init__()
        act = get_activation(activation)
        self.mean_head = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            act,
            nn.Linear(hidden_size, output_dim),
        )
        self.log_std_head = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            get_activation(activation),
            nn.Linear(hidden_size, output_dim),
        )
        self.min_log_std = math.log(min_std)
        self.max_log_std = math.log(max_std)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean_head(x)
        log_std = self.log_std_head(x).clamp(self.min_log_std, self.max_log_std)
        std = log_std.exp()
        return mean, std

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        mean, std = self.forward(x)
        eps = torch.randn_like(mean)
        return mean + std * eps

    def log_prob(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mean, std = self.forward(x)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(target).sum(dim=-1)

    def nll_loss(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(x, target).mean()


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class CausalTransformerDecoder(nn.Module):
    """Decoder-only transformer with causal masking."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        context_length: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.context_length = context_length
        self.register_buffer(
            "causal_mask",
            nn.Transformer.generate_square_subsequent_mask(context_length),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        mask = self.causal_mask[:seq_len, :seq_len]
        return self.transformer(x, mask=mask, is_causal=True)


class GRUEncoder(nn.Module):
    """Multi-layer GRU encoder."""

    def __init__(self, input_dim: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output, h_n = self.gru(x, h)
        return output, h_n

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)


class StraightThroughEstimator(torch.autograd.Function):
    """Straight-through estimator for discrete sampling."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x.round()

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
        return grad


def straight_through_round(x: torch.Tensor) -> torch.Tensor:
    return StraightThroughEstimator.apply(x)

"""Shared neural network modules used across the codebase."""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: List[int],
    activation: str = "relu",
    output_activation: Optional[str] = None,
) -> nn.Module:
    """Build a Multi-Layer Perceptron with specified hidden dimensions."""
    layers: List[nn.Module] = []
    dims = [input_dim] + hidden_dims + [output_dim]
    act_fn = _get_activation(activation)

    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act_fn)
        elif output_activation is not None:
            layers.append(_get_activation(output_activation))

    return nn.Sequential(*layers)


def _get_activation(name: str) -> nn.Module:
    if name.lower() == "relu":
        return nn.ReLU()
    elif name.lower() == "elu":
        return nn.ELU()
    elif name.lower() == "tanh":
        return nn.Tanh()
    elif name.lower() == "sigmoid":
        return nn.Sigmoid()
    elif name.lower() == "leaky_relu":
        return nn.LeakyReLU()
    else:
        raise ValueError(f"Unknown activation: {name}")


class GaussianHead(nn.Module):
    """Predicts mean and log-standard-deviation of a Gaussian distribution.

    Outputs: mean (output_dim) and log_std (output_dim).
    The log_std is a learned parameter independent of input or computed via a small MLP.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        activation: str = "relu",
        min_std: float = 0.001,
        max_std: float = 10.0,
    ):
        super().__init__()
        self.min_log_std = torch.log(torch.tensor(min_std))
        self.max_log_std = torch.log(torch.tensor(max_std))

        self.mean_net = build_mlp(
            input_dim, output_dim, [hidden_dim], activation=activation
        )
        self.log_std_net = build_mlp(
            input_dim, output_dim, [hidden_dim], activation=activation
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean_net(x)
        log_std = self.log_std_net(x)
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        return mean, log_std


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer baseline."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1), :].unsqueeze(0)


class RecurrentStateSpaceModel(nn.Module):
    """Recurrent State-Space Model (RSSM) as used in PlaNet/Dreamer.

    Architecture follows Table S8:
      - GRU hidden size 256, 2 layers
      - Latent dimension 64, categorical with 32 categories
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        latent_dim: int = 64,
        num_categories: int = 32,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.num_categories = num_categories

        cat_dim = num_categories * latent_dim

        # Encoder: observation → latent posterior
        self.obs_encoder = build_mlp(
            obs_dim, 2 * cat_dim, [hidden_size], activation="relu"
        )

        # RNN cell updates hidden state
        self.rnn = nn.GRU(
            input_size=action_dim + cat_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Prior: hidden → latent prior
        self.prior_net = build_mlp(
            hidden_size, 2 * cat_dim, [hidden_size], activation="relu"
        )

        # Observation decoder: hidden + latent → observation
        self.obs_decoder = build_mlp(
            hidden_size + cat_dim, 2 * obs_dim, [hidden_size], activation="relu"
        )

    def encode_obs(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode observation to posterior mean/logit."""
        out = self.obs_encoder(obs)
        mean, logit = torch.chunk(out, 2, dim=-1)
        return mean, logit

    def get_prior(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.prior_net(h)
        mean, logit = torch.chunk(out, 2, dim=-1)
        return mean, logit

    def decode_obs(
        self, h: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.obs_decoder(torch.cat([h, z], dim=-1))
        mean, log_std = torch.chunk(out, 2, dim=-1)
        return mean, log_std

    def sample_latent(
        self, mean: torch.Tensor, logit: torch.Tensor, temperature: float = 1.0
    ) -> torch.Tensor:
        """Sample one-hot categorical latent."""
        batch_shape = mean.shape[:-1]
        cat_dim = self.num_categories * self.latent_dim
        mean_reshaped = mean.view(*batch_shape, self.latent_dim, self.num_categories)
        logits_reshaped = logit.view(*batch_shape, self.latent_dim, self.num_categories)
        probs = F.softmax(logits_reshaped / temperature, dim=-1)
        sample = torch.multinomial(
            probs.view(-1, self.num_categories), num_samples=1
        ).view(*batch_shape, self.latent_dim)
        onehot = F.one_hot(sample, num_classes=self.num_categories).float()
        return onehot.view(*batch_shape, cat_dim)

    def forward_step(
        self,
        prev_h: torch.Tensor,
        action: torch.Tensor,
        prev_z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single step of RSSM: prior, hidden state, posterior latent."""
        batch_size = action.shape[0]
        rnn_input = torch.cat([action, prev_z], dim=-1).unsqueeze(1)  # (B, 1, D)
        h, new_hidden = self.rnn(rnn_input, prev_h)  # h: (B, 1, H)
        h_flat = h.squeeze(1)  # (B, H)

        prior_mean, prior_logit = self.get_prior(h_flat)
        return prior_mean, prior_logit, h_flat, new_hidden

    def encode_step(
        self, h: torch.Tensor, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate posterior latent from hidden state and observation."""
        posterior_mean, posterior_logit = self.encode_obs(obs)
        return posterior_mean, posterior_logit

    def decode_step(
        self, h: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.decode_obs(h, z)

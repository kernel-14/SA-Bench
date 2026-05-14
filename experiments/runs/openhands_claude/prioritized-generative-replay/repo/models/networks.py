import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal timestep embedding for diffusion models."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class ResidualBlock(nn.Module):
    """Residual MLP block used in the diffusion denoising network."""

    def __init__(self, dim: int, time_embed_dim: int, cond_embed_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.time_proj = nn.Linear(time_embed_dim, dim)
        self.cond_proj = nn.Linear(cond_embed_dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.act = nn.Mish()

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        cond_emb: torch.Tensor,
    ) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act(self.fc1(h))
        h = h + self.time_proj(self.act(time_emb))
        h = h + self.cond_proj(self.act(cond_emb))
        h = self.norm2(h)
        h = self.fc2(h)
        return x + h


class ResidualMLP(nn.Module):
    """Residual MLP denoising network for the conditional diffusion model.

    Architecture follows SYNTHER (Lu et al., 2024): residual MLP blocks
    conditioned on diffusion timestep and relevance condition c.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_hidden_layers: int = 4,
        time_embed_dim: int = 128,
        cond_embed_dim: int = 128,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.Mish(),
        )
        self.cond_embed = nn.Sequential(
            nn.Linear(1, cond_embed_dim),
            nn.Mish(),
            nn.Linear(cond_embed_dim, cond_embed_dim),
            nn.Mish(),
        )
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, time_embed_dim, cond_embed_dim)
            for _ in range(n_hidden_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        h = self.input_proj(x)
        time_emb = self.time_embed(t)
        cond_emb = self.cond_embed(cond)
        for block in self.blocks:
            h = block(h, time_emb, cond_emb)
        return self.output_proj(h)


class MLP(nn.Module):
    """Standard multi-layer perceptron."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        activation: str = "relu",
        output_activation: Optional[str] = None,
        layer_norm: bool = False,
    ):
        super().__init__()
        act_fn = {"relu": nn.ReLU, "tanh": nn.Tanh, "mish": nn.Mish, "elu": nn.ELU}[activation]
        layers: List[nn.Module] = []
        in_dim = input_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_fn())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        if output_activation is not None:
            out_act_fn = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}[output_activation]
            layers.append(out_act_fn())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NoisyLinear(nn.Module):
    """Noisy linear layer for NoisyNets exploration (Fortunato et al., 2018).

    Replaces y = wx + b with y = (μ^w + σ^w ⊙ ε^w)x + μ^b + σ^b ⊙ ε^b.
    Initialization follows Section 3.2 of Fortunato et al. (2018).
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.sample_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def sample_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class NoisyMLP(nn.Module):
    """MLP with NoisyLinear layers for exploration."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        sigma_init: float = 0.5,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim
        for _ in range(n_hidden):
            layers.append(NoisyLinear(in_dim, hidden_dim, sigma_init))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(NoisyLinear(in_dim, output_dim, sigma_init))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def sample_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.sample_noise()


class CNNEncoder(nn.Module):
    """CNN visual encoder for pixel-based observations (DRQ-V2 style).

    Used for pixel-based DMC tasks. Generates latent representations
    f_θ(s) for diffusion model training in latent space.
    """

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        feature_dim: int = 50,
    ):
        super().__init__()
        assert len(obs_shape) == 3, "obs_shape must be (C, H, W)"
        c, h, w = obs_shape

        self.convs = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        conv_out_dim = self._get_conv_out(obs_shape)
        self.fc = nn.Linear(conv_out_dim, feature_dim)
        self.ln = nn.LayerNorm(feature_dim)

    def _get_conv_out(self, shape: Tuple[int, ...]) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, *shape)
            out = self.convs(dummy)
            return int(np.prod(out.shape[1:]))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.convs(obs)
        h = h.view(h.size(0), -1)
        h = self.ln(self.fc(h))
        return h


class RNDCNNEncoder(nn.Module):
    """Three-layer CNN encoder for RND relevance function (pixel-based tasks).

    Architecture: 3-layer CNN with bottleneck latent dim 64, feature output 512,
    followed by 2-layer MLP projection of dimension 512.
    Per Appendix A.1 of the paper.
    """

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        latent_dim: int = 64,
        output_dim: int = 512,
    ):
        super().__init__()
        c, h, w = obs_shape
        self.convs = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, latent_dim, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        conv_out_dim = self._get_conv_out(obs_shape)
        self.mlp = nn.Sequential(
            nn.Linear(conv_out_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def _get_conv_out(self, shape: Tuple[int, ...]) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, *shape)
            out = self.convs(dummy)
            return int(np.prod(out.shape[1:]))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.convs(obs)
        h = h.view(h.size(0), -1)
        return self.mlp(h)


class ResNet18Encoder(nn.Module):
    """ResNet-18 encoder for ECO relevance function.

    Output dimension 512, followed by a four-layer MLP with feature and
    output dimensions of 512. Per Appendix A.2 of the paper.
    """

    def __init__(self, output_dim: int = 512):
        super().__init__()
        import torchvision.models as models
        resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.mlp = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.backbone(obs)
        h = h.view(h.size(0), -1)
        return self.mlp(h)


class GaussianActor(nn.Module):
    """Gaussian policy network for SAC/REDQ.

    Outputs mean and log_std of a Gaussian distribution over actions.
    """

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        noisy: bool = False,
    ):
        super().__init__()
        if noisy:
            self.trunk = NoisyMLP(state_dim, hidden_dim, hidden_dim, n_hidden - 1)
        else:
            self.trunk = MLP(state_dim, hidden_dim, hidden_dim, n_hidden - 1)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(state)
        mean = self.mean_layer(h)
        log_std = self.log_std_layer(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def get_action(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(state)
        if deterministic:
            action = torch.tanh(mean)
            log_prob = torch.zeros(state.shape[0], 1, device=state.device)
        else:
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            x = dist.rsample()
            action = torch.tanh(x)
            log_prob = dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


class QNetwork(nn.Module):
    """Q-value network for SAC/REDQ."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        noisy: bool = False,
    ):
        super().__init__()
        if noisy:
            self.net = NoisyMLP(state_dim + action_dim, 1, hidden_dim, n_hidden)
        else:
            self.net = MLP(state_dim + action_dim, 1, hidden_dim, n_hidden)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))

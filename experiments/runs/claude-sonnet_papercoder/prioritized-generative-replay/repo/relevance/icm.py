## relevance/icm.py
"""Intrinsic Curiosity Module (ICM) relevance function for Prioritized Generative Replay (PGR).

Implements the primary relevance function F(s, a, s', r) = 0.5 * ||g(h(s), a) - h(s')||²
from Section 4.2 of the paper (Pathak et al., 2017). High prediction error indicates
transitions the agent has not yet mastered — ideal candidates for densified generation.

Architecture:
    - Feature encoder h: MLP (state-based) or CNN+MLP (pixel-based)
    - Forward dynamics model g: MLP predicting h(s') from (h(s), a)

Both networks are trained jointly via a single Adam optimizer, with h(s') detached
as a fixed target to prevent encoder collapse.

Config references (config.yaml):
    relevance.icm.latent_dim:        256   # encoder output dimension
    relevance.icm.hidden_dim:        256   # MLP hidden width
    relevance.icm.num_layers:        2     # number of hidden layers
    relevance.icm.lr:                1e-3  # Adam learning rate
    relevance.icm.cnn_bottleneck_dim: 64   # pixel encoder bottleneck channels
    relevance.icm.cnn_feature_dim:   512   # pixel encoder feature dimension
    relevance.icm.cnn_num_layers:    3     # pixel encoder conv layers
    relevance.icm.mlp_proj_dim:      512   # pixel encoder MLP projection dim
    relevance.icm.mlp_proj_layers:   2     # pixel encoder MLP projection depth
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from relevance.base import BaseRelevance


# ── Sub-network: MLP Feature Encoder ─────────────────────────────────────────


class MLPEncoder(nn.Module):
    """Multi-layer perceptron feature encoder h mapping observations to latent space.

    Architecture: input_dim → [hidden_dim → ReLU] × num_layers → output_dim
    No activation on the final output layer — raw latent features are returned.

    Used as the state-based encoder h in ICMRelevance when use_cnn=False.

    Attributes:
        input_dim: Dimension of the input observation vector.
        hidden_dim: Width of each hidden layer.
        output_dim: Dimension of the output latent feature vector (latent_dim).
        num_layers: Number of hidden layers (not counting input/output projections).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 2,
    ) -> None:
        """Initialises the MLP encoder.

        Builds a sequential MLP with num_layers hidden layers of width hidden_dim,
        each followed by ReLU activation. The final linear layer maps to output_dim
        with no activation.

        Args:
            input_dim: Input feature dimension. For state-based tasks this is
                obs_dim (e.g. 67 for quadruped-walk). For pixel tasks where the
                ICM receives DRQv2 latents, this is feature_dim (e.g. 50).
            hidden_dim: Width of each hidden layer. Corresponds to
                config.relevance.icm.hidden_dim (default 256).
            output_dim: Latent feature dimension. Corresponds to
                config.relevance.icm.latent_dim (default 256).
            num_layers: Number of hidden layers. Corresponds to
                config.relevance.icm.num_layers (default 2).
        """
        super().__init__()

        self.input_dim: int = input_dim
        self.hidden_dim: int = hidden_dim
        self.output_dim: int = output_dim
        self.num_layers: int = num_layers

        # Build layer list: input projection + num_layers hidden layers + output projection.
        layers: List[nn.Module] = []

        # Input projection: input_dim → hidden_dim with ReLU.
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        # Hidden layers: hidden_dim → hidden_dim with ReLU.
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Output projection: hidden_dim → output_dim, no activation.
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network: nn.Sequential = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes an observation batch into latent feature vectors.

        Args:
            x: Float32 tensor of shape (B, input_dim).

        Returns:
            Float32 tensor of shape (B, output_dim) — raw latent features
            with no activation applied to the final layer.
        """
        return self.network(x)


# ── Sub-network: CNN Feature Encoder (pixel-based tasks) ─────────────────────


class CNNEncoder(nn.Module):
    """Convolutional feature encoder h for pixel-based observations.

    Architecture (from paper Appendix A):
        3 × Conv(kernel=3, stride=2) layers with ReLU activations
        → Flatten
        → Linear projection to cnn_feature_dim
        → 2-layer MLP projection to latent_dim

    Used when ICMRelevance is instantiated with use_cnn=True, i.e. when the
    ICM receives raw pixel observations (e.g. DMLab with PPO backbone) rather
    than pre-encoded DRQv2 latents.

    Config references:
        relevance.icm.cnn_bottleneck_dim: 64   (intermediate conv channels)
        relevance.icm.cnn_feature_dim:   512   (linear projection output)
        relevance.icm.cnn_num_layers:    3     (number of conv layers)
        relevance.icm.mlp_proj_dim:      512   (MLP projection hidden dim)
        relevance.icm.mlp_proj_layers:   2     (MLP projection depth)

    Attributes:
        in_channels: Number of input image channels (e.g. 9 for 3-frame stack).
        image_size: Spatial resolution of input frames (e.g. 84).
        bottleneck_dim: Number of channels in the final conv layer.
        feature_dim: Output dimension of the linear projection after flatten.
        latent_dim: Final output dimension after MLP projection.
    """

    def __init__(
        self,
        in_channels: int = 9,
        image_size: int = 84,
        bottleneck_dim: int = 64,
        feature_dim: int = 512,
        cnn_num_layers: int = 3,
        mlp_proj_dim: int = 512,
        mlp_proj_layers: int = 2,
        latent_dim: int = 256,
    ) -> None:
        """Initialises the CNN encoder.

        Args:
            in_channels: Number of input channels. For frame-stacked pixel
                observations: frame_stack * 3 (e.g. 9 for 3 stacked RGB frames).
            image_size: Spatial resolution (height = width) of input frames.
                Corresponds to config.env.image_size (default 84).
            bottleneck_dim: Number of output channels in the final conv layer.
                Corresponds to config.relevance.icm.cnn_bottleneck_dim (default 64).
            feature_dim: Output dimension of the linear projection after flatten.
                Corresponds to config.relevance.icm.cnn_feature_dim (default 512).
            cnn_num_layers: Number of convolutional layers.
                Corresponds to config.relevance.icm.cnn_num_layers (default 3).
            mlp_proj_dim: Hidden dimension of the MLP projection.
                Corresponds to config.relevance.icm.mlp_proj_dim (default 512).
            mlp_proj_layers: Number of hidden layers in the MLP projection.
                Corresponds to config.relevance.icm.mlp_proj_layers (default 2).
            latent_dim: Final output dimension after MLP projection.
                Corresponds to config.relevance.icm.latent_dim (default 256).
        """
        super().__init__()

        self.in_channels: int = in_channels
        self.image_size: int = image_size
        self.bottleneck_dim: int = bottleneck_dim
        self.feature_dim: int = feature_dim
        self.latent_dim: int = latent_dim

        # ── Convolutional backbone ────────────────────────────────────────────
        # Build cnn_num_layers conv layers with stride=2 (halves spatial dims).
        # Channel progression: in_channels → 32 → 64 → bottleneck_dim
        conv_layers: List[nn.Module] = []
        channel_sizes: List[int] = [in_channels]

        # Intermediate channels: 32 for all layers except the last.
        for i in range(cnn_num_layers):
            if i == 0:
                out_ch = 32
            elif i == cnn_num_layers - 1:
                out_ch = bottleneck_dim
            else:
                out_ch = 64

            conv_layers.append(
                nn.Conv2d(
                    channel_sizes[-1],
                    out_ch,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            )
            conv_layers.append(nn.ReLU())
            channel_sizes.append(out_ch)

        self.conv_backbone: nn.Sequential = nn.Sequential(*conv_layers)

        # ── Compute flattened spatial dimension after conv layers ─────────────
        # Each stride=2 conv halves the spatial dimension.
        spatial_size: int = image_size
        for _ in range(cnn_num_layers):
            spatial_size = (spatial_size + 2 * 1 - 3) // 2 + 1  # formula for stride=2, pad=1, k=3
        flat_dim: int = bottleneck_dim * spatial_size * spatial_size

        # ── Linear projection: flat_dim → feature_dim ────────────────────────
        self.linear_proj: nn.Linear = nn.Linear(flat_dim, feature_dim)

        # ── MLP projection: feature_dim → latent_dim ─────────────────────────
        mlp_layers: List[nn.Module] = []
        mlp_layers.append(nn.Linear(feature_dim, mlp_proj_dim))
        mlp_layers.append(nn.ReLU())
        for _ in range(mlp_proj_layers - 1):
            mlp_layers.append(nn.Linear(mlp_proj_dim, mlp_proj_dim))
            mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Linear(mlp_proj_dim, latent_dim))
        self.mlp_proj: nn.Sequential = nn.Sequential(*mlp_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes a batch of pixel observations into latent feature vectors.

        Args:
            x: Float32 tensor of shape (B, in_channels, H, W) with pixel
                values normalized to [0, 1] or raw uint8 values. The caller
                is responsible for normalization.

        Returns:
            Float32 tensor of shape (B, latent_dim).
        """
        # Normalize to [0, 1] if values appear to be in uint8 range.
        if x.max() > 1.0:
            x = x.float() / 255.0

        # Conv backbone: (B, C, H, W) → (B, bottleneck_dim, H', W')
        features: torch.Tensor = self.conv_backbone(x)

        # Flatten spatial dimensions: (B, bottleneck_dim, H', W') → (B, flat_dim)
        features = features.view(features.size(0), -1)

        # Linear projection: (B, flat_dim) → (B, feature_dim)
        features = F.relu(self.linear_proj(features))

        # MLP projection: (B, feature_dim) → (B, latent_dim)
        return self.mlp_proj(features)


# ── Sub-network: MLP Forward Dynamics Model ──────────────────────────────────


class MLPForwardModel(nn.Module):
    """MLP forward dynamics model g predicting h(s') from (h(s), a).

    Architecture: (latent_dim + action_dim) → [hidden_dim → ReLU] × num_layers → latent_dim
    No activation on the final output layer.

    The input is the concatenation of the current latent state h(s) and the
    action a. The output is the predicted next latent state h(s').

    Attributes:
        latent_dim: Dimension of the latent state space (encoder output).
        action_dim: Dimension of the action space.
        hidden_dim: Width of each hidden layer.
        num_layers: Number of hidden layers.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        action_dim: int = 6,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ) -> None:
        """Initialises the forward dynamics model.

        Args:
            latent_dim: Dimension of the latent state space. Must match the
                output_dim of the paired MLPEncoder or CNNEncoder.
                Corresponds to config.relevance.icm.latent_dim (default 256).
            action_dim: Action space dimension. Inferred from the environment
                at PGRTrainer init time (e.g. 12 for quadruped-walk).
            hidden_dim: Width of each hidden layer. Corresponds to
                config.relevance.icm.hidden_dim (default 256).
            num_layers: Number of hidden layers. Corresponds to
                config.relevance.icm.num_layers (default 2).
        """
        super().__init__()

        self.latent_dim: int = latent_dim
        self.action_dim: int = action_dim
        self.hidden_dim: int = hidden_dim
        self.num_layers: int = num_layers

        # Input dimension: concatenation of latent state and action.
        input_dim: int = latent_dim + action_dim

        # Build layer list: input projection + num_layers hidden layers + output projection.
        layers: List[nn.Module] = []

        # Input projection: (latent_dim + action_dim) → hidden_dim with ReLU.
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        # Hidden layers: hidden_dim → hidden_dim with ReLU.
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Output projection: hidden_dim → latent_dim, no activation.
        layers.append(nn.Linear(hidden_dim, latent_dim))

        self.network: nn.Sequential = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predicts the next latent state given current latent state and action.

        Args:
            latent: Float32 tensor of shape (B, latent_dim) — encoded current
                observation h(s).
            action: Float32 tensor of shape (B, action_dim) — action taken.

        Returns:
            Float32 tensor of shape (B, latent_dim) — predicted next latent
            state ĥ(s') = g(h(s), a). No activation on output.
        """
        # Concatenate latent state and action along feature dimension.
        x: torch.Tensor = torch.cat([latent, action], dim=1)
        return self.network(x)


# ── Main Class: ICMRelevance ──────────────────────────────────────────────────


class ICMRelevance(BaseRelevance):
    """Intrinsic Curiosity Module relevance function for PGR.

    Implements F(s, a, s', r) = 0.5 * ||g(h(s), a) - h(s')||² from Section 4.2
    of the paper, following the ICM design of Pathak et al. (2017).

    High prediction error indicates transitions in regions of state space the
    agent has not yet mastered — these are the most valuable for densified
    generation via the conditional diffusion model.

    The encoder h and forward model g are trained jointly via a single Adam
    optimizer. The target h(s') is detached during training to prevent encoder
    collapse (the encoder cannot trivially satisfy the loss by mapping all
    observations to the same point).

    Score normalization to [0, 1] is NOT performed here — it is the
    responsibility of PGRTrainer, which normalizes per inner loop call using
    the min/max of all scores in D_real (Shared Knowledge point 3).

    Attributes:
        latent_dim: Encoder output dimension (config.relevance.icm.latent_dim).
        hidden_dim: MLP hidden width (config.relevance.icm.hidden_dim).
        num_layers: Number of hidden layers (config.relevance.icm.num_layers).
        use_cnn: Whether to use a CNN encoder for pixel observations.
        encoder: Feature encoder h (MLPEncoder or CNNEncoder).
        forward_model: Forward dynamics model g (MLPForwardModel).
        optimizer: Joint Adam optimizer over encoder + forward_model parameters.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        lr: float = 1e-3,
        device: str = "cuda",
        use_cnn: bool = False,
        cnn_config: Optional[Dict] = None,
    ) -> None:
        """Initialises the ICM relevance function.

        Instantiates the feature encoder h and forward dynamics model g,
        moves both to the target device, and creates a single Adam optimizer
        over all learnable parameters.

        Args:
            obs_dim: Flat observation dimension. For state-based tasks: the
                concatenated dm_control observation vector size (e.g. 67 for
                quadruped-walk). For pixel tasks with DRQv2: the CNN latent
                dimension (feature_dim=50). For pixel tasks with raw pixels
                (use_cnn=True): frame_stack * 3 * H * W (flattened).
                Corresponds to the obs_dim inferred from the environment
                wrapper at PGRTrainer init time.
            action_dim: Action space dimension (e.g. 12 for quadruped-walk,
                6 for Walker2d-v2). Inferred from the environment wrapper.
            latent_dim: Encoder output dimension. Corresponds to
                config.relevance.icm.latent_dim (default 256).
            hidden_dim: MLP hidden layer width. Corresponds to
                config.relevance.icm.hidden_dim (default 256).
            num_layers: Number of hidden layers in both encoder and forward
                model. Corresponds to config.relevance.icm.num_layers (default 2).
            lr: Adam optimizer learning rate. Corresponds to
                config.relevance.icm.lr (default 1e-3).
            device: PyTorch device string. Corresponds to
                config.hardware.device (default "cuda").
            use_cnn: If True, uses CNNEncoder for pixel observations (e.g.
                DMLab with PPO backbone). If False (default), uses MLPEncoder
                for state vectors or pre-encoded DRQv2 latents.
            cnn_config: Optional dict of CNN encoder hyperparameters. If None
                and use_cnn=True, defaults from config.yaml are used:
                    {
                        'in_channels': 9,
                        'image_size': 84,
                        'bottleneck_dim': 64,
                        'feature_dim': 512,
                        'cnn_num_layers': 3,
                        'mlp_proj_dim': 512,
                        'mlp_proj_layers': 2,
                    }
        """
        # Initialize BaseRelevance (which calls nn.Module.__init__).
        super().__init__(obs_dim=obs_dim, action_dim=action_dim, device=device)

        self.latent_dim: int = latent_dim
        self.hidden_dim: int = hidden_dim
        self.num_layers: int = num_layers
        self.use_cnn: bool = use_cnn

        # ── Instantiate feature encoder h ─────────────────────────────────────
        if use_cnn:
            # Pixel-based encoder: CNN + MLP projection.
            # Merge provided cnn_config with defaults from config.yaml.
            _cnn_cfg: Dict = {
                "in_channels": 9,           # frame_stack=3 * 3 channels
                "image_size": 84,           # config.env.image_size
                "bottleneck_dim": 64,       # config.relevance.icm.cnn_bottleneck_dim
                "feature_dim": 512,         # config.relevance.icm.cnn_feature_dim
                "cnn_num_layers": 3,        # config.relevance.icm.cnn_num_layers
                "mlp_proj_dim": 512,        # config.relevance.icm.mlp_proj_dim
                "mlp_proj_layers": 2,       # config.relevance.icm.mlp_proj_layers
                "latent_dim": latent_dim,
            }
            if cnn_config is not None:
                _cnn_cfg.update(cnn_config)

            self.encoder: nn.Module = CNNEncoder(
                in_channels=_cnn_cfg["in_channels"],
                image_size=_cnn_cfg["image_size"],
                bottleneck_dim=_cnn_cfg["bottleneck_dim"],
                feature_dim=_cnn_cfg["feature_dim"],
                cnn_num_layers=_cnn_cfg["cnn_num_layers"],
                mlp_proj_dim=_cnn_cfg["mlp_proj_dim"],
                mlp_proj_layers=_cnn_cfg["mlp_proj_layers"],
                latent_dim=latent_dim,
            )
        else:
            # State-based encoder: plain MLP.
            self.encoder: nn.Module = MLPEncoder(
                input_dim=obs_dim,
                hidden_dim=hidden_dim,
                output_dim=latent_dim,
                num_layers=num_layers,
            )

        # ── Instantiate forward dynamics model g ──────────────────────────────
        self.forward_model: MLPForwardModel = MLPForwardModel(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        # ── Move both networks to the target device ───────────────────────────
        # BaseRelevance inherits from nn.Module, so self.to(device) moves all
        # registered submodules (encoder, forward_model) to the device.
        self.to(device)

        # ── Create joint Adam optimizer ───────────────────────────────────────
        # Both encoder and forward model are trained jointly — the encoder
        # learns features that make forward prediction easier, following the
        # original ICM design (Pathak et al., 2017).
        self.optimizer: torch.optim.Adam = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.forward_model.parameters()),
            lr=lr,
        )

    # ── Private helper ────────────────────────────────────────────────────────

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encodes an observation batch into latent feature vectors.

        Thin wrapper around the encoder forward pass. Gradient flow is
        controlled by the calling context (torch.no_grad() in score(),
        normal forward pass in update()).

        Args:
            obs: Float32 tensor of shape (B, obs_dim) for state-based tasks,
                or (B, C, H, W) for pixel-based tasks (use_cnn=True).
                Must be on self.device.

        Returns:
            Float32 tensor of shape (B, latent_dim) — encoded latent features.
        """
        return self.encoder(obs)

    # ── BaseRelevance interface ───────────────────────────────────────────────

    def score(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        reward: torch.Tensor,
    ) -> torch.Tensor:
        """Computes per-transition ICM curiosity scores.

        Implements F(s, a, s', r) = 0.5 * ||g(h(s), a) - h(s')||² from
        Section 4.2 of the paper. The squared L2 norm is computed over the
        latent dimension (sum, not mean) to match the paper's ||·||² notation.

        The entire forward pass runs under torch.no_grad() for efficiency —
        scores are used as data labels (conditioning signals for the diffusion
        model and values stored in the replay buffer), not for backpropagation.

        Args:
            obs: Current observations, float32 tensor of shape (B, obs_dim).
                Must be on self.device.
            action: Actions taken, float32 tensor of shape (B, action_dim).
                Must be on self.device.
            next_obs: Next observations, float32 tensor of shape (B, obs_dim).
                Must be on self.device.
            reward: Rewards received, float32 tensor of shape (B, 1) or (B,).
                Not used by ICM but required by the BaseRelevance interface.
                Must be on self.device.

        Returns:
            Float32 tensor of shape (B, 1) containing per-transition curiosity
            scores (raw squared prediction errors). Detached from the
            computation graph. Values are unnormalized — normalization to
            [0, 1] is performed by PGRTrainer per inner loop call.
        """
        # Move inputs to device with float32 dtype.
        obs, action, next_obs = self.to_device(obs, action, next_obs)

        # All forward passes under no_grad — scores are labels, not gradients.
        with torch.no_grad():
            # Encode current and next observations.
            latent_s: torch.Tensor = self._encode(obs)           # (B, latent_dim)
            latent_s_prime: torch.Tensor = self._encode(next_obs)  # (B, latent_dim)

            # Predict next latent state from current latent and action.
            predicted_latent: torch.Tensor = self.forward_model(
                latent_s, action
            )  # (B, latent_dim)

            # Compute per-sample squared L2 norm: 0.5 * ||ĥ(s') - h(s')||²
            # .sum(dim=1) computes the squared norm over the latent dimension,
            # matching the paper's ||·||² notation (not mean over latent dim).
            # keepdim=True gives shape (B, 1) for consistent buffer storage.
            error: torch.Tensor = 0.5 * (
                (predicted_latent - latent_s_prime) ** 2
            ).sum(dim=1, keepdim=True)  # (B, 1)

        return error  # Already detached due to torch.no_grad() context.

    def update(self, batch: Dict[str, torch.Tensor]) -> float:
        """Performs one gradient step to update the ICM encoder and forward model.

        Called by PGRTrainer._update_relevance_scores() every
        config.relevance.update_freq=20 policy gradient steps (5% of all
        policy steps, per Section 5 of the paper).

        The training loss is the mean squared prediction error over the batch:
            L = 0.5 * mean(||g(h(s), a) - h(s').detach()||²)

        The target h(s') is detached to prevent encoder collapse — without
        this, the encoder could trivially satisfy the loss by mapping all
        observations to the same point, destroying the curiosity signal.

        Args:
            batch: Transition dict sampled from D_real by PGRTrainer.
                Contains float32 tensors on self.device with keys:
                    'observations':      (B, obs_dim)
                    'actions':           (B, action_dim)
                    'next_observations': (B, obs_dim)
                    'rewards':           (B, 1)
                    'dones':             (B, 1)
                Batch size corresponds to config.sampling.batch_size (default 256).

        Returns:
            Scalar ICM training loss as a Python float (via loss.item()).
            Logged by PGRTrainer as "relevance/icm_loss" for monitoring
            training stability.
        """
        # Extract and move tensors to device.
        obs: torch.Tensor = batch["observations"].to(
            device=self.device, dtype=torch.float32
        )
        action: torch.Tensor = batch["actions"].to(
            device=self.device, dtype=torch.float32
        )
        next_obs: torch.Tensor = batch["next_observations"].to(
            device=self.device, dtype=torch.float32
        )

        # ── Forward pass with gradients enabled ──────────────────────────────
        # Encode current observation — gradients flow through encoder.
        latent_s: torch.Tensor = self._encode(obs)  # (B, latent_dim)

        # Encode next observation — DETACH to use as a fixed target.
        # This prevents the encoder from collapsing by making h(s') trivially
        # predictable. Follows the ICM design of Pathak et al. (2017).
        latent_s_prime: torch.Tensor = self._encode(next_obs).detach()  # (B, latent_dim)

        # Predict next latent state from current latent and action.
        predicted_latent: torch.Tensor = self.forward_model(
            latent_s, action
        )  # (B, latent_dim)

        # Compute mean squared prediction error over batch and latent dims.
        # Using .mean() (not .sum()) gives a stable scalar loss regardless of
        # batch size or latent dimension — important for consistent lr tuning.
        loss: torch.Tensor = 0.5 * (
            (predicted_latent - latent_s_prime) ** 2
        ).mean()

        # ── Gradient step ─────────────────────────────────────────────────────
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def __repr__(self) -> str:
        """Returns a concise string representation of the ICM relevance function."""
        encoder_type: str = "CNN" if self.use_cnn else "MLP"
        return (
            f"ICMRelevance("
            f"obs_dim={self.obs_dim}, "
            f"action_dim={self.action_dim}, "
            f"latent_dim={self.latent_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers}, "
            f"encoder={encoder_type}, "
            f"device='{self.device}')"
        )

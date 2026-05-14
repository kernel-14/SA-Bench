## relevance/rnd.py
"""Random Network Distillation (RND) relevance function for Prioritized Generative Replay (PGR).

Implements the RND relevance function F(s, a, s', r) = 0.5 * ||f̂_θ(s') - f(s')||²
from Appendix A of the paper (Burda et al., 2018). A fixed randomly-initialized
target network f and a trainable predictor network f̂_θ share the same architecture.
High prediction error indicates novel transitions the predictor has not yet seen —
ideal candidates for densified generation via the conditional diffusion model.

Architecture (from paper Appendix A, pixel-based tasks):
    - 3-layer CNN with bottleneck dim 64 and feature output dim 512
    - Followed by a 2-layer MLP projection of dimension 512

For state-based tasks, both networks use an MLP consistent with the ICM encoder.

Config references (config.yaml):
    relevance.rnd.cnn_bottleneck_dim: 64    # CNN bottleneck channels
    relevance.rnd.cnn_feature_dim:   512    # linear projection output dim
    relevance.rnd.cnn_num_layers:    3      # number of conv layers
    relevance.rnd.mlp_proj_dim:      512    # MLP projection hidden dim
    relevance.rnd.mlp_proj_layers:   2      # MLP projection depth
    relevance.rnd.lr:                1e-3   # predictor Adam learning rate
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from relevance.base import BaseRelevance


# ── Sub-network: CNN Encoder (pixel-based tasks) ──────────────────────────────


class _CNNEncoder(nn.Module):
    """Convolutional feature encoder for pixel-based RND.

    Architecture (paper Appendix A):
        3 × Conv2d(kernel=3, stride=2, padding=1) with ReLU activations
        → Flatten
        → Linear projection to cnn_feature_dim (512)
        → 2-layer MLP projection of width mlp_proj_dim (512)

    Both the target network f and the predictor f̂_θ use this class with
    independent random initializations.

    Attributes:
        in_channels: Number of input image channels (frame_stack * 3).
        image_size: Spatial resolution of input frames (H = W).
        cnn_bottleneck_dim: Number of output channels in the final conv layer.
        cnn_feature_dim: Output dimension of the linear projection after flatten.
        cnn_num_layers: Number of convolutional layers.
        mlp_proj_dim: Hidden and output dimension of the MLP projection.
        mlp_proj_layers: Number of hidden layers in the MLP projection.
    """

    def __init__(
        self,
        in_channels: int = 9,
        image_size: int = 84,
        cnn_bottleneck_dim: int = 64,
        cnn_feature_dim: int = 512,
        cnn_num_layers: int = 3,
        mlp_proj_dim: int = 512,
        mlp_proj_layers: int = 2,
    ) -> None:
        """Initialises the CNN encoder.

        Args:
            in_channels: Number of input channels. For frame-stacked pixel
                observations: frame_stack * 3 (e.g. 9 for 3 stacked RGB frames).
                Corresponds to config.env.frame_stack * 3.
            image_size: Spatial resolution (height = width) of input frames.
                Corresponds to config.env.image_size (default 84).
            cnn_bottleneck_dim: Number of output channels in the final conv layer.
                Corresponds to config.relevance.rnd.cnn_bottleneck_dim (default 64).
            cnn_feature_dim: Output dimension of the linear projection after flatten.
                Corresponds to config.relevance.rnd.cnn_feature_dim (default 512).
            cnn_num_layers: Number of convolutional layers.
                Corresponds to config.relevance.rnd.cnn_num_layers (default 3).
            mlp_proj_dim: Hidden and output dimension of the MLP projection.
                Corresponds to config.relevance.rnd.mlp_proj_dim (default 512).
            mlp_proj_layers: Number of hidden layers in the MLP projection.
                Corresponds to config.relevance.rnd.mlp_proj_layers (default 2).
        """
        super().__init__()

        self.in_channels: int = in_channels
        self.image_size: int = image_size
        self.cnn_bottleneck_dim: int = cnn_bottleneck_dim
        self.cnn_feature_dim: int = cnn_feature_dim
        self.cnn_num_layers: int = cnn_num_layers
        self.mlp_proj_dim: int = mlp_proj_dim
        self.mlp_proj_layers: int = mlp_proj_layers

        # ── Convolutional backbone ────────────────────────────────────────────
        # Channel progression: in_channels → 32 → 64 → cnn_bottleneck_dim
        # Each conv uses stride=2 to halve spatial dimensions.
        conv_layers: List[nn.Module] = []
        in_ch: int = in_channels

        for i in range(cnn_num_layers):
            if i == cnn_num_layers - 1:
                # Final conv layer outputs cnn_bottleneck_dim channels.
                out_ch: int = cnn_bottleneck_dim
            elif i == 0:
                out_ch = 32
            else:
                out_ch = 64

            conv_layers.append(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            )
            conv_layers.append(nn.ReLU())
            in_ch = out_ch

        self.conv_backbone: nn.Sequential = nn.Sequential(*conv_layers)

        # ── Compute flattened spatial dimension after conv layers ─────────────
        # Each stride=2 conv with kernel=3, padding=1 halves spatial dims:
        # out_size = floor((in_size + 2*pad - kernel) / stride) + 1
        #          = floor((in_size + 2*1 - 3) / 2) + 1
        #          = floor((in_size - 1) / 2) + 1
        spatial_size: int = image_size
        for _ in range(cnn_num_layers):
            spatial_size = (spatial_size - 1) // 2 + 1

        flat_dim: int = cnn_bottleneck_dim * spatial_size * spatial_size

        # ── Linear projection: flat_dim → cnn_feature_dim ────────────────────
        self.linear_proj: nn.Linear = nn.Linear(flat_dim, cnn_feature_dim)

        # ── MLP projection: cnn_feature_dim → mlp_proj_dim ───────────────────
        # Builds mlp_proj_layers hidden layers of width mlp_proj_dim.
        # Input: cnn_feature_dim (512), output: mlp_proj_dim (512).
        mlp_layers: List[nn.Module] = []
        mlp_in_dim: int = cnn_feature_dim

        for i in range(mlp_proj_layers):
            mlp_layers.append(nn.Linear(mlp_in_dim, mlp_proj_dim))
            # ReLU after all layers except the final output layer.
            if i < mlp_proj_layers - 1:
                mlp_layers.append(nn.ReLU())
            mlp_in_dim = mlp_proj_dim

        self.mlp_proj: nn.Sequential = nn.Sequential(*mlp_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes a batch of pixel observations into feature vectors.

        Handles both raw uint8 pixel inputs (values in [0, 255]) and
        pre-normalized float inputs (values in [0, 1]).

        Args:
            x: Float32 or uint8 tensor of shape (B, in_channels, H, W).
                Must be on the same device as the network parameters.

        Returns:
            Float32 tensor of shape (B, mlp_proj_dim) — encoded feature
            vectors with no activation on the final output layer.
        """
        # Normalize to [0, 1] if values appear to be in uint8 range.
        x_f: torch.Tensor = x.float()
        if x_f.max() > 1.0:
            x_f = x_f / 255.0

        # Conv backbone: (B, C, H, W) → (B, cnn_bottleneck_dim, H', W')
        features: torch.Tensor = self.conv_backbone(x_f)

        # Flatten spatial dimensions: (B, cnn_bottleneck_dim, H', W') → (B, flat_dim)
        features = features.view(features.size(0), -1)

        # Linear projection: (B, flat_dim) → (B, cnn_feature_dim) with ReLU.
        features = F.relu(self.linear_proj(features))

        # MLP projection: (B, cnn_feature_dim) → (B, mlp_proj_dim)
        return self.mlp_proj(features)


# ── Sub-network: MLP Encoder (state-based tasks) ─────────────────────────────


class _MLPEncoder(nn.Module):
    """Multi-layer perceptron feature encoder for state-based RND.

    Architecture: input_dim → [hidden_dim → ReLU] × num_layers → output_dim
    No activation on the final output layer.

    Used as both the target network f and predictor f̂_θ when use_cnn=False.

    Attributes:
        input_dim: Dimension of the input observation vector.
        hidden_dim: Width of each hidden layer.
        output_dim: Dimension of the output feature vector.
        num_layers: Number of hidden layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 2,
    ) -> None:
        """Initialises the MLP encoder.

        Args:
            input_dim: Input feature dimension. For state-based tasks this is
                obs_dim (e.g. 67 for quadruped-walk, 17 for cheetah-run).
                For pixel tasks with DRQv2 pre-encoding, this is feature_dim=50.
            hidden_dim: Width of each hidden layer. Corresponds to
                config.relevance.icm.hidden_dim (default 256) — reused for RND
                state-based tasks since config.yaml does not specify a separate
                RND MLP hidden dim.
            output_dim: Output feature dimension. Corresponds to
                config.relevance.icm.latent_dim (default 256).
            num_layers: Number of hidden layers (default 2).
        """
        super().__init__()

        self.input_dim: int = input_dim
        self.hidden_dim: int = hidden_dim
        self.output_dim: int = output_dim
        self.num_layers: int = num_layers

        # Build layer list: input projection + (num_layers - 1) hidden layers + output.
        layers: List[nn.Module] = []

        # Input projection: input_dim → hidden_dim with ReLU.
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        # Additional hidden layers: hidden_dim → hidden_dim with ReLU.
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Output projection: hidden_dim → output_dim, no activation.
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network: nn.Sequential = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes an observation batch into feature vectors.

        Args:
            x: Float32 tensor of shape (B, input_dim).

        Returns:
            Float32 tensor of shape (B, output_dim) — raw feature vectors
            with no activation applied to the final layer.
        """
        return self.network(x)


# ── Main Class: RNDRelevance ──────────────────────────────────────────────────


class RNDRelevance(BaseRelevance):
    """Random Network Distillation relevance function for PGR.

    Implements F(s, a, s', r) = 0.5 * ||f̂_θ(s') - f(s')||² from Appendix A
    of the paper (Burda et al., 2018). The score depends only on the next
    observation s', making it more robust to the noisy-TV problem than
    prediction-error-based ICM (which depends on the full transition dynamics).

    The target network f is randomly initialized and permanently frozen.
    The predictor f̂_θ is trained to minimize the MSE with respect to f's
    outputs. Regions of observation space that f̂_θ has not yet learned to
    predict (novel states) receive high scores — ideal for densified generation.

    Score normalization to [0, 1] is NOT performed here — it is the
    responsibility of PGRTrainer, which normalizes per inner loop call using
    the min/max of all scores in D_real (Shared Knowledge point 3 in the
    design document).

    Attributes:
        latent_dim: Output dimension of the MLP encoder (state-based tasks).
        hidden_dim: MLP hidden layer width (state-based tasks).
        num_layers: Number of hidden layers in the MLP encoder.
        use_cnn: Whether to use a CNN encoder for pixel observations.
        target_network: Fixed randomly-initialized network f (frozen).
        predictor_network: Trainable predictor network f̂_θ.
        optimizer: Adam optimizer over predictor_network parameters only.
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        lr: float = 1e-3,
        use_cnn: bool = False,
        device: str = "cuda",
        cnn_config: Optional[Dict] = None,
    ) -> None:
        """Initialises the RND relevance function.

        Instantiates both the target network f and predictor f̂_θ with the
        same architecture but independent random initializations. Freezes
        the target network immediately after construction. Creates an Adam
        optimizer over the predictor's parameters only.

        Args:
            obs_dim: Flat observation dimension. For state-based tasks: the
                concatenated dm_control observation vector size (e.g. 67 for
                quadruped-walk). For pixel tasks with DRQv2 pre-encoding:
                feature_dim=50. For raw pixel tasks (use_cnn=True): this
                argument is ignored — the CNN encoder handles the input shape
                via cnn_config.
            latent_dim: Output dimension of the MLP encoder (state-based tasks).
                Corresponds to config.relevance.icm.latent_dim (default 256).
                For CNN-based tasks, the output dimension is mlp_proj_dim (512).
            hidden_dim: MLP hidden layer width (state-based tasks). Corresponds
                to config.relevance.icm.hidden_dim (default 256).
            num_layers: Number of hidden layers in the MLP encoder (state-based).
                Corresponds to config.relevance.icm.num_layers (default 2).
            lr: Adam optimizer learning rate for the predictor network.
                Corresponds to config.relevance.rnd.lr (default 1e-3).
            use_cnn: If True, uses _CNNEncoder for pixel observations (e.g.
                DMLab with PPO backbone or raw pixel inputs). If False (default),
                uses _MLPEncoder for state vectors or pre-encoded DRQv2 latents.
                Set to True only when the policy does NOT pre-encode observations
                (i.e., config.policy.type != "drqv2").
            device: PyTorch device string. Corresponds to
                config.hardware.device (default "cuda").
            cnn_config: Optional dict of CNN encoder hyperparameters. If None
                and use_cnn=True, defaults from config.yaml are used:
                    {
                        'in_channels': 9,           # frame_stack=3 * 3 channels
                        'image_size': 84,           # config.env.image_size
                        'cnn_bottleneck_dim': 64,   # config.relevance.rnd.cnn_bottleneck_dim
                        'cnn_feature_dim': 512,     # config.relevance.rnd.cnn_feature_dim
                        'cnn_num_layers': 3,        # config.relevance.rnd.cnn_num_layers
                        'mlp_proj_dim': 512,        # config.relevance.rnd.mlp_proj_dim
                        'mlp_proj_layers': 2,       # config.relevance.rnd.mlp_proj_layers
                    }
        """
        # Initialize BaseRelevance (which calls nn.Module.__init__).
        # RND does not use action_dim internally, but the base class requires it.
        # Pass 0 as a sentinel value — it is stored but never used by this class.
        super().__init__(obs_dim=obs_dim, action_dim=0, device=device)

        self.latent_dim: int = latent_dim
        self.hidden_dim: int = hidden_dim
        self.num_layers: int = num_layers
        self.use_cnn: bool = use_cnn

        # ── Build CNN config with defaults from config.yaml ───────────────────
        _cnn_cfg: Dict = {
            "in_channels": 9,           # frame_stack=3 * 3 channels
            "image_size": 84,           # config.env.image_size
            "cnn_bottleneck_dim": 64,   # config.relevance.rnd.cnn_bottleneck_dim
            "cnn_feature_dim": 512,     # config.relevance.rnd.cnn_feature_dim
            "cnn_num_layers": 3,        # config.relevance.rnd.cnn_num_layers
            "mlp_proj_dim": 512,        # config.relevance.rnd.mlp_proj_dim
            "mlp_proj_layers": 2,       # config.relevance.rnd.mlp_proj_layers
        }
        if cnn_config is not None:
            _cnn_cfg.update(cnn_config)

        # ── Instantiate target network f (frozen) ─────────────────────────────
        if use_cnn:
            self.target_network: nn.Module = _CNNEncoder(
                in_channels=_cnn_cfg["in_channels"],
                image_size=_cnn_cfg["image_size"],
                cnn_bottleneck_dim=_cnn_cfg["cnn_bottleneck_dim"],
                cnn_feature_dim=_cnn_cfg["cnn_feature_dim"],
                cnn_num_layers=_cnn_cfg["cnn_num_layers"],
                mlp_proj_dim=_cnn_cfg["mlp_proj_dim"],
                mlp_proj_layers=_cnn_cfg["mlp_proj_layers"],
            )
        else:
            self.target_network = _MLPEncoder(
                input_dim=obs_dim,
                hidden_dim=hidden_dim,
                output_dim=latent_dim,
                num_layers=num_layers,
            )

        # ── Instantiate predictor network f̂_θ (trainable) ────────────────────
        # Same architecture class, different random initialization (PyTorch
        # initializes weights randomly by default for each new instance).
        if use_cnn:
            self.predictor_network: nn.Module = _CNNEncoder(
                in_channels=_cnn_cfg["in_channels"],
                image_size=_cnn_cfg["image_size"],
                cnn_bottleneck_dim=_cnn_cfg["cnn_bottleneck_dim"],
                cnn_feature_dim=_cnn_cfg["cnn_feature_dim"],
                cnn_num_layers=_cnn_cfg["cnn_num_layers"],
                mlp_proj_dim=_cnn_cfg["mlp_proj_dim"],
                mlp_proj_layers=_cnn_cfg["mlp_proj_layers"],
            )
        else:
            self.predictor_network = _MLPEncoder(
                input_dim=obs_dim,
                hidden_dim=hidden_dim,
                output_dim=latent_dim,
                num_layers=num_layers,
            )

        # ── Freeze target network ─────────────────────────────────────────────
        # Two safeguards:
        # 1. Set requires_grad=False on all parameters — prevents any accidental
        #    gradient accumulation even if the optimizer were misconfigured.
        # 2. Set to eval() mode permanently — disables dropout/batchnorm
        #    stochasticity (not present in our architecture, but defensive).
        for param in self.target_network.parameters():
            param.requires_grad = False
        self.target_network.eval()

        # ── Move both networks to the target device ───────────────────────────
        # BaseRelevance inherits from nn.Module, so self.to(device) moves all
        # registered submodules (target_network, predictor_network) to device.
        self.to(device)

        # ── Create Adam optimizer for predictor only ──────────────────────────
        # The target network has requires_grad=False on all parameters, so
        # even if accidentally included, it would not receive gradient updates.
        # We explicitly pass only predictor parameters for clarity and safety.
        self.optimizer: torch.optim.Adam = torch.optim.Adam(
            self.predictor_network.parameters(),
            lr=lr,
        )

    # ── BaseRelevance interface ───────────────────────────────────────────────

    def score(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        reward: torch.Tensor,
    ) -> torch.Tensor:
        """Computes per-transition RND novelty scores.

        Implements F(s, a, s', r) = 0.5 * ||f̂_θ(s') - f(s')||² from
        Appendix A of the paper. Only next_obs (s') is used; obs, action,
        and reward are accepted to satisfy the BaseRelevance interface but
        are ignored internally.

        The entire forward pass runs under torch.no_grad() for efficiency —
        scores are used as data labels (conditioning signals for the diffusion
        model and values stored in the replay buffer), not for backpropagation.

        Args:
            obs: Current observations, float32 tensor of shape (B, obs_dim).
                Accepted for interface compliance but not used by RND.
                Must be on self.device.
            action: Actions taken, float32 tensor of shape (B, action_dim).
                Accepted for interface compliance but not used by RND.
                Must be on self.device.
            next_obs: Next observations, float32 tensor of shape (B, obs_dim)
                for state-based tasks, or (B, C, H, W) for pixel tasks with
                use_cnn=True. This is the only input used by RND.
                Must be on self.device.
            reward: Rewards received, float32 tensor of shape (B, 1) or (B,).
                Accepted for interface compliance but not used by RND.
                Must be on self.device.

        Returns:
            Float32 tensor of shape (B, 1) containing per-transition RND
            novelty scores (raw squared prediction errors). Detached from
            the computation graph. Values are unnormalized — normalization
            to [0, 1] is performed by PGRTrainer per inner loop call.
        """
        # Move next_obs to device with float32 dtype.
        next_obs_f: torch.Tensor = next_obs.to(
            device=self.device, dtype=torch.float32
        )

        # All forward passes under no_grad — scores are labels, not gradients.
        with torch.no_grad():
            # Compute target features — frozen network, no gradient.
            target_feat: torch.Tensor = self.target_network(next_obs_f)  # (B, D)

            # Compute predictor features — also no gradient in score() context.
            pred_feat: torch.Tensor = self.predictor_network(next_obs_f)  # (B, D)

            # Per-sample squared L2 norm: 0.5 * ||pred - target||²
            # .sum(dim=-1) computes the squared norm over the feature dimension,
            # matching the paper's ||·||² notation (not mean over feature dim).
            # keepdim=True gives shape (B, 1) for consistent buffer storage.
            error: torch.Tensor = 0.5 * (
                (pred_feat - target_feat) ** 2
            ).sum(dim=-1, keepdim=True)  # (B, 1)

        return error  # Already detached due to torch.no_grad() context.

    def update(self, batch: Dict[str, torch.Tensor]) -> float:
        """Performs one gradient step to update the RND predictor network.

        Called by PGRTrainer._update_relevance_scores() every
        config.relevance.update_freq=20 policy gradient steps (5% of all
        policy steps, per Section 5 of the paper).

        The training loss is the mean squared prediction error between the
        predictor and the frozen target network outputs on next_obs:
            L_RND = 0.5 * mean(||f̂_θ(s') - f(s').detach()||²)

        The target output is detached to prevent any gradient flow into the
        frozen target network (which has requires_grad=False, but .detach()
        provides an additional explicit safeguard).

        Only next_obs is used from the batch — obs, actions, rewards, and
        dones are ignored by RND.

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
            Scalar RND training loss as a Python float (via loss.item()).
            Logged by PGRTrainer as "relevance/rnd_loss" for monitoring
            predictor training stability.
        """
        # Extract and move next_obs to device with float32 dtype.
        next_obs: torch.Tensor = batch["next_observations"].to(
            device=self.device, dtype=torch.float32
        )

        # ── Set predictor to training mode ────────────────────────────────────
        # Target network stays in eval() mode permanently (set in __init__).
        self.predictor_network.train()

        # ── Compute target features (no gradient) ────────────────────────────
        # The target network has requires_grad=False on all parameters.
        # We additionally use torch.no_grad() and .detach() for explicit safety.
        with torch.no_grad():
            target_feat: torch.Tensor = self.target_network(next_obs)

        # Detach target features — explicit safeguard against any accidental
        # gradient flow into the frozen target network.
        target_feat = target_feat.detach()

        # ── Compute predictor features (gradient flows here) ─────────────────
        pred_feat: torch.Tensor = self.predictor_network(next_obs)  # (B, D)

        # ── Compute MSE loss ──────────────────────────────────────────────────
        # F.mse_loss computes mean((pred - target)^2) over all elements.
        # We scale by 0.5 to match the paper's 0.5 * ||·||² formulation.
        # Using F.mse_loss (mean reduction) gives a stable scalar loss
        # regardless of batch size or feature dimension.
        loss: torch.Tensor = 0.5 * F.mse_loss(pred_feat, target_feat)

        # ── Gradient step ─────────────────────────────────────────────────────
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def __repr__(self) -> str:
        """Returns a concise string representation of the RND relevance function."""
        encoder_type: str = "CNN" if self.use_cnn else "MLP"
        return (
            f"RNDRelevance("
            f"obs_dim={self.obs_dim}, "
            f"latent_dim={self.latent_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers}, "
            f"encoder={encoder_type}, "
            f"device='{self.device}')"
        )

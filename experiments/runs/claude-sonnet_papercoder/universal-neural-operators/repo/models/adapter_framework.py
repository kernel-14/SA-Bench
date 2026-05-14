## Code: models/adapter_framework.py

```python
## models/adapter_framework.py
"""
Adapter-based multi-physics neural operator pretraining framework described in:
  "Towards Universal Neural Operators through Multiphysics Pretraining"

Implements the central adapter routing system (Section 3 of the paper):

    "In the pre-training phase the entire parameters set
    (θ_{P_1}, ..., θ_{P_N}, θ_F, θ_{L_1}, ..., θ_{L_N}) is subject to
    optimization."

    "In the fine-tuning stage we fix the parameters θ_F... only the new
    adapter parameters (θ_{P_ft}, θ_{L_ft}) are trained."

    "The lift and proj blocks are considered as the adapters... they are
    introduced to contain different cardinality input sets, projecting into
    the fixed number of hidden features."

Architecture:
    Input a → LiftingAdapter[physics_id] → FNO/MambaFNO/PerceiverNO/CodaNO
              → ProjectionAdapter[physics_id] → Output u

Classes:
  LiftingAdapter     - problem-specific input projection (θ_{L_i})
  ProjectionAdapter  - problem-specific output projection (θ_{P_i})
  AdapterFramework   - routing + freeze/unfreeze + checkpoint management

Tensor layout convention (Shared Knowledge #1):
  Channel-first: [B, C, L] for 1D, [B, C, H, W] for 2D.
  LiftingAdapter and ProjectionAdapter apply MLPs pointwise over spatial dims.

Physics ID strings (Shared Knowledge #2):
  No dots — use 'p' for decimal point:
    'burgers_nu0p01', 'gray_scott_F0p035_k0p065', 'heat_alpha0p01'
  Validated on register_adapter() — dots raise ValueError.

Checkpoint format (Shared Knowledge #4):
  {
    'model_state_dict': state_dict,
    'adapter_registry': {physics_id: {'n_in': int, 'n_out': int}},
    'backbone_type': str,
    'hidden_dim': int,
    'epoch': int or None,
    'val_loss': float or None,
  }

Config alignment (config.yaml):
  adapter.lifting_hidden_multiplier: 2   -> intermediate MLP width multiplier
  adapter.projection_hidden_multiplier: 2
  training.finetune.freeze_backbone: true -> enforced via freeze_backbone()

Dependencies: torch, torch.nn, os, typing, logging.
Backbone imports: models/fno_backbone.py, models/mamba_fno.py,
                  models/perceiver_no.py, models/coda_no.py.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Backbone imports — all four backbone types for checkpoint reconstruction
# ---------------------------------------------------------------------------

from models.fno_backbone import FNOBackbone
from models.coda_no import CodaNO
from models.perceiver_no import PerceiverNO

# MambaFNO import is guarded because mamba_ssm requires CUDA.
try:
    from models.mamba_fno import MambaFNO
    _MAMBA_AVAILABLE: bool = True
except ImportError:
    MambaFNO = None  # type: ignore[assignment, misc]
    _MAMBA_AVAILABLE = False

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

_logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backbone registry for checkpoint reconstruction
# ---------------------------------------------------------------------------

# Maps backbone class name (as stored in checkpoint) to the class itself.
# Used by load_checkpoint to validate backbone type consistency.
_BACKBONE_REGISTRY: Dict[str, type] = {
    "FNOBackbone": FNOBackbone,
    "PerceiverNO": PerceiverNO,
    "CodaNO": CodaNO,
}
if _MAMBA_AVAILABLE and MambaFNO is not None:
    _BACKBONE_REGISTRY["MambaFNO"] = MambaFNO

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# MLP hidden width multiplier for lifting and projection adapters.
# From config.yaml: adapter.lifting_hidden_multiplier: 2
# and adapter.projection_hidden_multiplier: 2
_LIFTING_HIDDEN_MULTIPLIER: int = 2
_PROJECTION_HIDDEN_MULTIPLIER: int = 2


# ---------------------------------------------------------------------------
# LiftingAdapter
# ---------------------------------------------------------------------------


class LiftingAdapter(nn.Module):
    """Problem-specific lifting adapter (θ_{L_i}).

    Maps problem-specific input functions ``a`` with ``n_in`` channels to
    the fixed hidden dimension ``hidden_dim`` used by the shared backbone.
    This is the ``L`` component in the paper's pipeline:

        a → LiftingAdapter → v_0 (lifted features, hidden_dim channels)

    Architecture (2-layer MLP applied pointwise over spatial locations):
        Linear(n_in, hidden_dim * 2) → GELU → Linear(hidden_dim * 2, hidden_dim)

    The intermediate width ``hidden_dim * 2`` follows the
    ``adapter.lifting_hidden_multiplier: 2`` setting in config.yaml.

    Tensor layout:
        Input:  [B, n_in, *spatial]   (channel-first, any spatial dims)
        Output: [B, hidden_dim, *spatial]

    The MLP is applied pointwise: spatial dimensions are temporarily moved
    to the batch dimension, the MLP is applied to the channel dimension,
    then spatial dimensions are restored. This is equivalent to a 1×1
    convolution but implemented as a linear layer for clarity.

    Padding handling (Shared Knowledge #5):
        During multi-physics pretraining, the collate_fn zero-pads inputs
        to the maximum n_in in the batch. This adapter only reads its first
        ``n_in`` channels: ``a[:, :self.n_in, ...]``. Padding channels are
        silently ignored.

    Attributes:
        n_in: Number of input channels (physics-specific).
        hidden_dim: Output channel count (shared backbone dimension).
        physics_id: Physics identifier string for logging/identification.
        _net: Sequential MLP: Linear → GELU → Linear.

    Example::

        adapter = LiftingAdapter(n_in=2, hidden_dim=128,
                                 physics_id='gray_scott_F0p035_k0p065')
        a = torch.randn(8, 2, 64, 64)   # [B, n_in, H, W]
        v0 = adapter(a)                  # [B, 128, 64, 64]
    """

    def __init__(
        self,
        n_in: int,
        hidden_dim: int,
        physics_id: str,
    ) -> None:
        """Initialise LiftingAdapter.

        Args:
            n_in: Number of input channels. Physics-specific. Examples:
                1 for Burgers/Advection, 2 for Gray-Scott/RD,
                3 for heat+convection.
            hidden_dim: Output channel count. Must match the backbone's
                hidden dimension. From config.yaml models.{model}.hidden_dim.
            physics_id: Physics identifier string. Stored as a plain
                attribute for logging/identification. Must follow the
                convention in Shared Knowledge #2 (no dots).

        Raises:
            ValueError: If n_in <= 0 or hidden_dim <= 0.
        """
        super().__init__()

        if n_in <= 0:
            raise ValueError(
                f"n_in must be a positive integer, got {n_in}."
            )
        if hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be a positive integer, got {hidden_dim}."
            )

        self.n_in: int = n_in
        self.hidden_dim: int = hidden_dim
        self.physics_id: str = physics_id

        # ── 2-layer MLP ───────────────────────────────────────────────────
        # Intermediate width = hidden_dim * _LIFTING_HIDDEN_MULTIPLIER (= 2)
        # per config.yaml adapter.lifting_hidden_multiplier: 2.
        intermediate_dim: int = hidden_dim * _LIFTING_HIDDEN_MULTIPLIER

        self._net: nn.Sequential = nn.Sequential(
            nn.Linear(n_in, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, hidden_dim),
        )

        # ── Weight initialization ─────────────────────────────────────────
        # Xavier uniform for stable training across different n_in values.
        for module in self._net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        _logger.debug(
            "LiftingAdapter: physics_id='%s', n_in=%d, hidden_dim=%d, "
            "intermediate_dim=%d. Parameters: %d.",
            physics_id,
            n_in,
            hidden_dim,
            intermediate_dim,
            sum(p.numel() for p in self.parameters()),
        )

    def forward(self, a: Tensor) -> Tensor:
        """Apply the lifting MLP pointwise over spatial locations.

        Reads only the first ``self.n_in`` channels of the input, ignoring
        any zero-padding added by the multi-physics collate_fn. This is the
        key mechanism that allows a single DataLoader to serve adapters with
        different input cardinalities.

        The MLP is applied by temporarily moving the channel dimension to
        the last position (as required by nn.Linear), applying the MLP,
        then restoring the channel-first layout.

        Args:
            a: Input tensor of shape [B, C_in, *spatial] where C_in >= n_in.
                The first n_in channels are the actual input functions;
                channels n_in..C_in are zero-padding (ignored).
                Supports both 1D ([B, C, L]) and 2D ([B, C, H, W]) inputs.

        Returns:
            Lifted feature tensor of shape [B, hidden_dim, *spatial].
            Same spatial dimensions as input, channel count = hidden_dim.

        Raises:
            ValueError: If a has fewer than 2 dimensions.
            ValueError: If a has fewer than n_in channels.
        """
        if a.dim() < 2:
            raise ValueError(
                f"LiftingAdapter expects at least 2D input [B, C, ...], "
                f"got {a.dim()}D tensor with shape {tuple(a.shape)}."
            )

        c_in: int = a.shape[1]
        if c_in < self.n_in:
            raise ValueError(
                f"LiftingAdapter for physics_id='{self.physics_id}' "
                f"expects at least {self.n_in} input channels, "
                f"but got {c_in}. "
                f"Check that the dataset provides the correct number of "
                f"input functions."
            )

        # ── Step 1: Slice to first n_in channels (ignore padding) ─────────
        # a_sliced: [B, n_in, *spatial]
        a_sliced: Tensor = a[:, : self.n_in]

        # ── Step 2: Move channel dim to last position for nn.Linear ───────
        # nn.Linear operates on the last dimension.
        # [B, n_in, *spatial] -> [B, *spatial, n_in]
        #
        # Use permute with dynamic dimension detection to handle both 1D
        # ([B, C, L]) and 2D ([B, C, H, W]) inputs uniformly.
        batch_size: int = a_sliced.shape[0]
        spatial_shape: Tuple[int, ...] = tuple(a_sliced.shape[2:])
        n_spatial_dims: int = len(spatial_shape)

        # Build permutation: [0, 2, 3, ..., n_spatial+1, 1]
        # Moves channel dim (1) to last position.
        perm_fwd: List[int] = (
            [0]
            + list(range(2, 2 + n_spatial_dims))
            + [1]
        )
        # Inverse permutation: [0, n_spatial+1, 1, 2, ..., n_spatial]
        # Moves last dim back to position 1 (channel-first).
        perm_inv: List[int] = [0, n_spatial_dims + 1] + list(range(1, n_spatial_dims + 1))

        # [B, n_in, *spatial] -> [B, *spatial, n_in]
        a_last: Tensor = a_sliced.permute(*perm_fwd).contiguous()

        # ── Step 3: Apply MLP (operates on last dim: n_in -> hidden_dim) ──
        # [B, *spatial, n_in] -> [B, *spatial, hidden_dim]
        v_last: Tensor = self._net(a_last)

        # ── Step 4: Restore channel-first layout ──────────────────────────
        # [B, *spatial, hidden_dim] -> [B, hidden_dim, *spatial]
        v: Tensor = v_last.permute(*perm_inv).contiguous()

        return v


# ---------------------------------------------------------------------------
# ProjectionAdapter
# ---------------------------------------------------------------------------


class ProjectionAdapter(nn.Module):
    """Problem-specific projection adapter (θ_{P_i}).

    Maps the backbone's hidden representation (``hidden_dim`` channels) back
    to problem-specific output channels (``n_out``). This is the ``P``
    component in the paper's pipeline:

        v_out (hidden_dim channels) → ProjectionAdapter → u (n_out channels)

    Architecture (2-layer MLP applied pointwise over spatial locations):
        Linear(hidden_dim, hidden_dim * 2) → GELU → Linear(hidden_dim * 2, n_out)

    The intermediate width ``hidden_dim * 2`` follows the
    ``adapter.projection_hidden_multiplier: 2`` setting in config.yaml.

    Tensor layout:
        Input:  [B, hidden_dim, *spatial]   (channel-first, any spatial dims)
        Output: [B, n_out, *spatial]

    Normalization note:
        The output is in normalized space (since targets were normalized
        during data loading). The ``Evaluator`` is responsible for
        denormalizing predictions before computing NMAE. This adapter
        performs no denormalization.

    Attributes:
        hidden_dim: Input channel count (shared backbone dimension).
        n_out: Number of output channels (physics-specific).
        physics_id: Physics identifier string for logging/identification.
        _net: Sequential MLP: Linear → GELU → Linear.

    Example::

        adapter = ProjectionAdapter(hidden_dim=128, n_out=2,
                                    physics_id='gray_scott_F0p035_k0p065')
        v = torch.randn(8, 128, 64, 64)   # [B, hidden_dim, H, W]
        u = adapter(v)                     # [B, 2, 64, 64]
    """

    def __init__(
        self,
        hidden_dim: int,
        n_out: int,
        physics_id: str,
    ) -> None:
        """Initialise ProjectionAdapter.

        Args:
            hidden_dim: Input channel count. Must match the backbone's
                hidden dimension. From config.yaml models.{model}.hidden_dim.
            n_out: Number of output channels. Physics-specific. Examples:
                1 for Burgers/Advection/heat, 2 for Gray-Scott/RD.
            physics_id: Physics identifier string. Stored as a plain
                attribute for logging/identification. Must follow the
                convention in Shared Knowledge #2 (no dots).

        Raises:
            ValueError: If hidden_dim <= 0 or n_out <= 0.
        """
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be a positive integer, got {hidden_dim}."
            )
        if n_out <= 0:
            raise ValueError(
                f"n_out must be a positive integer, got {n_out}."
            )

        self.hidden_dim: int = hidden_dim
        self.n_out: int = n_out
        self.physics_id: str = physics_id

        # ── 2-layer MLP ───────────────────────────────────────────────────
        # Intermediate width = hidden_dim * _PROJECTION_HIDDEN_MULTIPLIER (= 2)
        # per config.yaml adapter.projection_hidden_multiplier: 2.
        intermediate_dim: int = hidden_dim * _PROJECTION_HIDDEN_MULTIPLIER

        self._net: nn.Sequential = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, n_out),
        )

        # ── Weight initialization ─────────────────────────────────────────
        for module in self._net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        _logger.debug(
            "ProjectionAdapter: physics_id='%s', hidden_dim=%d, n_out=%d, "
            "intermediate_dim=%d. Parameters: %d.",
            physics_id,
            hidden_dim,
            n_out,
            intermediate_dim,
            sum(p.numel() for p in self.parameters()),
        )

    def forward(self, v: Tensor) -> Tensor:
        """Apply the projection MLP pointwise over spatial locations.

        Args:
            v: Hidden representation tensor of shape [B, hidden_dim, *spatial].
                Supports both 1D ([B, C, L]) and 2D ([B, C, H, W]) inputs.

        Returns:
            Output tensor of shape [B, n_out, *spatial].
            Same spatial dimensions as input, channel count = n_out.

        Raises:
            ValueError: If v has fewer than 2 dimensions.
            ValueError: If v's channel dimension does not match hidden_dim.
        """
        if v.dim() < 2:
            raise ValueError(
                f"ProjectionAdapter expects at least 2D input [B, C, ...], "
                f"got {v.dim()}D tensor with shape {tuple(v.shape)}."
            )

        c_in: int = v.shape[1]
        if c_in != self.hidden_dim:
            raise ValueError(
                f"ProjectionAdapter for physics_id='{self.physics_id}' "
                f"expects {self.hidden_dim} input channels (hidden_dim), "
                f"but got {c_in}. "
                f"Ensure the backbone outputs hidden_dim channels."
            )

        # ── Step 1: Move channel dim to last position for nn.Linear ───────
        # [B, hidden_dim, *spatial] -> [B, *spatial, hidden_dim]
        n_spatial_dims: int = v.dim() - 2

        perm_fwd: List[int] = (
            [0]
            + list(range(2, 2 + n_spatial_dims))
            + [1]
        )
        perm_inv: List[int] = [0, n_spatial_dims + 1] + list(range(1, n_spatial_dims + 1))

        # [B, hidden_dim, *spatial] -> [B, *spatial, hidden_dim]
        v_last: Tensor = v.permute(*perm_fwd).contiguous()

        # ── Step 2: Apply MLP (hidden_dim -> n_out) ───────────────────────
        # [B, *spatial, hidden_dim] -> [B, *spatial, n_out]
        u_last: Tensor = self._net(v_last)

        # ── Step 3: Restore channel-first layout ──────────────────────────
        # [B, *spatial, n_out] -> [B, n_out, *spatial]
        u: Tensor = u_last.permute(*perm_inv).contiguous()

        return u


# ---------------------------------------------------------------------------
# AdapterFramework
# ---------------------------------------------------------------------------


class AdapterFramework(nn.Module):
    """Central adapter routing framework for multi-physics neural operator
    pretraining and fine-tuning.

    Wraps any backbone (FNOBackbone, MambaFNO, PerceiverNO, CodaNO) with a
    registry of problem-specific lifting and projection adapters. Implements
    the pretraining/fine-tuning protocol from Section 3 of the paper:

    Pretraining:
        All parameters (backbone + all adapters) are jointly optimized.
        Each mini-batch routes through the adapter pair for its physics_id.

    Fine-tuning:
        Backbone is frozen (freeze_backbone()). Only the new adapter pair
        for the target physics is trained (get_adapter_params(physics_id)).

    The backbone is completely agnostic to which physics problem is being
    solved — it only sees hidden_dim-channel tensors. All physics-specific
    logic is encapsulated in the adapter pairs.

    Attributes:
        hidden_dim: Shared hidden dimension for backbone and all adapters.
        _backbone: Shared backbone module (θ_F). Frozen during fine-tuning.
        _lifting_adapters: nn.ModuleDict mapping physics_id -> LiftingAdapter.
        _projection_adapters: nn.ModuleDict mapping physics_id ->
            ProjectionAdapter.
        _adapter_registry: Plain dict mapping physics_id ->
            {'n_in': int, 'n_out': int}. Saved in checkpoints for
            reconstruction.

    Example::

        from models.fno_backbone import FNOBackbone

        backbone = FNOBackbone(hidden_dim=128, n_modes=16, n_layers=4)
        model = AdapterFramework(backbone=backbone, hidden_dim=128)

        # Register adapters for pretraining physics
        model.register_adapter('burgers_nu0p01', n_in=1, n_out=1)
        model.register_adapter('gray_scott_F0p035_k0p065', n_in=2, n_out=2)

        # Pretrain: all params
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Fine-tune: freeze backbone, train new adapter only
        model.freeze_backbone()
        model.register_adapter('heat_conv_alpha0p01', n_in=3, n_out=1)
        ft_optimizer = torch.optim.Adam(
            model.get_adapter_params('heat_conv_alpha0p01'), lr=1e-4
        )
    """

    def __init__(
        self,
        backbone: nn.Module,
        hidden_dim: int,
    ) -> None:
        """Initialise AdapterFramework.

        Args:
            backbone: Shared backbone module. Any nn.Module that accepts
                [B, hidden_dim, *spatial] tensors and returns tensors of
                the same shape. Typically FNOBackbone, MambaFNO, PerceiverNO,
                or CodaNO. The backbone is stored as self._backbone and
                registered as a submodule for parameter tracking.
            hidden_dim: Shared hidden dimension. Must match the backbone's
                expected input/output channel count. From config.yaml
                models.{model_type}.hidden_dim (e.g., 64 for FNO, 128 for
                Perceiver/CoDA-NO).

        Raises:
            ValueError: If hidden_dim <= 0.
            TypeError: If backbone is not an nn.Module instance.
        """
        super().__init__()

        if not isinstance(backbone, nn.Module):
            raise TypeError(
                f"backbone must be an nn.Module instance, "
                f"got {type(backbone).__name__}."
            )
        if hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be a positive integer, got {hidden_dim}."
            )

        self.hidden_dim: int = hidden_dim

        # ── Backbone (shared θ_F) ─────────────────────────────────────────
        # Registered as a submodule so parameters are tracked by PyTorch.
        # Frozen during fine-tuning via freeze_backbone().
        self._backbone: nn.Module = backbone

        # ── Adapter registries ────────────────────────────────────────────
        # nn.ModuleDict ensures parameters are tracked, state_dict() works,
        # and .to(device) propagates correctly to all adapters.
        self._lifting_adapters: nn.ModuleDict = nn.ModuleDict()
        self._projection_adapters: nn.ModuleDict = nn.ModuleDict()

        # Plain Python dict for metadata (not parameters).
        # Saved in checkpoints for adapter reconstruction on load.
        self._adapter_registry: Dict[str, Dict[str, int]] = {}

        _logger.info(
            "AdapterFramework: backbone_type='%s', hidden_dim=%d. "
            "Backbone parameters: %d.",
            type(backbone).__name__,
            hidden_dim,
            sum(p.numel() for p in backbone.parameters()),
        )

    # -----------------------------------------------------------------------
    # Adapter registration
    # -----------------------------------------------------------------------

    def register_adapter(
        self,
        physics_id: str,
        n_in: int,
        n_out: int,
    ) -> None:
        """Register a lifting/projection adapter pair for a physics problem.

        Creates a LiftingAdapter and ProjectionAdapter for the given
        physics_id and registers them in the internal ModuleDicts. The new
        adapters are automatically moved to the same device as the backbone.

        If an adapter for this physics_id already exists, it is overwritten
        with a warning. This can happen legitimately when loading a checkpoint
        and then re-registering (load_checkpoint calls register_adapter before
        load_state_dict).

        Args:
            physics_id: Physics identifier string. Must follow the convention
                in Shared Knowledge #2: no dots, use 'p' for decimal point.
                Examples: 'burgers_nu0p01', 'gray_scott_F0p035_k0p065'.
                Used as a key in nn.ModuleDict — dots are invalid.
            n_in: Number of input channels for this physics. Examples:
                1 for Burgers/Advection, 2 for Gray-Scott/RD,
                3 for heat+convection.
            n_out: Number of output channels for this physics. Examples:
                1 for Burgers/Advection/heat, 2 for Gray-Scott/RD.

        Raises:
            ValueError: If physics_id contains a dot (invalid for
                nn.ModuleDict keys).
            ValueError: If n_in <= 0 or n_out <= 0.

        Note:
            New adapters are moved to the same device as the backbone
            automatically. If the backbone has no parameters (e.g., an
            identity module), adapters are created on CPU.
        """
        # ── Validate physics_id ───────────────────────────────────────────
        if "." in physics_id:
            raise ValueError(
                f"physics_id='{physics_id}' contains a dot ('.'). "
                f"nn.ModuleDict keys must not contain dots. "
                f"Replace dots with 'p' for decimal values: "
                f"e.g., 'burgers_nu0p01' instead of 'burgers_nu0.01'. "
                f"See Shared Knowledge #2 in the design document."
            )

        if n_in <= 0:
            raise ValueError(
                f"n_in must be a positive integer, got {n_in} "
                f"for physics_id='{physics_id}'."
            )
        if n_out <= 0:
            raise ValueError(
                f"n_out must be a positive integer, got {n_out} "
                f"for physics_id='{physics_id}'."
            )

        # ── Warn on duplicate registration ───────────────────────────────
        if physics_id in self._adapter_registry:
            existing: Dict[str, int] = self._adapter_registry[physics_id]
            _logger.warning(
                "register_adapter: physics_id='%s' is already registered "
                "(n_in=%d, n_out=%d). Overwriting with new adapter "
                "(n_in=%d, n_out=%d).",
                physics_id,
                existing["n_in"],
                existing["n_out"],
                n_in,
                n_out,
            )

        # ── Determine target device ───────────────────────────────────────
        # New adapters should be on the same device as the backbone.
        backbone_params = list(self._backbone.parameters())
        if backbone_params:
            target_device: torch.device = backbone_params[0].device
        else:
            target_device = torch.device("cpu")

        # ── Create adapters ───────────────────────────────────────────────
        lifting_adapter: LiftingAdapter = LiftingAdapter(
            n_in=n_in,
            hidden_dim=self.hidden_dim,
            physics_id=physics_id,
        ).to(target_device)

        projection_adapter: ProjectionAdapter = ProjectionAdapter(
            hidden_dim=self.hidden_dim,
            n_out=n_out,
            physics_id=physics_id,
        ).to(target_device)

        # ── Register in ModuleDicts ───────────────────────────────────────
        self._lifting_adapters[physics_id] = lifting_adapter
        self._projection_adapters[physics_id] = projection_adapter

        # ── Store metadata for checkpoint reconstruction ──────────────────
        self._adapter_registry[physics_id] = {"n_in": n_in, "n_out": n_out}

        _logger.info(
            "register_adapter: physics_id='%s', n_in=%d, n_out=%d, "
            "device='%s'. "
            "Total registered physics: %d.",
            physics_id,
            n_in,
            n_out,
            str(target_device),
            len(self._adapter_registry),
        )

    # -----------------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------------

    def forward(self, a: Tensor, physics_id: str) -> Tensor:
        """Route input through the adapter pair and shared backbone.

        Implements the three-stage pipeline from the paper:
            a → LiftingAdapter[physics_id] → backbone → ProjectionAdapter[physics_id] → u

        The backbone is completely agnostic to the physics problem — it only
        sees hidden_dim-channel tensors. All physics-specific logic is in
        the adapter pair.

        Args:
            a: Input tensor of shape [B, C_in, *spatial] where C_in >= n_in
                for the specified physics_id. Extra channels (zero-padding
                from collate_fn) are silently ignored by LiftingAdapter.
                Supports both 1D ([B, C, L]) and 2D ([B, C, H, W]) inputs.
            physics_id: Physics identifier string. Must be registered via
                register_adapter() before calling forward().

        Returns:
            Prediction tensor of shape [B, n_out, *spatial] where n_out is
            the output channel count registered for this physics_id.
            In normalized space (denormalization is the Evaluator's
            responsibility).

        Raises:
            KeyError: If physics_id is not registered. The error message
                lists all currently registered physics IDs.
        """
        # ── Validate physics_id ───────────────────────────────────────────
        if physics_id not in self._lifting_adapters:
            registered: List[str] = sorted(self._adapter_registry.keys())
            raise KeyError(
                f"Physics ID '{physics_id}' is not
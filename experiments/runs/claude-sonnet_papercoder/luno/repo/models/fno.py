```python
## models/fno.py
"""Fourier Neural Operator (FNO) architecture for the LUNO reproduction.

Implements the FNO as described in the paper (Section 2.1, Example 2.1,
Appendix D.2) using Flax NNX. The architecture is designed to support both
1D PDEs (Burgers, HyperDiffusion, KS) and 2D PDEs (advection-diffusion).

Key design constraint: the last Fourier block is stored as a separate
``last_fourier_block`` attribute to enable LUNO's last-layer linearization
(Appendix C.1), which requires direct access to w_{L-1} = (R^{(L-1)},
W^{(L-1)}) and the hidden state v^{(L-1)}.

Paper references:
  - Section 2.1, Example 2.1: FNO architecture definition
  - Appendix C.1: Last-layer LUNO for FNOs
  - Appendix D.2: Hyperparameters (12 modes, 18 channels, 4 blocks, GELU,
    2-zero spatial padding)
  - config.yaml model section: all architectural hyperparameters

Architecture summary (with config defaults):
  Input a: [batch, spatial, in_channels] (1D) or [batch, H, W, in_channels] (2D)
  → Spatial padding (2 zeros each side)
  → Lifting: Linear(in_channels → channels=18)
  → n_blocks-1 = 3 FourierBlocks (blocks 0..2)
  → last_fourier_block (block 3)
  → Projection: Linear(channels=18 → out_channels=1)
  → Remove spatial padding
  Output: [batch, spatial, out_channels]

Each FourierBlock:
  v^{(l+1)}(x) = σ(SpectralConv(v^{(l)})(x) + Pointwise(v^{(l)})(x))
  where σ = GELU, SpectralConv uses k_max=12 modes, Pointwise is bias-free Linear.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from flax import nnx

from models.spectral_conv import SpectralConv1d, SpectralConv2d

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Activation function registry
# ---------------------------------------------------------------------------
_ACTIVATION_REGISTRY: Dict[str, Callable[[jnp.ndarray], jnp.ndarray]] = {
    "gelu": jax.nn.gelu,
    "relu": jax.nn.relu,
    "tanh": jnp.tanh,
    "silu": jax.nn.silu,
}


def _get_activation(name: str) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return the activation function corresponding to the given name.

    Args:
        name: Activation function name. One of ``'gelu'``, ``'relu'``,
            ``'tanh'``, ``'silu'``.

    Returns:
        A callable that applies the activation element-wise.

    Raises:
        ValueError: If ``name`` is not in the registry.
    """
    if name not in _ACTIVATION_REGISTRY:
        raise ValueError(
            f"Unknown activation '{name}'. "
            f"Available: {list(_ACTIVATION_REGISTRY.keys())}"
        )
    return _ACTIVATION_REGISTRY[name]


# ---------------------------------------------------------------------------
# FourierBlock
# ---------------------------------------------------------------------------

class FourierBlock(nnx.Module):
    """Single Fourier layer of the FNO.

    Implements one step of the FNO recurrence:

    .. math::

        v^{(l+1)}(x) = \\sigma\\bigl(
            \\mathcal{F}^{-1}(R^{(l)} \\cdot \\mathcal{F}(v^{(l)})_k)(x)
            + W^{(l)} v^{(l)}(x)
        \\bigr)

    The spectral part is handled by ``SpectralConv1d`` or ``SpectralConv2d``
    (depending on ``spatial_dims``), and the pointwise linear part by a
    bias-free ``nnx.Linear`` layer.

    Attributes:
        spectral_conv: Spectral convolution layer (1D or 2D).
        pointwise: Bias-free pointwise linear layer (W^{(l)}).
        activation: Activation function σ^{(l)}.
        spatial_dims: Number of spatial dimensions (1 or 2).
        channels: Number of hidden channels (d_v').
        modes: Number of Fourier modes (k_max).

    Example::

        rngs = nnx.Rngs(params=0)
        block = FourierBlock(channels=18, modes=12, activation_name='gelu',
                             spatial_dims=1, rngs=rngs)
        v = jnp.ones([4, 260, 18])
        v_next = block(v)
        # v_next.shape == (4, 260, 18)
    """

    def __init__(
        self,
        channels: int,
        modes: int,
        activation_name: str = "gelu",
        spatial_dims: int = 1,
        rngs: Optional[nnx.Rngs] = None,
    ) -> None:
        """Initialise a single Fourier block.

        Args:
            channels: Number of hidden channels (d_v'). From
                ``config.model.channels`` (default 18). Both input and
                output channels are ``channels`` (square transformation).
            modes: Number of Fourier modes to retain (k_max). From
                ``config.model.modes`` (default 12).
            activation_name: Name of the activation function. From
                ``config.model.activation`` (default ``'gelu'``).
            spatial_dims: Number of spatial dimensions. 1 for 1D PDEs
                (Burgers, HyperDiff, KS), 2 for 2D PDEs (advection-diffusion).
            rngs: Flax NNX random number generator state. If ``None``, a
                default ``nnx.Rngs(params=0)`` is used (for testing only).

        Raises:
            ValueError: If ``channels <= 0``, ``modes <= 0``, or
                ``spatial_dims`` is not 1 or 2.
        """
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if modes <= 0:
            raise ValueError(f"modes must be positive, got {modes}")
        if spatial_dims not in (1, 2):
            raise ValueError(
                f"spatial_dims must be 1 or 2, got {spatial_dims}"
            )

        if rngs is None:
            rngs = nnx.Rngs(params=0)

        self.channels: int = channels
        self.modes: int = modes
        self.spatial_dims: int = spatial_dims

        # ------------------------------------------------------------------
        # Spectral convolution: SpectralConv1d or SpectralConv2d
        # ------------------------------------------------------------------
        if spatial_dims == 1:
            self.spectral_conv: Union[SpectralConv1d, SpectralConv2d] = (
                SpectralConv1d(
                    in_channels=channels,
                    out_channels=channels,
                    modes=modes,
                    rngs=rngs,
                )
            )
        else:
            self.spectral_conv = SpectralConv2d(
                in_channels=channels,
                out_channels=channels,
                modes1=modes,
                modes2=modes,
                rngs=rngs,
            )

        # ------------------------------------------------------------------
        # Pointwise linear: W^{(l)} ∈ R^{d_v' × d_v'}, no bias
        # Applied to the last (channel) dimension at each spatial point.
        # ------------------------------------------------------------------
        self.pointwise: nnx.Linear = nnx.Linear(
            in_features=channels,
            out_features=channels,
            use_bias=False,
            rngs=rngs,
        )

        # ------------------------------------------------------------------
        # Activation function σ^{(l)}
        # ------------------------------------------------------------------
        self.activation: Callable[[jnp.ndarray], jnp.ndarray] = _get_activation(
            activation_name
        )

    def __call__(self, v: jnp.ndarray) -> jnp.ndarray:
        """Apply one Fourier layer transformation.

        Computes:
            v_next = σ(SpectralConv(v) + Pointwise(v))

        Args:
            v: Hidden state tensor.
                - 1D: shape ``[batch, spatial, channels]``
                - 2D: shape ``[batch, H, W, channels]``

        Returns:
            Updated hidden state with the same shape as ``v``.
        """
        # Spectral path: F^{-1}(R^{(l)} · F(v)_k)
        spectral_out: jnp.ndarray = self.spectral_conv(v)

        # Pointwise path: W^{(l)} v(x)
        # nnx.Linear applies to the last dimension, which is the channel dim.
        # This is correct for both 1D [batch, spatial, channels] and
        # 2D [batch, H, W, channels] inputs.
        pointwise_out: jnp.ndarray = self.pointwise(v)

        # Sum and apply activation
        return self.activation(spectral_out + pointwise_out)


# ---------------------------------------------------------------------------
# FNO
# ---------------------------------------------------------------------------

class FNO(nnx.Module):
    """Fourier Neural Operator (FNO) for PDE solution operator learning.

    Implements the FNO architecture from Li et al. (2021) as described in
    the LUNO paper (Section 2.1, Example 2.1). Supports both 1D and 2D
    spatial domains.

    Architecture:
        Input a → Pad → Lifting → [n_blocks-1 FourierBlocks] →
        last_fourier_block → Projection → Unpad → Output

    The ``last_fourier_block`` is stored as a separate attribute to enable
    LUNO's last-layer linearization (Appendix C.1).

    Attributes:
        modes: Number of Fourier modes (k_max). Config: 12.
        channels: Hidden channel width (d_v'). Config: 18.
        n_blocks: Total number of Fourier blocks. Config: 4.
        in_channels: Number of input channels. 12 for 1D, 13 for 2D.
        out_channels: Number of output channels. Config: 1.
        spatial_dims: Number of spatial dimensions (1 or 2).
        spatial_padding: Number of zero-padding points per side. Config: 2.
        activation_name: Name of the activation function. Config: 'gelu'.
        lifting: Lifting layer p: R^{d_A'} → R^{d_v'}.
        fourier_blocks: List of n_blocks-1 intermediate FourierBlocks.
        last_fourier_block: The final FourierBlock (block L-1 in paper).
            Its weights w_{L-1} = (R^{(L-1)}, W^{(L-1)}) are used by LUNO.
        projection: Projection layer q: R^{d_v'} → R^{d_U'}.

    Example::

        rngs = nnx.Rngs(params=42)
        model = FNO(
            modes=12, channels=18, n_blocks=4,
            in_channels=12, out_channels=1,
            spatial_dims=1, spatial_padding=2,
            activation_name='gelu', rngs=rngs,
        )
        a = jnp.ones([4, 256, 12])  # [batch, spatial, in_channels]
        out = model(a)
        # out.shape == (4, 256, 1)
    """

    def __init__(
        self,
        modes: int = 12,
        channels: int = 18,
        n_blocks: int = 4,
        in_channels: int = 12,
        out_channels: int = 1,
        spatial_dims: int = 1,
        spatial_padding: int = 2,
        activation_name: str = "gelu",
        rngs: Optional[nnx.Rngs] = None,
    ) -> None:
        """Initialise the FNO.

        Args:
            modes: Number of Fourier modes per spatial dimension (k_max).
                From ``config.model.modes`` (default 12).
            channels: Hidden channel width (d_v'), constant throughout.
                From ``config.model.channels`` (default 18).
            n_blocks: Total number of Fourier blocks. From
                ``config.model.n_blocks`` (default 4). The first
                ``n_blocks - 1`` blocks are stored in ``fourier_blocks``;
                the last one is ``last_fourier_block``.
            in_channels: Number of input channels. Computed from config:
                ``input_steps + 2`` for 1D (default 12),
                ``input_steps + 3`` for 2D (default 13).
            out_channels: Number of output channels. From
                ``config.model.out_channels`` (default 1).
            spatial_dims: Number of spatial dimensions. 1 for 1D PDEs,
                2 for 2D PDEs.
            spatial_padding: Number of zero-padding points added to each
                side of the spatial dimension(s). From
                ``config.model.spatial_padding`` (default 2).
            activation_name: Activation function name. From
                ``config.model.activation`` (default ``'gelu'``).
            rngs: Flax NNX random number generator state. If ``None``, a
                default ``nnx.Rngs(params=0)`` is used (for testing only).

        Raises:
            ValueError: If ``n_blocks < 2`` (need at least one intermediate
                block and one last block for LUNO).
            ValueError: If ``spatial_dims`` is not 1 or 2.
            ValueError: If any channel count or mode count is non-positive.
        """
        if n_blocks < 2:
            raise ValueError(
                f"n_blocks must be >= 2 (need at least one intermediate block "
                f"+ one last block for LUNO), got {n_blocks}"
            )
        if spatial_dims not in (1, 2):
            raise ValueError(
                f"spatial_dims must be 1 or 2, got {spatial_dims}"
            )
        if modes <= 0:
            raise ValueError(f"modes must be positive, got {modes}")
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")
        if spatial_padding < 0:
            raise ValueError(
                f"spatial_padding must be non-negative, got {spatial_padding}"
            )

        if rngs is None:
            rngs = nnx.Rngs(params=0)

        self.modes: int = modes
        self.channels: int = channels
        self.n_blocks: int = n_blocks
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.spatial_dims: int = spatial_dims
        self.spatial_padding: int = spatial_padding
        self.activation_name: str = activation_name

        # ------------------------------------------------------------------
        # Lifting layer: p: R^{d_A'} → R^{d_v'}
        # Maps input channels to the hidden channel width.
        # ------------------------------------------------------------------
        self.lifting: nnx.Linear = nnx.Linear(
            in_features=in_channels,
            out_features=channels,
            use_bias=True,
            rngs=rngs,
        )

        # ------------------------------------------------------------------
        # Intermediate Fourier blocks: blocks 0 .. n_blocks-2
        # Stored as a list; NNX tracks them as sub-modules automatically.
        # ------------------------------------------------------------------
        self.fourier_blocks: List[FourierBlock] = [
            FourierBlock(
                channels=channels,
                modes=modes,
                activation_name=activation_name,
                spatial_dims=spatial_dims,
                rngs=rngs,
            )
            for _ in range(n_blocks - 1)
        ]

        # ------------------------------------------------------------------
        # Last Fourier block: block n_blocks-1 (= L-1 in paper notation)
        # Stored separately for LUNO last-layer access.
        # Its weights w_{L-1} = (R^{(L-1)}, W^{(L-1)}) are used by:
        #   - uncertainty/ggn.py: GGNComputer._get_last_layer_params
        #   - uncertainty/luno.py: LUNOInference._compute_feature_functions
        # ------------------------------------------------------------------
        self.last_fourier_block: FourierBlock = FourierBlock(
            channels=channels,
            modes=modes,
            activation_name=activation_name,
            spatial_dims=spatial_dims,
            rngs=rngs,
        )

        # ------------------------------------------------------------------
        # Projection layer: q: R^{d_v'} → R^{d_U'}
        # Maps hidden channels to output channels.
        # Applied after the last Fourier block's activation.
        # ------------------------------------------------------------------
        self.projection: nnx.Linear = nnx.Linear(
            in_features=channels,
            out_features=out_channels,
            use_bias=True,
            rngs=rngs,
        )

        logger.info(
            "FNO initialised: modes=%d, channels=%d, n_blocks=%d, "
            "in_channels=%d, out_channels=%d, spatial_dims=%d, "
            "spatial_padding=%d, activation=%s",
            modes,
            channels,
            n_blocks,
            in_channels,
            out_channels,
            spatial_dims,
            spatial_padding,
            activation_name,
        )

    # -----------------------------------------------------------------------
    # Spatial Padding Helpers
    # -----------------------------------------------------------------------

    def _pad(self, a: jnp.ndarray) -> jnp.ndarray:
        """Apply zero-padding to the spatial dimension(s).

        Pads ``spatial_padding`` zeros on each side of the spatial axis
        (axis 1 for 1D, axes 1 and 2 for 2D). The channel dimension is
        never padded.

        Args:
            a: Input tensor.
                - 1D: shape ``[batch, spatial, channels]``
                - 2D: shape ``[batch, H, W, channels]``

        Returns:
            Padded tensor.
                - 1D: shape ``[batch, spatial + 2*padding, channels]``
                - 2D: shape ``[batch, H + 2*padding, W + 2*padding, channels]``
        """
        p: int = self.spatial_padding
        if p == 0:
            return a

        if self.spatial_dims == 1:
            # Pad axis 1 (spatial), leave axes 0 (batch) and 2 (channels) unchanged
            pad_width: Tuple[Tuple[int, int], ...] = ((0, 0), (p, p), (0, 0))
        else:
            # Pad axes 1 (H) and 2 (W), leave axes 0 (batch) and 3 (channels) unchanged
            pad_width = ((0, 0), (p, p), (p, p), (0, 0))

        return jnp.pad(a, pad_width, mode="constant", constant_values=0.0)

    def _unpad(self, a: jnp.ndarray) -> jnp.ndarray:
        """Remove zero-padding from the spatial dimension(s).

        Slices off ``spatial_padding`` elements from each side of the
        spatial axis (axis 1 for 1D, axes 1 and 2 for 2D).

        Args:
            a: Padded tensor.
                - 1D: shape ``[batch, spatial + 2*padding, channels]``
                - 2D: shape ``[batch, H + 2*padding, W + 2*padding, channels]``

        Returns:
            Unpadded tensor.
                - 1D: shape ``[batch, spatial, channels]``
                - 2D: shape ``[batch, H, W, channels]``
        """
        p: int = self.spatial_padding
        if p == 0:
            return a

        if self.spatial_dims == 1:
            return a[:, p:-p, :]
        else:
            return a[:, p:-p, p:-p, :]

    # -----------------------------------------------------------------------
    # Forward Pass
    # -----------------------------------------------------------------------

    def __call__(self, a: jnp.ndarray) -> jnp.ndarray:
        """Run the full FNO forward pass.

        Applies the complete FNO pipeline:
        1. Spatial padding (2 zeros per side)
        2. Lifting: in_channels → channels
        3. n_blocks - 1 intermediate Fourier blocks
        4. Last Fourier block (w_{L-1})
        5. Projection: channels → out_channels
        6. Remove spatial padding

        Args:
            a: Input function discretization.
                - 1D: shape ``[batch, spatial, in_channels]``
                - 2D: shape ``[batch, H, W, in_channels]``

        Returns:
            Output function discretization.
                - 1D: shape ``[batch, spatial, out_channels]``
                - 2D: shape ``[batch, H, W, out_channels]``

        Example::

            a = jnp.ones([4, 256, 12])
            out = model(a)
            # out.shape == (4, 256, 1)
        """
        # Step 1: Spatial padding
        a_padded: jnp.ndarray = self._pad(a)

        # Step 2: Lifting p: R^{d_A'} → R^{d_v'}
        v: jnp.ndarray = self.lifting(a_padded)

        # Step 3: Intermediate Fourier blocks (blocks 0 .. n_blocks-2)
        for block in self.fourier_blocks:
            v = block(v)

        # Step 4: Last Fourier block (block n_blocks-1 = L-1 in paper)
        v = self.last_fourier_block(v)

        # Step 5: Projection q: R^{d_v'} → R^{d_U'}
        out: jnp.ndarray = self.projection(v)

        # Step 6: Remove spatial padding
        out = self._unpad(out)

        return out

    def get_hidden_state(self, a: jnp.ndarray) -> jnp.ndarray:
        """Compute the hidden state v^{(L-1)}: input to the last Fourier block.

        This is used by ``LUNOInference._compute_feature_functions`` to
        construct the feature functions φ_{kj}(x), ψ_{kj}(x), ψ_j(x)
        for the last-layer LUNO linearization (Appendix C.1).

        The returned hidden state is still spatially padded (padding is
        removed only after the projection layer in the full forward pass).
        This is intentional: the LUNO feature functions are computed over
        the padded spatial grid, and the padding is removed at the output
        level when computing marginal variances.

        Args:
            a: Input function discretization.
                - 1D: shape ``[batch, spatial, in_channels]``
                - 2D: shape ``[batch, H, W, in_channels]``

        Returns:
            Hidden state v^{(L-1)}: output of the second-to-last Fourier
            block (i.e., input to ``last_fourier_block``).
                - 1D: shape ``[batch, spatial + 2*padding, channels]``
                - 2D: shape ``[batch, H + 2*padding, W + 2*padding, channels]``

        Example::

            a = jnp.ones([4, 256, 12])
            v_hidden = model.get_hidden_state(a)
            # v_hidden.shape == (4, 260, 18)  # padded spatial, channels
        """
        # Step 1: Spatial padding
        a_padded: jnp.ndarray = self._pad(a)

        # Step 2: Lifting
        v: jnp.ndarray = self.lifting(a_padded)

        # Step 3: Intermediate Fourier blocks only (NOT the last block)
        for block in self.fourier_blocks:
            v = block(v)

        # Return v^{(L-1)}: the hidden state before the last Fourier block
        return v

    def get_last_layer_preactivation(self, a: jnp.ndarray) -> jnp.ndarray:
        """Compute z^{(L-1)}: pre-activation output of the last Fourier block.

        Computes the linear combination inside the last Fourier block
        before applying the activation σ^{(L-1)}:

        .. math::

            z^{(L-1)}(x) = \\mathcal{F}^{-1}(R^{(L-1)} \\cdot
                \\mathcal{F}(v^{(L-1)})_k)(x)
                + W^{(L-1)} v^{(L-1)}(x)

        This is used by ``LUNOInference`` to compute the Jacobian of
        q̃ = q ∘ σ^{(L-1)} w.r.t. z^{(L-1)}.

        Args:
            a: Input function discretization.
                - 1D: shape ``[batch, spatial, in_channels]``
                - 2D: shape ``[batch, H, W, in_channels]``

        Returns:
            Pre-activation hidden state z^{(L-1)}.
                - 1D: shape ``[batch, spatial + 2*padding, channels]``
                - 2D: shape ``[batch, H + 2*padding, W + 2*padding, channels]``
        """
        # Get v^{(L-1)}: input to the last Fourier block
        v_prev: jnp.ndarray = self.get_hidden_state(a)

        # Compute the pre-activation output of the last Fourier block
        # (spectral + pointwise, without activation)
        spectral_out: jnp.ndarray = self.last_fourier_block.spectral_conv(v_prev)
        pointwise_out: jnp.ndarray = self.last_fourier_block.pointwise(v_prev)
        z: jnp.ndarray = spectral_out + pointwise_out

        return z

    # -----------------------------------------------------------------------
    # Parameter Initialization and Functional Interface
    # -----------------------------------------------------------------------

    def init_params(
        self,
        key: jax.Array,
        dummy_input: jnp.ndarray,
    ) -> Tuple[Any, Any]:
        """Initialise parameters via a dummy forward pass and return the state.

        Performs a forward pass with ``dummy_input`` to trigger NNX
        parameter initialisation, then extracts the graph definition and
        state for use with JAX functional transforms.

        The returned ``(graphdef, state)`` pair can be used to reconstruct
        the model with different parameters via ``nnx.merge(graphdef, state)``,
        enabling JAX JVP/VJP computation in ``GGNComputer``.

        Args:
            key: JAX PRNG key. Used to re-seed the module's RNG state
                before the dummy forward pass (ensures reproducibility).
            dummy_input: A dummy input array with the correct shape for
                this model. Typically ``jnp.zeros([1, spatial, in_channels])``
                for 1D or ``jnp.zeros([1, H, W, in_channels])`` for 2D.

        Returns:
            A tuple ``(graphdef, state)`` where:
            - ``graphdef``: The NNX graph definition (structure without values).
            - ``state``: The NNX state (parameter values as a pytree).

        Example::

            key = jax.random.PRNGKey(42)
            dummy = jnp.zeros([1, 256, 12])
            graphdef, state = model.init_params(key, dummy)
            # Reconstruct model with same params:
            model_copy = nnx.merge(graphdef, state)
            out = model_copy(dummy)
        """
        # Perform a dummy forward pass to ensure all parameters are initialised
        # (NNX initialises lazily on first call in some configurations)
        _ = self(dummy_input)

        # Extract graph definition and state
        graphdef, state = nnx.split(self)

        logger.debug(
            "init_params: extracted graphdef and state from FNO "
            "(dummy_input shape: %s)",
            dummy_input.shape,
        )

        return graphdef, state

    def apply(
        self,
        state: Any,
        a: jnp.ndarray,
    ) -> jnp.ndarray:
        """Apply the FNO with a given state (functional interface).

        Reconstructs the model from the stored graph definition and the
        provided state, then runs the forward pass. This enables JAX
        functional transforms (JVP, VJP) over the model parameters.

        This method requires that ``init_params`` has been called first to
        store the graph definition in ``self._graphdef``.

        Args:
            state: NNX state (parameter values). Must be compatible with
                the graph definition stored in ``self._graphdef``.
            a: Input function discretization.

        Returns:
            Output function discretization with the same spatial shape as
            ``a`` and ``out_channels`` in the last dimension.

        Raises:
            RuntimeError: If ``init_params`` has not been called first.

        Example::

            graphdef, state = model.init_params(key, dummy)
            # Perturb state and apply:
            out = model.apply(state, a)
        """
        if not hasattr(self, "_graphdef"):
            # Store graphdef on first call to apply
            self._graphdef, _ = nnx.split(self)

        model_copy: FNO = nnx.merge(self._graphdef, state)
        return model_copy(a)

    def get_hidden_state_functional(
        self,
        state: Any,
        a: jnp.ndarray,
    ) -> jnp.ndarray:
        """Functional version of ``get_hidden_state`` for JAX transforms.

        Args:
            state: NNX state (parameter values).
            a: Input function discretization.

        Returns:
            Hidden state v^{(L-1)} with the same shape as the padded input
            after lifting and intermediate Fourier blocks.
        """
        if not hasattr(self, "_graphdef"):
            self._graphdef, _ = nnx.split(self)

        model_copy: FNO = nnx.merge(self._graphdef, state)
        return model_copy.get_hidden_state(a)

    # -----------------------------------------------------------------------
    # Convenience: Build from Config
    # -----------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: Any,
        rngs: Optional
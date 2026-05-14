## ngpt_components.py
"""Core nGPT-specific primitives for the Normalized Transformer.

This module implements the four foundational components that distinguish nGPT
from a standard GPT:

1. ScaledParameter  — Section 2.5: effective learning rate control via the
                      s_init/s_scale two-scalar trick.
2. NormLinear       — Sections 2.3.2, 2.4.2: normalized weight matrices where
                      each embedding-dimension vector has unit L2 norm.
3. NormEmbedding    — Section 2.1: normalized token embedding tables where
                      each token's embedding vector has unit L2 norm.
4. HypersphereUpdate — Section 2.2: LERP and SLERP update equations for
                       moving points on the unit hypersphere.

This file has NO dependencies on other project files. It imports only from
PyTorch and the Python standard library, making it independently testable.

All four classes are used by model.py. The post-step normalization flow is:
    trainer.py → model.normalize_all_weights()
               → NormLinear.normalize_weights() / NormEmbedding.normalize_weights()
               → modifies .weight.data in-place via .copy_()

Typical usage (from model.py):
    from ngpt_components import (
        ScaledParameter,
        NormLinear,
        NormEmbedding,
        HypersphereUpdate,
    )

    # Eigen learning rate for attention block
    alpha_a = ScaledParameter(shape=(d_model,), s_init=0.05,
                               s_scale=1.0/math.sqrt(d_model))

    # Normalized query projection
    Wq = NormLinear(in_features=d_model, out_features=d_k, norm_dim=1)

    # Normalized input embedding table
    E_input = NormEmbedding(num_embeddings=vocab_size, embedding_dim=d_model)

    # LERP update on the hypersphere
    h = HypersphereUpdate.lerp_update(h, hA, alpha_a.forward())
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Small epsilon for numerical stability in SLERP (arccos, division by sin)
_SLERP_EPS: float = 1e-8


# ---------------------------------------------------------------------------
# ScaledParameter
# ---------------------------------------------------------------------------

class ScaledParameter(nn.Module):
    """Learnable scaling parameter with controlled effective Adam learning rate.

    Implements the two-scalar trick from Section 2.5 of the nGPT paper. The
    core insight is that Adam's effective step size for a parameter is
    approximately proportional to the parameter's magnitude (since Adam
    normalizes gradients by their running variance estimate). By storing
    ``s_scale`` as the actual parameter value and recovering ``s_init`` in the
    forward pass via multiplication by ``s_init / s_scale``, we decouple:

        - The value the network sees (starts at ``s_init``)
        - The effective Adam learning rate (proportional to ``s_scale``)

    Setting ``s_scale = 1/sqrt(d_model)`` makes the effective learning rate
    for this parameter comparable to other normalized parameters in the network
    (whose magnitudes are also ~1/sqrt(d_model)).

    Example (from Section 2.6):
        - alpha_A: s_init=0.05, s_scale=1/sqrt(d_model)
        - sqk:     s_init=1.0,  s_scale=1/sqrt(d_model)
        - su, sv:  s_init=1.0,  s_scale=1.0
        - sz:      s_init=1.0,  s_scale=1/sqrt(d_model)

    Attributes:
        param: The stored nn.Parameter, initialized to ``s_scale``. Adam
            updates this value. Shape matches the ``shape`` argument.
        multiplier: Fixed float constant ``s_init / s_scale``. Applied in
            the forward pass to recover the actual value. NOT a parameter.
        shape: The shape of the parameter tensor.

    Note:
        The caller is responsible for applying ``torch.abs()`` to the output
        of ``forward()`` when using this as an eigen learning rate (alpha_A,
        alpha_M), as the paper uses ``|alpha|`` during the forward pass
        (Appendix A.2). This class does not enforce positivity internally to
        remain general-purpose.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        s_init: float = 1.0,
        s_scale: float = 1.0,
    ) -> None:
        """Initialize the ScaledParameter.

        Args:
            shape: Shape of the parameter tensor. Examples:
                - (d_model,) for alpha_a, alpha_m
                - (n_heads, d_k) for sqk (one vector per head)
                - (d_mlp,) for su, sv
                - (vocab_size,) for sz
            s_init: The initial actual value seen by the network during the
                forward pass. The parameter is initialized so that
                ``forward()`` returns a tensor filled with ``s_init``.
            s_scale: Controls the effective Adam learning rate. The stored
                parameter is initialized to this value. Setting
                ``s_scale = 1/sqrt(d_model)`` gives the same effective LR
                as other normalized parameters.

        Raises:
            ValueError: If ``s_scale`` is zero (would cause division by zero
                when computing the multiplier).
        """
        super().__init__()

        if s_scale == 0.0:
            raise ValueError(
                f"s_scale must be non-zero, got s_scale={s_scale}. "
                "A zero s_scale would cause division by zero when computing "
                "the multiplier s_init / s_scale."
            )

        self.shape: Tuple[int, ...] = shape

        # Store the parameter initialized to s_scale.
        # Adam's variance estimate will be initialized around s_scale^2,
        # giving effective step size proportional to s_scale.
        self.param: nn.Parameter = nn.Parameter(
            torch.full(shape, fill_value=float(s_scale))
        )

        # Fixed multiplier: s_init / s_scale.
        # At initialization: param * multiplier = s_scale * (s_init/s_scale) = s_init.
        # Stored as a plain Python float (not a tensor) to avoid accidental
        # gradient flow and unnecessary tensor allocation.
        self.multiplier: float = float(s_init) / float(s_scale)

    def forward(self) -> Tensor:
        """Compute the actual parameter value seen by the network.

        Returns the stored parameter multiplied by the fixed multiplier,
        recovering the intended actual value. At initialization, this returns
        a tensor filled with ``s_init``. After training, values will have
        drifted from ``s_init`` based on gradient updates.

        Returns:
            Tensor of shape ``self.shape`` containing the actual parameter
            values. At initialization, all elements equal ``s_init``.
        """
        return self.param * self.multiplier

    def get_actual_value(self) -> Tensor:
        """Return the actual parameter value (alias for forward()).

        Provided for semantic clarity when called outside of a model forward
        pass context, e.g., in analysis code that inspects learned parameter
        distributions (evaluation.py: analyze_learned_parameters).

        Returns:
            Tensor of shape ``self.shape`` containing the actual parameter
            values. Equivalent to calling ``forward()``.
        """
        return self.forward()

    def extra_repr(self) -> str:
        """Return a string with extra information for repr."""
        return (
            f"shape={self.shape}, "
            f"multiplier={self.multiplier:.6f}"
        )


# ---------------------------------------------------------------------------
# NormLinear
# ---------------------------------------------------------------------------

class NormLinear(nn.Module):
    """Linear layer with weight vectors normalized to unit L2 norm.

    Implements the normalized weight matrices from Sections 2.3.2 and 2.4.2.
    Each weight vector along the embedding dimension (``d_model``) is
    constrained to unit norm, so that the dot product with the hidden state
    ``h`` (also unit norm) represents a cosine similarity bounded in [-1, 1].

    The normalization is applied in two places:
    1. During the forward pass (via ``_get_normalized_weight()``): ensures
       gradients are computed with respect to normalized weights.
    2. After each optimizer step (via ``normalize_weights()``): snaps the
       stored weights back to the hypersphere, preventing drift.

    Normalization convention (``norm_dim`` parameter):
        The ``norm_dim`` argument is passed directly to ``F.normalize(weight,
        dim=norm_dim)``. Given PyTorch's weight shape convention
        ``(out_features, in_features)``:

        - ``norm_dim=1``: normalizes each row (each ``in_features``-dim vector).
          Use when ``in_features = d_model``, i.e., the weight left-multiplies
          the hidden state h. Applies to: Wq, Wk, Wv, Wu, Wv_mlp.

        - ``norm_dim=0``: normalizes each column (each ``out_features``-dim
          vector). Use when ``out_features = d_model``, i.e., the weight
          produces d_model-dim output vectors. Applies to: Wo, WoMLP.

    Note:
        Bias terms are not supported (``bias=False`` is the paper default,
        Section 2.4.1: "we omit bias terms"). The ``bias`` parameter is
        accepted for API compatibility but raises an error if set to True.

    Attributes:
        in_features: Input feature dimension.
        out_features: Output feature dimension.
        norm_dim: Dimension along which to normalize (0 or 1).
        weight: The learnable weight parameter of shape
            ``(out_features, in_features)``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        norm_dim: int = 0,
        bias: bool = False,
    ) -> None:
        """Initialize NormLinear with normalized weights.

        Args:
            in_features: Size of each input sample (typically ``d_model`` or
                ``d_k`` or ``d_mlp``).
            out_features: Size of each output sample.
            norm_dim: Dimension along which to normalize the weight matrix.
                Pass ``norm_dim=1`` when ``in_features = d_model`` (normalize
                rows). Pass ``norm_dim=0`` when ``out_features = d_model``
                (normalize columns). Defaults to 0.
            bias: Must be False. Bias terms are not used in nGPT (paper
                Section 2.4.1). Raises ValueError if True.

        Raises:
            ValueError: If ``bias=True`` (not supported in nGPT).
            ValueError: If ``norm_dim`` is not 0 or 1.
        """
        super().__init__()

        if bias:
            raise ValueError(
                "NormLinear does not support bias terms. "
                "The nGPT paper omits bias terms in all linear layers "
                "(Section 2.4.1: 'we omit bias terms'). Set bias=False."
            )

        if norm_dim not in (0, 1):
            raise ValueError(
                f"norm_dim must be 0 or 1, got norm_dim={norm_dim}. "
                "norm_dim=1 normalizes rows (use when in_features=d_model). "
                "norm_dim=0 normalizes columns (use when out_features=d_model)."
            )

        self.in_features: int = in_features
        self.out_features: int = out_features
        self.norm_dim: int = norm_dim

        # Initialize weight with normal distribution.
        # The exact initialization values do not matter for nGPT because
        # normalize_weights() is called immediately after model construction.
        # We use std=1/sqrt(in_features) as a reasonable starting point.
        self.weight: nn.Parameter = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        nn.init.normal_(self.weight, mean=0.0, std=1.0 / math.sqrt(in_features))

    def _get_normalized_weight(self) -> Tensor:
        """Return the weight matrix normalized along ``norm_dim``.

        This is called during the forward pass to ensure gradients flow
        through the normalized weights. Does NOT modify ``self.weight.data``
        in-place — returns a new tensor.

        Returns:
            Normalized weight tensor of shape ``(out_features, in_features)``.
            Each slice along ``norm_dim`` has unit L2 norm.
        """
        return F.normalize(self.weight, p=2, dim=self.norm_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Compute the linear transformation with normalized weights.

        Applies ``F.linear(x, normalized_weight)``, which computes
        ``x @ normalized_weight.T``.

        Args:
            x: Input tensor of shape ``(..., in_features)``.

        Returns:
            Output tensor of shape ``(..., out_features)``.
        """
        return F.linear(x, self._get_normalized_weight())

    def normalize_weights(self) -> None:
        """Normalize weight vectors in-place after an optimizer step.

        Called by ``nGPTModel.normalize_all_weights()`` after every call to
        ``optimizer.step()``. Snaps the stored weight back to the hypersphere,
        preventing accumulated floating-point drift.

        Uses ``weight.data.copy_()`` rather than direct assignment to preserve
        the tensor's storage identity. This is critical: the optimizer's
        momentum buffers (m, v in Adam) hold references to the same underlying
        storage as ``weight``. Using ``.copy_()`` updates the values in-place
        while keeping those references valid.

        The ``torch.no_grad()`` context prevents this operation from being
        tracked by autograd, which would create spurious gradient nodes.
        """
        with torch.no_grad():
            self.weight.data.copy_(
                F.normalize(self.weight.data, p=2, dim=self.norm_dim)
            )

    def extra_repr(self) -> str:
        """Return a string with extra information for repr."""
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"norm_dim={self.norm_dim}, "
            f"bias=False"
        )


# ---------------------------------------------------------------------------
# NormEmbedding
# ---------------------------------------------------------------------------

class NormEmbedding(nn.Module):
    """Embedding table with each token vector normalized to unit L2 norm.

    Implements Section 2.1's requirement that both ``E_input`` and ``E_output``
    have normalized rows. After normalization, each token embedding
    ``E[i] ∈ R^d_model`` satisfies ``||E[i]||_2 = 1``, so the dot product
    with the unit-norm hidden state ``h`` represents a cosine similarity
    bounded in [-1, 1].

    Like NormLinear, normalization is applied both during the forward pass
    (for correct gradient flow) and after each optimizer step (to prevent
    drift).

    The logit computation in nGPTModel uses ``E_output.weight`` directly
    (not through ``forward()``) for the full matrix multiplication:
        ``logits = (h @ E_output.weight.T) * sz``
    The weight is kept normalized via ``normalize_weights()``, so this is
    equivalent to using normalized embeddings.

    Attributes:
        num_embeddings: Vocabulary size (32000 for LLaMA-2, 50257 for GPT-2).
        embedding_dim: Embedding dimension (``d_model``).
        weight: The learnable embedding parameter of shape
            ``(num_embeddings, embedding_dim)``.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ) -> None:
        """Initialize NormEmbedding with normalized token vectors.

        Args:
            num_embeddings: Vocabulary size. Typically 32000 (LLaMA-2
                tokenizer, config.yaml data.vocab_size) or 50257 (GPT-2
                fallback tokenizer).
            embedding_dim: Embedding dimension, equal to ``d_model``.
        """
        super().__init__()

        self.num_embeddings: int = num_embeddings
        self.embedding_dim: int = embedding_dim

        # Initialize with normal distribution.
        # Exact values don't matter — normalize_weights() is called after init.
        self.weight: nn.Parameter = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim)
        )
        nn.init.normal_(
            self.weight,
            mean=0.0,
            std=1.0 / math.sqrt(embedding_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Look up normalized embeddings for a batch of token indices.

        Normalizes the full weight matrix before indexing. This ensures that
        during the backward pass, gradients flow through the normalized
        embeddings (correct behavior for hypersphere optimization).

        Args:
            x: Integer tensor of token indices, shape ``(batch_size,
                seq_len)`` or any shape. Values must be in
                ``[0, num_embeddings)``.

        Returns:
            Embedding tensor of shape ``(*x.shape, embedding_dim)``.
            Each embedding vector has unit L2 norm.
        """
        # Normalize the full weight matrix, then index into it.
        # F.normalize(weight, dim=1) normalizes each row (each token embedding).
        normalized_weight = F.normalize(self.weight, p=2, dim=1)
        return F.embedding(x, normalized_weight)

    def normalize_weights(self) -> None:
        """Normalize all token embedding vectors in-place after an optimizer step.

        Called by ``nGPTModel.normalize_all_weights()`` after every call to
        ``optimizer.step()``. Normalizes all rows of the embedding matrix so
        each token's embedding vector has unit L2 norm.

        Uses ``weight.data.copy_()`` to preserve storage identity for the
        optimizer's momentum buffers. Wrapped in ``torch.no_grad()`` to
        prevent autograd tracking.
        """
        with torch.no_grad():
            self.weight.data.copy_(
                F.normalize(self.weight.data, p=2, dim=1)
            )

    def extra_repr(self) -> str:
        """Return a string with extra information for repr."""
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}"
        )


# ---------------------------------------------------------------------------
# HypersphereUpdate
# ---------------------------------------------------------------------------

class HypersphereUpdate:
    """Stateless utility class for geometric operations on the unit hypersphere.

    Implements the core nGPT update equations from Section 2.2:
        - LERP update (Equations 10/11): the default update used in all
          paper experiments.
        - SLERP update (Equation 6): the geodesic interpolation, available
          as an ablation (Table 6, Appendix A.9).
        - normalize: projects a vector back onto the unit hypersphere.

    All methods are ``@staticmethod`` because this class has no learnable
    parameters or instance state. It exists purely for namespace organization.

    The LERP update is the paper's default (Section 2.2):
        "Our experiments suggest that SLERP can be approximated by simple
        linear interpolation (LERP)"

    The SLERP update is available as an ablation that adds ~10% training time
    per step (Table 6) with negligible accuracy difference.

    Note on positivity of alpha:
        Both ``lerp_update`` and ``slerp_update`` apply ``torch.abs(alpha)``
        internally to enforce the positivity constraint from Section 2.5 and
        Appendix A.2: "alpha = |alpha| is used during the forward pass."
        This keeps the update semantics self-contained.
    """

    @staticmethod
    def normalize(x: Tensor) -> Tensor:
        """Project a tensor onto the unit hypersphere along the last dimension.

        Normalizes each vector along ``dim=-1`` to have unit L2 norm. This
        is the retraction step in Riemannian optimization (Appendix A.4):
        after an update step moves a point off the hypersphere, normalization
        maps it back to the manifold.

        Works correctly for any tensor shape:
            - Hidden states: ``(batch, seq_len, d_model)`` → normalizes each
              ``d_model``-dim vector.
            - Query/key vectors: ``(batch, n_heads, seq_len, d_k)`` → normalizes
              each ``d_k``-dim vector.

        Args:
            x: Input tensor of any shape. The last dimension is normalized.

        Returns:
            Tensor of the same shape as ``x``, with each vector along the
            last dimension having unit L2 norm. Zero vectors remain zero
            (F.normalize handles this gracefully).
        """
        return F.normalize(x, p=2, dim=-1)

    @staticmethod
    def lerp_update(
        h: Tensor,
        h_new: Tensor,
        alpha: Tensor,
    ) -> Tensor:
        """Update the hidden state via LERP followed by hypersphere projection.

        Implements Equations 10 and 11 from the paper:
            h ← Norm(h + α(h_new - h))

        This is a linear interpolation between ``h`` and ``h_new``, controlled
        by the eigen learning rate ``alpha``, followed by normalization back
        onto the unit hypersphere.

        The ``alpha`` tensor has shape ``(d_model,)`` and broadcasts over the
        batch and sequence dimensions of ``h``. Each dimension of the hidden
        state can move at a different rate toward ``h_new``.

        Positivity enforcement: ``torch.abs(alpha)`` is applied to ensure
        the update moves ``h`` toward ``h_new`` (not away from it). This
        implements the ``α = |α|`` convention from Appendix A.2.

        Args:
            h: Current hidden state tensor of shape
                ``(batch_size, seq_len, d_model)``. Should be unit norm
                (on the hypersphere) before calling this method.
            h_new: Target hidden state (output of attention or MLP block),
                shape ``(batch_size, seq_len, d_model)``. Should be unit norm
                (normalized by the attention/MLP block before this call).
            alpha: Eigen learning rate tensor of shape ``(d_model,)``.
                Typically the output of ``ScaledParameter.forward()``.
                Values are made positive via ``torch.abs()`` internally.

        Returns:
            Updated hidden state tensor of shape
            ``(batch_size, seq_len, d_model)`` with unit L2 norm along the
            last dimension (on the hypersphere).
        """
        # Enforce positivity: alpha = |alpha| (Appendix A.2)
        alpha_pos: Tensor = torch.abs(alpha)

        # Linear interpolation: h + alpha * (h_new - h)
        # alpha_pos has shape (d_model,), h has shape (batch, seq_len, d_model)
        # Broadcasting naturally aligns the last dimension.
        interpolated: Tensor = h + alpha_pos * (h_new - h)

        # Project back onto the unit hypersphere (retraction step)
        return F.normalize(interpolated, p=2, dim=-1)

    @staticmethod
    def slerp_update(
        h: Tensor,
        h_new: Tensor,
        alpha: Tensor,
    ) -> Tensor:
        """Update the hidden state via SLERP (geodesic interpolation).

        Implements Equation 6 from the paper:
            SLERP(a, b; α) = sin((1-α)θ)/sin(θ) * a + sin(αθ)/sin(θ) * b

        where θ = arccos(a · b) is the angle between the two unit vectors.

        This is the geodesic (shortest path) interpolation on the hypersphere.
        The paper notes it can be approximated by LERP (the default), but
        SLERP is available as an ablation (Table 6, Appendix A.9).

        Numerical stability:
            - ``dot`` is clamped to ``[-1+eps, 1-eps]`` before ``arccos`` to
              prevent NaN from floating-point values slightly outside [-1, 1].
            - When ``sin(theta) ≈ 0`` (vectors nearly parallel or antipodal),
              falls back to LERP to avoid division by zero.

        Args:
            h: Current hidden state tensor of shape
                ``(batch_size, seq_len, d_model)``. Should be unit norm.
            h_new: Target hidden state tensor of shape
                ``(batch_size, seq_len, d_model)``. Should be unit norm.
            alpha: Eigen learning rate tensor of shape ``(d_model,)``.
                Values are made positive via ``torch.abs()`` internally.

        Returns:
            Updated hidden state tensor of shape
            ``(batch_size, seq_len, d_model)`` with unit L2 norm along the
            last dimension.
        """
        # Enforce positivity: alpha = |alpha| (Appendix A.2)
        alpha_pos: Tensor = torch.abs(alpha)  # shape: (d_model,)

        # Compute the angle between h and h_new.
        # dot product along last dim, keep dim for broadcasting.
        # h and h_new are unit norm, so dot = cos(theta).
        dot: Tensor = (h * h_new).sum(dim=-1, keepdim=True)  # (batch, seq, 1)

        # Clamp to valid arccos domain to prevent NaN from floating-point errors
        dot_clamped: Tensor = dot.clamp(
            min=-1.0 + _SLERP_EPS,
            max=1.0 - _SLERP_EPS,
        )

        # Angle between the two vectors: theta in [0, pi]
        theta: Tensor = torch.arccos(dot_clamped)  # (batch, seq, 1)
        sin_theta: Tensor = torch.sin(theta)        # (batch, seq, 1)

        # SLERP coefficients.
        # alpha_pos has shape (d_model,), theta has shape (batch, seq, 1).
        # Broadcasting: alpha_pos * theta → (batch, seq, d_model).
        coeff_h: Tensor = (
            torch.sin((1.0 - alpha_pos) * theta) / (sin_theta + _SLERP_EPS)
        )  # (batch, seq, d_model)
        coeff_new: Tensor = (
            torch.sin(alpha_pos * theta) / (sin_theta + _SLERP_EPS)
        )  # (batch, seq, d_model)

        slerp_result: Tensor = coeff_h * h + coeff_new * h_new

        # Fallback to LERP when sin(theta) ≈ 0 (vectors nearly parallel or
        # antipodal). Use torch.where for a differentiable fallback.
        lerp_result: Tensor = h + alpha_pos * (h_new - h)

        # near_zero has shape (batch, seq, 1), broadcasts over d_model
        near_zero: Tensor = sin_theta.abs() < _SLERP_EPS
        result: Tensor = torch.where(near_zero, lerp_result, slerp_result)

        # Project back onto the unit hypersphere (retraction step)
        return F.normalize(result, p=2, dim=-1)

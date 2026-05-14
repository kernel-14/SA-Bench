```python
## model/embeddings.py
"""Positional and timestep embedding modules for Ca2-VDM.

This module implements four embedding components that are foundational to the
Ca2-VDM architecture:

1. SinusoidalEmbedding  — fixed sinusoidal lookup table (backbone for TPE/SPE).
2. TimestepEmbedding    — maps diffusion timestep t to a dim-dimensional vector;
                          enables cache sharing via tEmb(0) for clean prefix frames.
3. CyclicTPE            — cyclic temporal positional embeddings supporting
                          long autoregressive generation beyond training length.
4. SpatialPositionalEmbedding — 2D sinusoidal spatial positional embeddings
                                for the H×W latent grid.

All modules are free of internal project dependencies and can be imported
independently.

Paper: Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal
Generation and Cache Sharing (Sec. 3.2, 3.3).

Configuration references (config.yaml):
    model.model_dim:                    1152
    model.vae_downsample:               8
    evaluation.resolution:              256
    diffusion.T:                        1000
    video_prediction.ca2vdm_and_os_ext.l_train: 33
    t2v.l_train_max:                    65
    autoregressive.cyclic_tpe.enabled:  true
    autoregressive.clean_prefix_timestep: 0
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. SinusoidalEmbedding
# ---------------------------------------------------------------------------


class SinusoidalEmbedding(nn.Module):
    """Fixed sinusoidal positional embedding lookup table.

    Precomputes a table of shape ``(max_len, dim)`` using the standard
    Transformer sinusoidal formula (Vaswani et al., 2017):

        PE[pos, 2i]   = sin(pos / 10000^(2i / dim))
        PE[pos, 2i+1] = cos(pos / 10000^(2i / dim))

    The table is registered as a non-trainable buffer so it moves with the
    module's ``.to(device)`` call but is never updated by the optimizer.

    Used as the backbone for both :class:`CyclicTPE` (temporal) and
    :class:`SpatialPositionalEmbedding` (spatial).

    Attributes:
        dim: Embedding dimensionality. Must be even.
        max_len: Maximum number of positions in the lookup table.
        embedding_table: Buffer of shape ``(max_len, dim)`` containing the
                         precomputed sinusoidal embeddings.

    Example::

        emb = SinusoidalEmbedding(dim=1152, max_len=65)
        positions = torch.tensor([0, 1, 2, 32])
        out = emb(positions)  # shape (4, 1152)
    """

    def __init__(self, dim: int = 1152, max_len: int = 65) -> None:
        """Precompute the sinusoidal embedding table.

        Args:
            dim: Embedding dimensionality. Must be a positive even integer.
                 From config.yaml: ``model.model_dim: 1152``.
            max_len: Maximum sequence length (number of positions).
                     For CyclicTPE: ``l_train`` (33 or 65).
                     For SpatialPositionalEmbedding: ``H*W`` (1024 for 32×32).

        Raises:
            ValueError: If ``dim`` is not a positive even integer.
            ValueError: If ``max_len`` is not a positive integer.
        """
        super().__init__()

        if dim <= 0 or dim % 2 != 0:
            raise ValueError(
                f"dim must be a positive even integer, got {dim}."
            )
        if max_len <= 0:
            raise ValueError(
                f"max_len must be a positive integer, got {max_len}."
            )

        self.dim: int = dim
        self.max_len: int = max_len

        # Build the sinusoidal table of shape (max_len, dim).
        # Use log-space computation for numerical stability.
        positions: torch.Tensor = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        # div_term shape: (dim // 2,)
        div_term: torch.Tensor = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32)
            * -(math.log(10000.0) / dim)
        )

        table: torch.Tensor = torch.zeros(max_len, dim)
        table[:, 0::2] = torch.sin(positions * div_term)
        table[:, 1::2] = torch.cos(positions * div_term)
        # table: (max_len, dim)

        # Register as a buffer: moves with .to(device), not updated by optimizer.
        self.register_buffer("embedding_table", table)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Look up sinusoidal embeddings for the given positions.

        Args:
            positions: Integer tensor of arbitrary shape with values in
                       ``[0, max_len)``. Typically shape ``(L,)`` for a
                       sequence of length ``L``.

        Returns:
            Float tensor of shape ``positions.shape + (dim,)`` containing
            the precomputed sinusoidal embeddings. No gradient flows through
            this operation (buffer + integer indexing).

        Raises:
            IndexError: If any value in ``positions`` is outside
                        ``[0, max_len)``.
        """
        # embedding_table is a buffer; indexing with a long tensor is safe.
        return self.embedding_table[positions.long()]


# ---------------------------------------------------------------------------
# 2. TimestepEmbedding
# ---------------------------------------------------------------------------


class TimestepEmbedding(nn.Module):
    """Maps a scalar diffusion timestep to a ``dim``-dimensional embedding.

    This is the critical module enabling **cache sharing** in Ca2-VDM:
    clean prefix frames always receive ``tEmb(0)`` (a fixed, constant vector),
    while denoising target frames receive ``tEmb(t)`` for the current
    diffusion timestep ``t``. Because ``tEmb(0)`` is constant, the KV
    features of clean prefix frames are independent of ``t``, allowing the
    same KV-cache to be shared across all 100 denoising steps.

    Architecture (following DiT / PixArt-α):
        1. Sinusoidal frequency embedding: scalar ``t`` → ``(dim,)`` vector.
           Uses the same sinusoidal formula as :class:`SinusoidalEmbedding`
           but applied to a single scalar position.
        2. MLP refinement: ``Linear(dim, dim) → SiLU → Linear(dim, dim)``.
           This is the only learned part of the timestep embedding.

    The sinusoidal stage maps ``t ∈ [0, T]`` to a fixed ``dim``-dimensional
    vector. The MLP then refines this into a task-specific embedding.

    Attributes:
        dim: Embedding dimensionality. From config.yaml: ``model.model_dim: 1152``.
        mlp: Two-layer MLP (``Linear → SiLU → Linear``) for learned refinement.

    Example::

        t_emb = TimestepEmbedding(dim=1152)
        # Batch of timesteps, one per frame in a sequence.
        t_vec = torch.tensor([0, 0, 0, 500, 500, 500, 500, 500])  # (L,)
        out = t_emb(t_vec)  # shape (L, 1152)
        # t_vec[i] = 0 for clean prefix frames → constant tEmb(0) → cache sharing.
    """

    def __init__(self, dim: int = 1152) -> None:
        """Initialise the timestep embedding module.

        Args:
            dim: Embedding dimensionality. Must be a positive even integer.
                 From config.yaml: ``model.model_dim: 1152``.

        Raises:
            ValueError: If ``dim`` is not a positive even integer.
        """
        super().__init__()

        if dim <= 0 or dim % 2 != 0:
            raise ValueError(
                f"dim must be a positive even integer, got {dim}."
            )

        self.dim: int = dim

        # Learnable MLP: Linear(dim, dim) → SiLU → Linear(dim, dim).
        # Matches the DiT / PixArt-α timestep embedding design.
        self.mlp: nn.Sequential = nn.Sequential(
            nn.Linear(dim, dim, bias=True),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=True),
        )

        # Precompute the sinusoidal frequency table for timestep embedding.
        # We use T_max = 10000 to cover the full DDPM range [0, 1000] with
        # room to spare. The actual DDPM T=1000 is from config.yaml diffusion.T.
        # Using a larger table ensures t=0 and t=1000 are both well-represented.
        _T_max: int = 10000
        div_term: torch.Tensor = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32)
            * -(math.log(10000.0) / dim)
        )
        # Register as buffer so it moves with .to(device).
        self.register_buffer("_div_term", div_term)

    def _sinusoidal_embed(self, t: torch.Tensor) -> torch.Tensor:
        """Compute sinusoidal frequency embedding for a batch of timesteps.

        Applies the standard sinusoidal formula to scalar timestep values,
        producing a ``dim``-dimensional vector per timestep.

        Args:
            t: Float or integer tensor of shape ``(N,)`` containing timestep
               values. Values should be in ``[0, T]`` where ``T=1000``
               (config.yaml: ``diffusion.T: 1000``).

        Returns:
            Float32 tensor of shape ``(N, dim)`` containing the sinusoidal
            frequency embeddings. No gradient flows through this operation.
        """
        # Ensure float for multiplication.
        t_float: torch.Tensor = t.float().unsqueeze(1)  # (N, 1)
        # _div_term: (dim // 2,) — broadcast over N.
        args: torch.Tensor = t_float * self._div_term.unsqueeze(0)  # (N, dim//2)

        # Interleave sin and cos: [sin(t*w0), cos(t*w0), sin(t*w1), cos(t*w1), ...]
        emb: torch.Tensor = torch.zeros(
            t.shape[0], self.dim, dtype=torch.float32, device=t.device
        )
        emb[:, 0::2] = torch.sin(args)
        emb[:, 1::2] = torch.cos(args)

        return emb  # (N, dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute timestep embeddings for a batch of timestep values.

        Handles both 1-D ``(B,)`` and 2-D ``(B, L)`` input shapes, enabling
        per-frame timestep embeddings where each frame in a sequence can have
        a different timestep (e.g., ``t_vec[i, j] = 0`` for clean prefix
        frames and ``t_vec[i, j] = t`` for denoising target frames).

        Args:
            t: Integer or float tensor of shape ``(B,)`` or ``(B, L)``
               containing diffusion timestep values. Values in ``[0, T]``
               where ``T=1000`` (config.yaml: ``diffusion.T: 1000``).
               - ``t=0`` for clean prefix frames (cache sharing trick).
               - ``t ∈ [1, T]`` for denoising target frames.

        Returns:
            Float32 tensor of shape ``(B, dim)`` if input is ``(B,)``, or
            ``(B, L, dim)`` if input is ``(B, L)``. The output for ``t=0``
            is a fixed constant vector ``tEmb(0)`` that is the same across
            all calls, enabling KV-cache sharing.
        """
        original_shape: torch.Size = t.shape
        # Flatten to 1-D for uniform processing.
        t_flat: torch.Tensor = t.reshape(-1)  # (N,)

        # Stage 1: sinusoidal frequency embedding (fixed, no gradient).
        with torch.no_grad():
            sin_emb: torch.Tensor = self._sinusoidal_embed(t_flat)  # (N, dim)

        # Stage 2: learned MLP refinement.
        out: torch.Tensor = self.mlp(sin_emb)  # (N, dim)

        # Restore original batch/sequence shape.
        if len(original_shape) == 1:
            # Input was (B,) → output (B, dim).
            return out
        else:
            # Input was (B, L) → output (B, L, dim).
            return out.reshape(*original_shape, self.dim)


# ---------------------------------------------------------------------------
# 3. CyclicTPE
# ---------------------------------------------------------------------------


class CyclicTPE(nn.Module):
    """Cyclic Temporal Positional Embeddings for autoregressive video generation.

    Implements the Cyclic-TPE mechanism described in Section 3.3 of the paper.
    This module solves the TPE exhaustion problem that arises during long
    autoregressive generation when the KV-cache is active:

    **Problem:** The model is trained on sequences of length ``L_train``.
    During inference, the autoregressive generation can exceed ``L_train``
    frames. When the KV-cache queue is full (``P_k == P_max``), the oldest
    cached KVs are dequeued, but their TPE assignments are already baked into
    the cached K and V tensors. We cannot reassign TPEs to the cached KVs.

    **Solution (training):** Each training sample is assigned a TPE sequence
    that is cyclically shifted by a random offset ``δ ∈ [0, l_train)``.
    This teaches the model that TPE index ``k`` and ``(k + δ) % l_train``
    represent the same relative temporal position.

    **Solution (inference):** Once the queue is full (``P_k == P_max``),
    the new ``l``-frame chunk is assigned TPEs starting from index ``0``
    (cyclic wrap-around). The model has been trained to handle this via the
    cyclic shift training procedure.

    Attributes:
        dim: Embedding dimensionality. From config.yaml: ``model.model_dim: 1152``.
        l_train: Maximum training sequence length. Used as the period of the
                 cyclic shift. From config.yaml:
                 - ``video_prediction.ca2vdm_and_os_ext.l_train: 33``
                 - ``t2v.l_train_max: 65``
        base_emb: Underlying :class:`SinusoidalEmbedding` of shape
                  ``(l_train, dim)``.

    Example (training)::

        tpe = CyclicTPE(dim=1152, l_train=33)
        # Called once per training sample with the full sequence length.
        train_tpe = tpe.get_train_tpe(L=33, device=torch.device('cuda'))
        # train_tpe: (33, 1152) with a random cyclic offset applied.

    Example (inference)::

        tpe = CyclicTPE(dim=1152, l_train=33)
        # AR step 0: p_k=0, queue not full.
        chunk_tpe = tpe.get_inference_tpe(ar_step=0, l=8, p_k=0, p_max=25,
                                          device=torch.device('cuda'))
        # chunk_tpe: (8, 1152), indices [0, 1, 2, 3, 4, 5, 6, 7]

        # AR step 3: p_k=25=p_max, queue full → cyclic wrap-around.
        chunk_tpe = tpe.get_inference_tpe(ar_step=3, l=8, p_k=25, p_max=25,
                                          device=torch.device('cuda'))
        # chunk_tpe: (8, 1152), indices [0, 1, 2, 3, 4, 5, 6, 7]
    """

    def __init__(self, dim: int = 1152, l_train: int = 33) -> None:
        """Initialise the CyclicTPE module.

        Args:
            dim: Embedding dimensionality. Must be a positive even integer.
                 From config.yaml: ``model.model_dim: 1152``.
            l_train: Training sequence length (period of cyclic shift).
                     From config.yaml:
                     - ``video_prediction.ca2vdm_and_os_ext.l_train: 33``
                     - ``t2v.l_train_max: 65``

        Raises:
            ValueError: If ``dim`` is not a positive even integer.
            ValueError: If ``l_train`` is not a positive integer.
        """
        super().__init__()

        if dim <= 0 or dim % 2 != 0:
            raise ValueError(
                f"dim must be a positive even integer, got {dim}."
            )
        if l_train <= 0:
            raise ValueError(
                f"l_train must be a positive integer, got {l_train}."
            )

        self.dim: int = dim
        self.l_train: int = l_train

        # Underlying sinusoidal embedding table of shape (l_train, dim).
        # Not a learned parameter — only the buffer inside base_emb is stored.
        self.base_emb: SinusoidalEmbedding = SinusoidalEmbedding(
            dim=dim, max_len=l_train
        )

    def get_train_tpe(
        self,
        L: int,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Sample a cyclically shifted TPE sequence for training.

        Called once per training sample (not per batch). Each call samples a
        fresh random offset ``δ ∈ [0, l_train)``, producing a different cyclic
        shift for each sample. This trains the model to handle all possible
        cyclic assignments, preparing it for the inference-time wrap-around.

        Paper (Sec. 3.3): "in the training stage, each sample is assigned a
        TPE sequence that is cyclically shifted with a random offset."

        Args:
            L: Length of the TPE sequence to return. Must satisfy ``L <= l_train``.
               Typically equals ``config.l_train`` (the full training clip length).
            device: Target device for the output tensor.

        Returns:
            Float32 tensor of shape ``(L, dim)`` containing the cyclically
            shifted sinusoidal embeddings.

        Raises:
            ValueError: If ``L > l_train``.
        """
        if L > self.l_train:
            raise ValueError(
                f"L ({L}) must be <= l_train ({self.l_train}). "
                "The cyclic TPE table only covers l_train positions."
            )

        # Sample a random cyclic offset δ ∈ [0, l_train).
        delta: int = torch.randint(0, self.l_train, (1,)).item()  # type: ignore[assignment]

        indices: torch.Tensor = self._cyclic_shift(L=L, offset=int(delta), device=device)
        return self.base_emb(indices)  # (L, dim)

    def get_inference_tpe(
        self,
        ar_step: int,
        l: int,
        p_k: int,
        p_max: int,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Compute TPEs for a single autoregressive chunk during inference.

        Implements the two-regime cyclic assignment described in Sec. 3.3:

        **Regime 1 — queue not yet full (``p_k < p_max``):**
            The cumulative generation is still within the training length
            ``L_train = p_max + l``. Assign TPEs sequentially:
            ``indices[i] = p_k + i`` for ``i ∈ [0, l)``.
            These indices are in ``[0, l_train)`` since ``p_k + l <= p_max + l = l_train``.

        **Regime 2 — queue full (``p_k == p_max``):**
            The cumulative generation has exceeded ``l_train``. The oldest
            KV-cache chunk is being dequeued. Assign TPEs from index 0
            (cyclic wrap-around): ``indices[i] = i`` for ``i ∈ [0, l)``.
            This matches the training pattern where a cyclically shifted
            sample has the denoising target at the beginning of the TPE table.

        Paper (Sec. 3.3): "the denoising target will be assigned those TPEs
        indexed from the beginning."

        Args:
            ar_step: Current autoregressive step index (0-indexed). Used only
                     for logging/debugging; the actual regime is determined by
                     comparing ``p_k`` with ``p_max``.
            l: Chunk length (number of frames per AR step).
               From config.yaml: ``video_prediction.chunk_len: 8`` or
               ``t2v.chunk_len: 16``.
            p_k: Number of currently cached (generated) frames at this AR step.
                 Equals the current length of the KV-cache queue in frames.
                 - ``p_k = 0`` at AR step 0 (first chunk, no cache yet).
                 - ``p_k`` grows by ``l`` each AR step until it reaches ``p_max``.
            p_max: Maximum number of conditional frames (KV-cache queue capacity).
                   From config.yaml:
                   - ``video_prediction.ca2vdm_and_os_ext.p_max: 25``
                   - ``t2v.p_max: 49``
            device: Target device for the output tensor.

        Returns:
            Float32 tensor of shape ``(l, dim)`` containing the TPEs for the
            current ``l``-frame chunk.

        Raises:
            ValueError: If ``p_k > p_max``.
            ValueError: If ``l <= 0``.
        """
        if l <= 0:
            raise ValueError(f"l must be a positive integer, got {l}.")
        if p_k > p_max:
            raise ValueError(
                f"p_k ({p_k}) must be <= p_max ({p_max}). "
                "The KV-cache queue cannot exceed its maximum capacity."
            )

        if p_k < p_max:
            # Regime 1: sequential assignment starting from p_k.
            # Indices: [p_k, p_k+1, ..., p_k+l-1]
            # These are within [0, l_train) since p_k + l <= p_max + l = l_train.
            start_idx: int = p_k
            indices: torch.Tensor = self._cyclic_shift(
                L=l, offset=start_idx, device=device
            )
        else:
            # Regime 2: cyclic wrap-around — assign from index 0.
            # Indices: [0, 1, ..., l-1]
            # p_k == p_max: queue is full, oldest chunk being dequeued.
            indices = self._cyclic_shift(L=l, offset=0, device=device)

        return self.base_emb(indices)  # (l, dim)

    def _cyclic_shift(
        self,
        L: int,
        offset: int,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Compute cyclically shifted position indices.

        Returns a 1-D integer tensor of length ``L`` where element ``i`` is
        ``(offset + i) % l_train``. This implements the cyclic wrap-around
        that allows the model to handle sequences longer than ``l_train``.

        Args:
            L: Number of positions to generate.
            offset: Starting offset for the cyclic shift. Must be in
                    ``[0, l_train)``.
            device: Target device for the output tensor.

        Returns:
            Long tensor of shape ``(L,)`` with values in ``[0, l_train)``.
        """
        indices: torch.Tensor = torch.tensor(
            [(offset + i) % self.l_train for i in range(L)],
            dtype=torch.long,
            device=device,
        )
        return indices


# ---------------------------------------------------------------------------
# 4. SpatialPositionalEmbedding
# ---------------------------------------------------------------------------


class SpatialPositionalEmbedding(nn.Module):
    """2D sinusoidal spatial positional embeddings for the latent grid.

    Provides spatial positional information to the Transformer by encoding
    the 2D position ``(row, col)`` of each spatial token in the ``H×W``
    latent grid. Following ViT (Dosovitskiy et al., 2020) as cited in the
    paper: "sinusoidal spatial and temporal positional embeddings (i.e., SPEs
    and TPEs) are added to the frame sequence following Vision Transformer."

    The 2D sinusoidal embedding decomposes into row and column components:
    - Row embedding: sinusoidal over ``h`` positions, shape ``(h, dim//2)``.
    - Column embedding: sinusoidal over ``w`` positions, shape ``(w, dim//2)``.
    - Combined: for position ``(r, c)``, concatenate ``row_emb[r]`` and
      ``col_emb[c]``, giving a ``(h*w, dim)`` table.

    The spatial grid dimensions are derived from the config:
        ``h = w = resolution / vae_downsample = 256 / 8 = 32``
        ``h * w = 1024``

    The embedding table is registered as a non-trainable buffer.

    Attributes:
        dim: Embedding dimensionality. From config.yaml: ``model.model_dim: 1152``.
        h: Latent grid height. ``resolution / vae_downsample = 32``.
        w: Latent grid width. ``resolution / vae_downsample = 32``.
        num_patches: Total number of spatial tokens. ``h * w = 1024``.
        embedding_table: Buffer of shape ``(h*w, dim)`` containing the
                         precomputed 2D sinusoidal embeddings.

    Example::

        spe = SpatialPositionalEmbedding(dim=1152, h=32, w=32)
        # Get the full embedding table for adding to spatial tokens.
        emb = spe()  # shape (1, 1024, 1152) — ready for broadcasting over batch.

        # Add to spatial tokens x of shape (B, H*W, dim).
        x = x + spe()  # broadcasts over batch dimension.
    """

    def __init__(
        self,
        dim: int = 1152,
        h: int = 32,
        w: int = 32,
    ) -> None:
        """Precompute the 2D sinusoidal spatial embedding table.

        Args:
            dim: Embedding dimensionality. Must be a positive even integer.
                 From config.yaml: ``model.model_dim: 1152``.
            h: Latent grid height in tokens.
               ``resolution / vae_downsample = 256 / 8 = 32``
               (config.yaml: ``evaluation.resolution: 256``,
               ``model.vae_downsample: 8``).
            w: Latent grid width in tokens. Same as ``h`` for square grids.

        Raises:
            ValueError: If ``dim`` is not a positive even integer.
            ValueError: If ``h`` or ``w`` is not a positive integer.
        """
        super().__init__()

        if dim <= 0 or dim % 2 != 0:
            raise ValueError(
                f"dim must be a positive even integer, got {dim}."
            )
        if h <= 0 or w <= 0:
            raise ValueError(
                f"h and w must be positive integers, got h={h}, w={w}."
            )

        self.dim: int = dim
        self.h: int = h
        self.w: int = w
        self.num_patches: int = h * w

        # Build the 2D sinusoidal embedding table.
        # Each spatial position (r, c) gets a dim-dimensional embedding formed
        # by concatenating a (dim//2)-dimensional row embedding and a
        # (dim//2)-dimensional column embedding.
        half_dim: int = dim // 2

        # Row embeddings: shape (h, half_dim).
        row_emb: torch.Tensor = self._build_1d_sinusoidal(
            num_positions=h, dim=half_dim
        )
        # Column embeddings: shape (w, half_dim).
        col_emb: torch.Tensor = self._build_1d_sinusoidal(
            num_positions=w, dim=half_dim
        )

        # Combine row and column embeddings for all (r, c) pairs.
        # row_emb[r]: (half_dim,) — repeated for each column.
        # col_emb[c]: (half_dim,) — repeated for each row.
        # Result: (h*w, dim) where each row is [row_emb[r], col_emb[c]].
        row_expanded: torch.Tensor = row_emb.unsqueeze(1).expand(h, w, half_dim)
        # row_expanded: (h, w, half_dim)
        col_expanded: torch.Tensor = col_emb.unsqueeze(0).expand(h, w, half_dim)
        # col_expanded: (h, w, half_dim)

        # Concatenate along the last dimension and flatten spatial dims.
        table: torch.Tensor = torch.cat([row_expanded, col_expanded], dim=-1)
        # table: (h, w, dim)
        table = table.reshape(h * w, dim)
        # table: (h*w, dim)

        # Register as a non-trainable buffer.
        self.register_buffer("embedding_table", table)

    @staticmethod
    def _build_1d_sinusoidal(num_positions: int, dim: int) -> torch.Tensor:
        """Build a 1D sinusoidal embedding table.

        Uses the standard Transformer sinusoidal formula with log-space
        computation for numerical stability.

        Args:
            num_positions: Number of positions (rows or columns).
            dim: Embedding dimensionality for this component. Must be even.

        Returns:
            Float32 tensor of shape ``(num_positions, dim)``.
        """
        positions: torch.Tensor = torch.arange(
            num_positions, dtype=torch.float32
        ).unsqueeze(1)  # (num_positions, 1)

        div_term: torch.Tensor = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32)
            * -(math.log(10000.0) / dim)
        )  # (dim // 2,)

        table: torch.Tensor = torch.zeros(num_positions, dim)
        table[:, 0::2] = torch.sin(positions * div_term)
        table[:, 1::2]
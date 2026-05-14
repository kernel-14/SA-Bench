```python
## models/frvae/residual_quantizer.py
"""Frequency-guided Residual Quantizer for the FR-VAE.

Implements the residual quantization scheme described in Section 3.1.2 of the
NFIG paper. Given a list of frequency-decomposed feature maps {f_hat_i} from
FrequencyDecomposer, this module:

  1. Progressively computes residuals across frequency bands (low → high).
  2. Downsamples each residual+component to the band's token resolution.
  3. Quantizes via nearest-neighbor lookup in a shared codebook Z ∈ R^(K×C).
  4. Returns discrete token indices (for NFIGTransformer) and upsampled
     quantized representations (for VQGANDecoder via FrequencyDecomposer.compose).

Paper equations (Section 3.1.2):
    For i = 0:
        v_0 = argmin_v || f_hat_0 - T(v, H', W') ||^2
        R_0 = f_hat_0 - T(v_0^q, H', W')

    For i >= 1:
        v_i = argmin_v || (R_{i-1} + f_hat_i) - T(v, H', W') ||^2
        R_i = R_{i-1} + f_hat_i - T(v_i^q, H', W')

Config values used (config.yaml frvae section):
    codebook_size:       4096   (K)
    codebook_dim:        768    (C)
    scale_factors:       [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
    latent_spatial_size: 16     (H' = W')
    total_tokens:        680    (sum of s_i^2)
    commitment_loss_weight: 0.25 (beta)
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ResidualQuantizer(nn.Module):
    """Frequency-guided residual quantizer with a shared codebook.

    Converts a list of n frequency-band feature maps (each at full latent
    resolution H'×W') into discrete token indices via a progressive residual
    scheme. Each level captures what the previous levels failed to represent.

    A single shared codebook Z ∈ R^(K×C) is used across all 10 frequency
    bands. The codebook is implemented as nn.Embedding(K, C) and updated
    via gradient descent (codebook loss + commitment loss).

    Attributes:
        codebook_size: Number of codebook entries K = 4096.
        codebook_dim: Codebook vector dimension C = 768.
        scale_factors: List of n spatial sizes [s_0, ..., s_{n-1}].
            From config: [1, 2, 3, 4, 5, 6, 8, 10, 13, 16].
        H_prime: Full latent feature map height H' = 16.
        W_prime: Full latent feature map width W' = 16.
        num_scales: Number of frequency bands n = 10.
        commitment_loss_weight: Beta for commitment loss = 0.25.
        codebook: Shared nn.Embedding(K, C) — the learnable codebook Z.
    """

    # Commitment loss weight beta (from config.frvae.commitment_loss_weight = 0.25).
    # Standard VQ-VAE value; not explicitly stated in the paper for FR-VAE.
    _DEFAULT_COMMITMENT_WEIGHT: float = 0.25

    # Small epsilon for numerical stability in L2 distance computation.
    _DISTANCE_EPS: float = 1e-10

    def __init__(
        self,
        codebook_size: int = 4096,
        codebook_dim: int = 768,
        scale_factors: Optional[List[int]] = None,
        H_prime: int = 16,
        W_prime: int = 16,
        commitment_loss_weight: float = _DEFAULT_COMMITMENT_WEIGHT,
    ) -> None:
        """Initialize the ResidualQuantizer.

        Args:
            codebook_size: Number of codebook entries K.
                From config.frvae.codebook_size = 4096.
            codebook_dim: Codebook vector dimension C.
                From config.frvae.codebook_dim = 768.
                Must equal config.frvae.latent_channels = 768.
            scale_factors: List of n spatial sizes for each frequency band.
                scale_factors[i] = s_i defines a token grid of s_i × s_i.
                From config.frvae.scale_factors = [1,2,3,4,5,6,8,10,13,16].
                Must be strictly increasing and positive.
                scale_factors[-1] must equal H_prime (full resolution at last band).
            H_prime: Spatial height of the full latent feature map.
                From config.frvae.latent_spatial_size = 16.
            W_prime: Spatial width of the full latent feature map.
                From config.frvae.latent_spatial_size = 16.
            commitment_loss_weight: Beta coefficient for the commitment loss term.
                From config.frvae.commitment_loss_weight = 0.25.

        Raises:
            ValueError: If scale_factors is empty or contains invalid values.
            ValueError: If scale_factors[-1] != H_prime.
            ValueError: If codebook_size or codebook_dim are not positive.
        """
        super().__init__()

        # --- Input validation ---
        if scale_factors is None:
            scale_factors = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]

        if not scale_factors:
            raise ValueError("scale_factors must be a non-empty list.")

        if any(s <= 0 for s in scale_factors):
            raise ValueError(
                f"All scale_factors must be positive integers. Got: {scale_factors}"
            )

        for idx in range(1, len(scale_factors)):
            if scale_factors[idx] <= scale_factors[idx - 1]:
                raise ValueError(
                    f"scale_factors must be strictly increasing. "
                    f"Got scale_factors[{idx - 1}]={scale_factors[idx - 1]} >= "
                    f"scale_factors[{idx}]={scale_factors[idx]}. "
                    f"Full list: {scale_factors}"
                )

        if scale_factors[-1] != H_prime:
            raise ValueError(
                f"scale_factors[-1]={scale_factors[-1]} must equal H_prime={H_prime}. "
                "The highest-frequency band must operate at full latent resolution."
            )

        if codebook_size <= 0:
            raise ValueError(
                f"codebook_size must be positive, got {codebook_size}."
            )

        if codebook_dim <= 0:
            raise ValueError(
                f"codebook_dim must be positive, got {codebook_dim}."
            )

        if H_prime <= 0 or W_prime <= 0:
            raise ValueError(
                f"H_prime and W_prime must be positive, got H_prime={H_prime}, "
                f"W_prime={W_prime}."
            )

        # --- Store configuration ---
        self.codebook_size: int = codebook_size
        self.codebook_dim: int = codebook_dim
        self.scale_factors: List[int] = list(scale_factors)
        self.H_prime: int = H_prime
        self.W_prime: int = W_prime
        self.num_scales: int = len(scale_factors)
        self.commitment_loss_weight: float = commitment_loss_weight

        # --- Shared codebook Z ∈ R^(K×C) ---
        # Implemented as nn.Embedding for gradient-based updates.
        # All 10 frequency bands share this single codebook.
        self.codebook: nn.Embedding = nn.Embedding(codebook_size, codebook_dim)
        self._init_codebook()

        # --- Internal state for loss computation ---
        # Stores (v_i, v_i_q) pairs from the last encode_all() call.
        # Used by FRVAETrainer to compute codebook/commitment losses.
        # Shape: List of (Tensor(B,C,h_i,w_i), Tensor(B,C,h_i,w_i)) tuples.
        self._last_v_pairs: List[Tuple[Tensor, Tensor]] = []

    def _init_codebook(self) -> None:
        """Initialize codebook weights for stable VQ training.

        Uses uniform initialization in [-1/K, 1/K] to ensure codebook entries
        start spread across a small range. This is a common VQ-VAE initialization
        strategy that avoids codebook collapse in early training.

        The range [-1/K, 1/K] is intentionally small so that encoder outputs
        (which may have larger magnitude) are initially mapped to nearby entries,
        encouraging diverse codebook usage from the start.
        """
        limit: float = 1.0 / self.codebook_size
        nn.init.uniform_(self.codebook.weight, -limit, limit)

    def _downsample(self, f: Tensor, h: int, w: int) -> Tensor:
        """Downsample a full-resolution feature map to scale resolution.

        Implements the downsampling step in the residual quantization:
            v_i = downsample(target, h_i, w_i)

        Uses bilinear interpolation for smooth downsampling. When the target
        resolution equals the input resolution (h == H_prime, w == W_prime),
        this is effectively a no-op (interpolation to same size).

        Args:
            f: Feature map of shape (B, C, H', W') at full latent resolution.
            h: Target height (scale_factors[i] for band i).
            w: Target width (scale_factors[i] for band i).

        Returns:
            Downsampled feature map of shape (B, C, h, w).
        """
        # Fast path: no-op when already at target resolution.
        if f.shape[-2] == h and f.shape[-1] == w:
            return f

        return F.interpolate(
            f,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )

    def _upsample(self, v: Tensor, H: int, W: int) -> Tensor:
        """Upsample a scale-resolution feature map to full latent resolution.

        Implements T(v_i, H', W') from the paper — the interpolation function
        that maps v_i ∈ R^(B,C,h_i,w_i) back to R^(B,C,H',W').

        Uses bilinear interpolation for smooth upsampling. When the input
        resolution already equals (H, W), this is a no-op.

        Args:
            v: Feature map of shape (B, C, h, w) at scale resolution.
            H: Target height (H_prime = 16).
            W: Target width (W_prime = 16).

        Returns:
            Upsampled feature map of shape (B, C, H, W).
        """
        # Fast path: no-op when already at target resolution.
        if v.shape[-2] == H and v.shape[-1] == W:
            return v

        return F.interpolate(
            v,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

    def _lookup(self, v: Tensor) -> Tuple[Tensor, Tensor]:
        """Quantize a continuous feature map via nearest-neighbor codebook lookup.

        Implements the vector quantization step:
            t_i^(j,k) = lookup(Z, argmin_{z∈Z} ||z - v_i^(j,k)||_2)

        Uses the efficient squared L2 distance formula:
            ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b^T

        Applies the straight-through estimator (STE) so gradients flow through
        the quantization bottleneck:
            v_q_st = v + (v_q - v).detach()

        Args:
            v: Continuous feature map of shape (B, C, h, w).

        Returns:
            Tuple of:
                - v_q_st: Quantized feature map with STE, shape (B, C, h, w).
                  Gradients flow through this as if v_q_st = v.
                - token_indices: Discrete codebook indices, shape (B, h, w).
                  Integer tensor with values in [0, K-1].
        """
        B, C, h, w = v.shape
        N: int = B * h * w  # Total number of spatial positions

        # --- Step 1: Flatten spatial dimensions for distance computation ---
        # (B, C, h, w) → (B, h, w, C) → (N, C)
        v_flat: Tensor = v.permute(0, 2, 3, 1).reshape(N, C)

        # --- Step 2: Compute squared L2 distances to all codebook entries ---
        # Codebook weight Z: shape (K, C)
        codebook_weight: Tensor = self.codebook.weight  # (K, C)

        # Efficient distance computation using the identity:
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b^T
        # v_sq: (N, 1) — squared norms of encoder outputs
        v_sq: Tensor = v_flat.pow(2).sum(dim=1, keepdim=True)  # (N, 1)

        # z_sq: (1, K) — squared norms of codebook entries
        z_sq: Tensor = codebook_weight.pow(2).sum(dim=1, keepdim=True).t()  # (1, K)

        # cross: (N, K) — cross terms -2*v·Z^T
        cross: Tensor = -2.0 * torch.mm(v_flat, codebook_weight.t())  # (N, K)

        # distances: (N, K) — squared L2 distances, clamped to non-negative
        distances: Tensor = (v_sq + z_sq + cross).clamp(min=0.0)  # (N, K)

        # --- Step 3: Find nearest codebook entry for each position ---
        # indices_flat: (N,) — index of nearest codebook entry
        indices_flat: Tensor = distances.argmin(dim=1)  # (N,)

        # Reshape to spatial grid: (N,) → (B, h, w)
        token_indices: Tensor = indices_flat.reshape(B, h, w)

        # --- Step 4: Lookup quantized vectors from codebook ---
        # v_q_flat: (N, C) — quantized feature vectors
        v_q_flat: Tensor = self.codebook(indices_flat)  # (N, C)

        # Reshape back to spatial format: (N, C) → (B, h, w, C) → (B, C, h, w)
        v_q: Tensor = v_q_flat.reshape(B, h, w, C).permute(0, 3, 1, 2)  # (B, C, h, w)

        # --- Step 5: Straight-through estimator ---
        # Forward pass: v_q_st = v_q (quantized values)
        # Backward pass: gradients flow through v as if v_q_st = v
        # This allows the encoder to receive gradients despite the non-differentiable argmin.
        v_q_st: Tensor = v + (v_q - v).detach()  # (B, C, h, w)

        return v_q_st, token_indices

    def get_codebook_loss(self, v: Tensor, v_q: Tensor) -> Tensor:
        """Compute the combined codebook and commitment loss for one frequency band.

        Two loss components (standard VQ-VAE formulation):

        1. Codebook loss (moves codebook entries toward encoder outputs):
               L_codebook = ||sg(v) - v_q||^2
           where sg(·) is stop-gradient. This updates the codebook entries.

        2. Commitment loss (moves encoder outputs toward codebook entries):
               L_commit = beta * ||v - sg(v_q)||^2
           where beta = config.frvae.commitment_loss_weight = 0.25.
           This encourages the encoder to commit to codebook entries.

        Note: v_q here should be the raw quantized tensor (before STE), not
        the STE version. The STE version is used for the reconstruction path.

        Args:
            v: Continuous feature map before quantization, shape (B, C, h, w).
               This is the downsampled target passed to _lookup().
            v_q: Quantized feature map (codebook lookup result), shape (B, C, h, w).
               This is the raw v_q (not the STE version v_q_st).

        Returns:
            Scalar tensor: L_codebook + L_commit.
            This is summed across all frequency bands by FRVAETrainer.
        """
        # Codebook loss: update codebook entries toward encoder outputs.
        # sg(v) means v.detach() — no gradient flows to encoder through this term.
        codebook_loss: Tensor = F.mse_loss(v.detach(), v_q)

        # Commitment loss: update encoder outputs toward codebook entries.
        # sg(v_q) means v_q.detach() — no gradient flows to codebook through this term.
        commitment_loss: Tensor = self.commitment_loss_weight * F.mse_loss(
            v, v_q.detach()
        )

        return codebook_loss + commitment_loss

    def quantize(
        self,
        f_hat_i: Tensor,
        residual_prev: Tensor,
        scale_idx: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Quantize a single frequency band using the residual scheme.

        Implements one step of the residual quantization loop from Section 3.1.2:

        For i = 0:
            target = f_hat_0
            v_0 = downsample(target, h_0, w_0)
            v_0_q, idx_0 = lookup(v_0)
            v_0_q_up = upsample(v_0_q, H', W')
            R_0 = target - v_0_q_up.detach()

        For i >= 1:
            target = R_{i-1} + f_hat_i
            v_i = downsample(target, h_i, w_i)
            v_i_q, idx_i = lookup(v_i)
            v_i_q_up = upsample(v_i_q, H', W')
            R_i = target - v_i_q_up.detach()

        The residual R_i is detached from the computation graph to prevent
        gradient explosion through the residual chain. Each level's quantization
        is trained independently via its own codebook/commitment loss.

        Args:
            f_hat_i: Frequency component at full resolution, shape (B, C, H', W').
                Output of FrequencyDecomposer.decompose() for band i.
            residual_prev: Accumulated residual from previous levels, shape (B, C, H', W').
                Should be zeros for i=0 (first frequency band).
                Should be R_{i-1} for i >= 1.
            scale_idx: Index into self.scale_factors. Determines the token grid
                resolution h_i = w_i = scale_factors[scale_idx].

        Returns:
            Tuple of four tensors:
                - v_i_q: Quantized feature map with STE, shape (B, C, h_i, w_i).
                  Used for codebook/commitment loss computation.
                - token_indices: Discrete token indices, shape (B, h_i, w_i).
                  Integer tensor with values in [0, K-1].
                  Consumed by NFIGTransformer during training.
                - v_i: Continuous (pre-quantization) feature map, shape (B, C, h_i, w_i).
                  Used for codebook/commitment loss computation.
                - residual_i: Updated residual at full resolution, shape (B, C, H', W').
                  Input to the next level's quantize() call.
                  Detached from the computation graph.

        Raises:
            IndexError: If scale_idx is out of range [0, num_scales - 1].
        """
        if scale_idx < 0 or scale_idx >= self.num_scales:
            raise IndexError(
                f"scale_idx={scale_idx} is out of range [0, {self.num_scales - 1}]."
            )

        # Get target spatial resolution for this frequency band.
        h: int = self.scale_factors[scale_idx]
        w: int = self.scale_factors[scale_idx]

        # --- Compute quantization target at full resolution ---
        # For i=0: target = f_hat_0
        # For i>=1: target = R_{i-1} + f_hat_i
        target: Tensor = residual_prev + f_hat_i  # (B, C, H', W')

        # --- Downsample target to scale resolution ---
        # v_i = downsample(target, h_i, w_i)
        # Shape: (B, C, H', W') → (B, C, h, w)
        v_i: Tensor = self._downsample(target, h, w)

        # --- Quantize via nearest-neighbor codebook lookup ---
        # v_i_q: (B, C, h, w) with STE — gradients flow through as if v_i_q = v_i
        # token_indices: (B, h, w) — discrete indices in [0, K-1]
        v_i_q: Tensor
        token_indices: Tensor
        v_i_q, token_indices = self._lookup(v_i)

        # --- Upsample quantized representation back to full resolution ---
        # T(v_i^q, H', W'): (B, C, h, w) → (B, C, H', W')
        v_i_q_up: Tensor = self._upsample(v_i_q, self.H_prime, self.W_prime)

        # --- Compute updated residual ---
        # R_i = target - T(v_i^q, H', W')
        # Detach v_i_q_up to prevent gradient flow through the residual chain.
        # Each level is trained independently via its own loss contribution.
        residual_i: Tensor = (target - v_i_q_up.detach()).detach()

        return v_i_q, token_indices, v_i, residual_i

    def encode_all(
        self,
        freq_components: List[Tensor],
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Encode all n frequency bands via progressive residual quantization.

        Processes all n=10 frequency bands sequentially from lowest to highest
        frequency. Each band's quantization target is the accumulated residual
        from all previous bands plus the current band's frequency component.

        This method also stores the (v_i, v_i_q) pairs in self._last_v_pairs
        for use by FRVAETrainer when computing codebook/commitment losses.

        Args:
            freq_components: List of n tensors from FrequencyDecomposer.decompose().
                Each tensor has shape (B, C, H', W').
                Ordered from lowest frequency (index 0) to highest (index n-1).
                len(freq_components) must equal self.num_scales = 10.

        Returns:
            Tuple of two lists:
                - token_indices_list: List of n token index tensors.
                  token_indices_list[i] has shape (B, h_i, w_i) = (B, s_i, s_i).
                  Integer tensors with values in [0, K-1].
                  Shapes: (B,1,1), (B,2,2), (B,3,3), ..., (B,16,16).
                  Consumed by NFIGTransformer during training.

                - quantized_upsampled_list: List of n upsampled quantized tensors.
                  quantized_upsampled_list[i] has shape (B, C, H', W').
                  These are T(v_i^q, H', W') — quantized representations at full res.
                  Summed by FrequencyDecomposer.compose() to get f_tilde.

        Raises:
            ValueError: If len(freq_components) != self.num_scales.
            ValueError: If any component has incorrect spatial dimensions.
        """
        if len(freq_components) != self.num_scales:
            raise ValueError(
                f"Expected {self.num_scales} frequency components, "
                f"got {len(freq_components)}. "
                f"scale_factors={self.scale_factors}"
            )

        # Validate spatial dimensions of all components.
        for idx, component in enumerate(freq_components):
            if component.shape[-2] != self.H_prime or component.shape[-1] != self.W_prime:
                raise ValueError(
                    f"freq_components[{idx}] has spatial dimensions "
                    f"{component.shape[-2]}×{component.shape[-1]}, "
                    f"expected {self.H_prime}×{self.W_prime}. "
                    f"Full shape: {tuple(component.shape)}."
                )

        # Infer batch size and device from the first component.
        B: int = freq_components[0].shape[0]
        C: int = freq_components[0].shape[1]
        device: torch.device = freq_components[0].device
        dtype: torch.dtype = freq_components[0].dtype

        # --- Initialize residual to zero (R_{-1} = 0) ---
        # Shape: (B, C, H', W') — same as frequency components.
        residual: Tensor = torch.zeros(
            B, C, self.H_prime, self.W_prime,
            device=device,
            dtype=dtype,
        )

        # --- Output accumulators ---
        token_indices_list: List[Tensor] = []
        quantized_upsampled_list: List[Tensor] = []

        # --- Store (v_i, v_i_q) pairs for loss computation ---
        # Reset from previous call.
        self._last_v_pairs = []

        # --- Progressive residual quantization loop ---
        for scale_idx in range(self.num_scales):
            f_hat_i: Tensor = freq_components[scale_idx]  # (B, C, H', W')

            # Quantize this frequency band.
            # v_i_q: (B, C, h_i, w_i) with STE
            # token_indices_i: (B, h_i, w_i)
            # v_i: (B, C, h_i, w_i) continuous (pre-quantization)
            # residual: (B, C, H', W') updated residual for next level
            v_i_q: Tensor
            token_indices_i: Tensor
            v_i: Tensor
            v_i_q, token_indices_i, v_i, residual = self.quantize(
                f_hat_i=f_hat_i,
                residual_prev=residual,
                scale_idx=scale_idx,
            )

            # Upsample quantized representation to full resolution.
            # T(v_i^q, H', W'): (B, C, h_i, w_i) → (B, C, H', W')
            v_i_q_up: Tensor = self._upsample(v_i_q, self.H_prime, self.W_prime)

            # Accumulate outputs.
            token_indices_list.append(token_indices_i)
            quantized_upsampled_list.append(v_i_q_up)

            # Store (v_i, v_i_q) pair for loss computation by FRVAETrainer.
            # v_i_q here is the STE version; for loss computation we need the
            # raw quantized values. We store v_i (continuous) and v_i_q (STE).
            # The trainer uses get_codebook_loss(v_i, v_i_q) where v_i_q is
            # the STE version — this is correct because:
            #   codebook_loss = mse(v_i.detach(), v_i_q)  → updates codebook
            #   commitment_loss = beta * mse(v_i, v_i_q.detach())  → updates encoder
            # The STE version v_i_q has the same values as raw v_q in the forward pass.
            self._last_v_pairs.append((v_i, v_i_q))

        return token_indices_list, quantized_upsampled_list

    def decode_all(self, token_indices: List[Tensor]) -> Tensor:
        """Decode a list of token index tensors into a composed feature map.

        Converts discrete token indices back to continuous feature vectors via
        codebook lookup, upsamples each to full resolution, and sums them to
        produce the reconstructed latent feature map f_tilde.

        This is the inference-time reconstruction path. No residual computation
        is needed — the reconstruction is simply the sum of all upsampled
        quantized representations:
            f_tilde = Σ_i T(lookup(Z, token_indices_i), H', W')

        Args:
            token_indices: List of n token index tensors.
                token_indices[i] has shape (B, h_i, w_i) = (B, s_i, s_i).
                Integer tensors with values in [0, K-1].
                Shapes: (B,1,1), (B,2,2), (B,3,3), ..., (B,16,16).

        Returns:
            Reconstructed latent feature map f_tilde of shape (B, C, H', W').
            For the default config: (B, 768, 16, 16).
            This is passed to VQGANDecoder.forward() to produce the final image.

        Raises:
            ValueError: If len(token_indices) != self.num_scales.
            ValueError: If any token index tensor has incorrect spatial dimensions.
        """
        if len(token_indices) != self.num_scales:
            raise ValueError(
                f"Expected {self.num_scales} token index tensors, "
                f"got {len(token_indices)}. "
                f"scale_factors={self.scale_factors}"
            )

        # Validate shapes and infer batch size from the first tensor.
        # token_indices[i] should have shape (B, s_i, s_i).
        for idx, indices in enumerate(token_indices):
            expected_h: int = self.scale_factors[idx]
            expected_w: int = self.scale_factors[idx]
            if indices.shape[-2] != expected_h or indices.shape[-1] != expected_w:
                raise ValueError(
                    f"token_indices[{idx}] has spatial dimensions "
                    f"{indices.shape[-2]}×{indices.shape[-1]}, "
                    f"expected {expected_h}×{expected_w} "
                    f"(scale_factors[{idx}]={self.scale_factors[idx]}). "
                    f"Full shape: {tuple(indices.shape)}."
                )

        B: int = token_indices[0].shape[0]
        device: torch.device = token_indices[0].device

        # Initialize the composed feature map to zero.
        # Shape: (B, C, H', W')
        f_tilde: Tensor = torch
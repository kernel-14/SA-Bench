## inference/sampler.py
"""Autoregressive sampler for the NFIG framework.

Implements the 10-step frequency-band generation loop described in Section 3.2
of the NFIG paper. At each step, the NFIGTransformer predicts the next frequency
band's tokens conditioned on all previously generated bands, with Classifier-Free
Guidance (CFG) and top-k filtering applied before sampling.

Inference configuration (paper Section 4.1):
    cfg_scale: 4.5   (config.nfig.cfg_scale)
    top_k:     990   (config.nfig.top_k)
    steps:     10    (one per frequency band, config.nfig.num_generation_steps)

Generation process (Section 3.2):
    p(T_1, T_2, ..., T_n) = Π_i p(T_i | T_1, T_2, ..., T_{i-1})

where T_i ∈ [K]^(h_i × w_i) is the token matrix for frequency band i.
All tokens within a band are predicted in parallel (next-frequency prediction).

CFG formula (standard):
    logits_cfg = logits_uncond + scale * (logits_cond - logits_uncond)
"""

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from models.frvae.frvae import FRVAE
from models.transformer.nfig_transformer import NFIGTransformer
from utils.config import NFIGConfig


class NFIGSampler:
    """Autoregressive sampler for NFIG image generation.

    Drives the 10-step frequency-band generation loop using a trained
    NFIGTransformer and FR-VAE decoder. Implements CFG and top-k sampling
    as described in paper Section 4.1.

    Both models are kept in eval mode with no gradient tracking throughout
    all sampling operations.

    Attributes:
        transformer: Trained NFIGTransformer in eval mode (no gradients).
        frvae: Trained FRVAE tokenizer/decoder in eval mode (no gradients).
        config: NFIGConfig with inference hyperparameters.
        scale_factors: List of n scale factors from config.
            From config.nfig.scale_factors = [1,2,3,4,5,6,8,10,13,16].
        band_sizes: Per-band token counts [s_i^2 for s_i in scale_factors].
            For default config: [1, 4, 9, 16, 25, 36, 64, 100, 169, 256].
        band_offsets: Cumulative token offsets into the flat sequence.
            For default config: [0, 1, 5, 14, 30, 55, 91, 155, 255, 424, 680].
        num_bands: Number of frequency bands n = 10.
        device: Device inferred from transformer parameters.
    """

    # Default per-call batch size for sample_batch() chunking.
    # Chosen to fit comfortably in GPU memory for 310M parameter model.
    _DEFAULT_CHUNK_SIZE: int = 16

    def __init__(
        self,
        transformer: NFIGTransformer,
        frvae: FRVAE,
        config: NFIGConfig,
    ) -> None:
        """Initialize the NFIGSampler.

        Sets both models to eval mode with no gradient tracking, and
        precomputes per-band token counts and cumulative offsets for
        efficient indexing during the generation loop.

        Args:
            transformer: Trained NFIGTransformer. Will be set to eval()
                and requires_grad_(False). Must be on the target device
                before passing to this constructor.
            frvae: Trained FRVAE tokenizer/decoder. Will be set to eval()
                and requires_grad_(False). Must be on the same device as
                transformer.
            config: NFIGConfig dataclass populated from config.yaml nfig
                section. Key inference values:
                  - cfg_scale:       4.5   (paper Section 4.1)
                  - top_k:           990   (paper Section 4.1)
                  - scale_factors:   [1,2,3,4,5,6,8,10,13,16]
                  - codebook_size:   4096
                  - num_classes:     1000
                  - null_class_id:   1000
                  - num_frequency_bands: 10
        """
        # --- Store references ---
        self.transformer: NFIGTransformer = transformer
        self.frvae: FRVAE = frvae
        self.config: NFIGConfig = config

        # --- Freeze both models for inference ---
        # eval() disables dropout and uses running stats for any BN layers.
        # requires_grad_(False) prevents gradient tape allocation during sampling.
        self.transformer.eval()
        self.transformer.requires_grad_(False)
        self.frvae.eval()
        self.frvae.requires_grad_(False)

        # --- Precompute band metadata from scale_factors ---
        # scale_factors: [1, 2, 3, 4, 5, 6, 8, 10, 13, 16] (from config)
        self.scale_factors: List[int] = list(config.scale_factors)
        self.num_bands: int = len(self.scale_factors)

        # band_sizes[i] = scale_factors[i]^2 = number of tokens in band i.
        # For default config: [1, 4, 9, 16, 25, 36, 64, 100, 169, 256]
        # Sum = 680 = config.nfig.total_tokens
        self.band_sizes: List[int] = [s * s for s in self.scale_factors]

        # band_offsets[i] = starting index of band i in the flat token sequence.
        # band_offsets[n] = total_tokens (sentinel for slicing).
        # For default config: [0, 1, 5, 14, 30, 55, 91, 155, 255, 424, 680]
        self.band_offsets: List[int] = [0]
        cumulative: int = 0
        for size in self.band_sizes:
            cumulative += size
            self.band_offsets.append(cumulative)

        # Verify total token count matches config.
        assert self.band_offsets[-1] == config.total_tokens, (
            f"Computed total tokens {self.band_offsets[-1]} does not match "
            f"config.total_tokens={config.total_tokens}. "
            f"scale_factors={self.scale_factors}"
        )

        # --- Infer device from transformer parameters ---
        # Both models must be on the same device; we use transformer as reference.
        try:
            self.device: torch.device = next(self.transformer.parameters()).device
        except StopIteration:
            # Fallback if transformer has no parameters (should not happen).
            self.device = torch.device("cpu")

    @torch.no_grad()
    def sample(
        self,
        class_labels: Tensor,
        cfg_scale: float = 4.5,
        top_k: int = 990,
    ) -> Tensor:
        """Generate one image per class label using autoregressive frequency prediction.

        Implements the 10-step generation loop from paper Section 3.2:
            p(T_1, ..., T_n) = Π_i p(T_i | T_1, ..., T_{i-1})

        At each step i, the transformer predicts band i's tokens conditioned
        on all previously generated bands (0..i-1). CFG and top-k filtering
        are applied before sampling.

        Args:
            class_labels: Class label tensor of shape (B,), dtype torch.long.
                Values in [0, config.num_classes - 1] = [0, 999] for ImageNet.
                Must be on the same device as the models.
            cfg_scale: Classifier-Free Guidance scale.
                From config.nfig.cfg_scale = 4.5 (paper Section 4.1).
                cfg_scale=0.0 means unconditional generation.
                cfg_scale=1.0 means no guidance (conditional only).
                cfg_scale=4.5 is the paper's optimal value.
            top_k: Top-k filtering parameter.
                From config.nfig.top_k = 990 (paper Section 4.1).
                Only the top-k logits are kept before softmax sampling.
                Must be in [1, config.codebook_size] = [1, 4096].

        Returns:
            Generated image batch of shape (B, 3, 256, 256).
            Values are in [-1, 1] (tanh output from VQGANDecoder).
            B = class_labels.shape[0].

        Raises:
            ValueError: If class_labels contains values outside [0, num_classes-1].
            ValueError: If cfg_scale < 0.
            ValueError: If top_k is not in [1, codebook_size].
        """
        # --- Input validation ---
        if cfg_scale < 0.0:
            raise ValueError(
                f"cfg_scale must be non-negative, got {cfg_scale}. "
                "cfg_scale=0.0 means unconditional generation; "
                "cfg_scale=4.5 is the paper's recommended value."
            )

        if not (1 <= top_k <= self.config.codebook_size):
            raise ValueError(
                f"top_k={top_k} must be in [1, codebook_size={self.config.codebook_size}]. "
                "From config.nfig.top_k = 990."
            )

        if class_labels.dim() != 1:
            raise ValueError(
                f"class_labels must be a 1D tensor of shape (B,), "
                f"got shape {tuple(class_labels.shape)}."
            )

        # Validate class label range.
        if class_labels.numel() > 0:
            min_label: int = class_labels.min().item()
            max_label: int = class_labels.max().item()
            if min_label < 0 or max_label >= self.config.num_classes:
                raise ValueError(
                    f"class_labels values must be in [0, {self.config.num_classes - 1}]. "
                    f"Got min={min_label}, max={max_label}. "
                    "Use config.nfig.null_class_id=1000 for unconditional generation."
                )

        B: int = class_labels.shape[0]

        # --- Autoregressive generation loop ---
        # token_seqs accumulates generated tokens band by band.
        # At step i: token_seqs has i tensors (bands 0..i-1).
        # After the loop: token_seqs has 10 tensors (all bands).
        token_seqs: List[Tensor] = []

        for band_idx in range(self.num_bands):
            # Generate tokens for band band_idx conditioned on token_seqs (bands 0..band_idx-1).
            # new_tokens: shape [B, band_sizes[band_idx]]
            new_tokens: Tensor = self._sample_next_band(
                token_seqs=token_seqs,
                class_labels=class_labels,
                band_idx=band_idx,
                cfg_scale=cfg_scale,
                top_k=top_k,
            )

            # Append generated tokens for this band to the context.
            token_seqs.append(new_tokens)

        # --- Decode token sequences to images ---
        # token_seqs: List of 10 tensors with shapes [B,1], [B,4], ..., [B,256]
        # frvae.tokens_to_image() handles:
        #   1. Reshape [B, h_i*w_i] → [B, h_i, w_i] for each band
        #   2. Codebook lookup → quantized feature maps
        #   3. Upsample and compose → f_tilde [B, 768, 16, 16]
        #   4. Decode → x_hat [B, 3, 256, 256]
        # frvae.tokens_to_image is decorated with @torch.no_grad() internally.
        images: Tensor = self.frvae.tokens_to_image(token_seqs)

        return images

    @torch.no_grad()
    def sample_batch(
        self,
        num_samples: int,
        class_label: int,
        cfg_scale: float = 4.5,
        top_k: int = 990,
    ) -> Tensor:
        """Generate multiple images of a single class, chunked for memory efficiency.

        Convenience wrapper around sample() that handles large generation requests
        by splitting into smaller chunks. Useful for generating the 50,000 samples
        needed for FID/IS evaluation (config.evaluation.num_samples = 50000).

        Args:
            num_samples: Total number of images to generate.
                For FID evaluation: 50000 (config.evaluation.num_samples).
                For qualitative inspection: any positive integer.
            class_label: Single ImageNet class index in [0, 999].
                All generated images will be conditioned on this class.
            cfg_scale: CFG scale. From config.nfig.cfg_scale = 4.5.
            top_k: Top-k filtering. From config.nfig.top_k = 990.

        Returns:
            Generated image tensor of shape (num_samples, 3, 256, 256).
            Values are in [-1, 1].

        Raises:
            ValueError: If num_samples <= 0.
            ValueError: If class_label is not in [0, num_classes - 1].
        """
        if num_samples <= 0:
            raise ValueError(
                f"num_samples must be a positive integer, got {num_samples}."
            )

        if not (0 <= class_label < self.config.num_classes):
            raise ValueError(
                f"class_label={class_label} must be in "
                f"[0, {self.config.num_classes - 1}]. "
                f"Got {class_label}."
            )

        # Collect generated image chunks.
        all_images: List[Tensor] = []
        remaining: int = num_samples

        while remaining > 0:
            # Determine chunk size for this iteration.
            chunk_size: int = min(self._DEFAULT_CHUNK_SIZE, remaining)

            # Create class label tensor for this chunk.
            # All samples in the chunk use the same class label.
            chunk_labels: Tensor = torch.full(
                (chunk_size,),
                fill_value=class_label,
                dtype=torch.long,
                device=self.device,
            )

            # Generate images for this chunk.
            chunk_images: Tensor = self.sample(
                class_labels=chunk_labels,
                cfg_scale=cfg_scale,
                top_k=top_k,
            )  # [chunk_size, 3, 256, 256]

            all_images.append(chunk_images.cpu())  # Move to CPU to free GPU memory
            remaining -= chunk_size

        # Concatenate all chunks along the batch dimension.
        # Shape: (num_samples, 3, 256, 256)
        result: Tensor = torch.cat(all_images, dim=0)

        return result

    def _sample_next_band(
        self,
        token_seqs: List[Tensor],
        class_labels: Tensor,
        band_idx: int,
        cfg_scale: float,
        top_k: int,
    ) -> Tensor:
        """Sample tokens for a single frequency band using CFG and top-k.

        Core sampling logic for one step of the autoregressive generation loop.
        Uses the batched CFG trick: both conditional and unconditional forward
        passes are computed in a single transformer call by doubling the batch.

        The transformer receives token_seqs (bands 0..band_idx-1) as context
        and outputs logits for all token positions. We extract the logits
        corresponding to band band_idx, apply CFG, then sample.

        Args:
            token_seqs: List of band_idx tensors (previously generated bands).
                token_seqs[i] has shape (B, band_sizes[i]) for i in [0, band_idx).
                Empty list for band_idx=0 (no context for the first band).
            class_labels: Conditional class labels of shape (B,), dtype torch.long.
                Values in [0, num_classes - 1].
            band_idx: Index of the frequency band to generate (0-based).
                Must be in [0, num_bands - 1] = [0, 9].
            cfg_scale: CFG scale factor (4.5 from config).
            top_k: Top-k filtering parameter (990 from config).

        Returns:
            Sampled token indices for band band_idx of shape (B, band_sizes[band_idx]).
            dtype: torch.long, values in [0, codebook_size - 1] = [0, 4095].

        Raises:
            IndexError: If band_idx is out of range [0, num_bands - 1].
            RuntimeError: If token_seqs has incorrect length (must equal band_idx).
        """
        if band_idx < 0 or band_idx >= self.num_bands:
            raise IndexError(
                f"band_idx={band_idx} is out of range [0, {self.num_bands - 1}]."
            )

        if len(token_seqs) != band_idx:
            raise RuntimeError(
                f"token_seqs has {len(token_seqs)} tensors but band_idx={band_idx}. "
                f"token_seqs must contain exactly band_idx={band_idx} tensors "
                "(one per previously generated band)."
            )

        B: int = class_labels.shape[0]
        K: int = self.config.codebook_size  # 4096
        band_size: int = self.band_sizes[band_idx]  # h_i * w_i tokens for this band

        # ------------------------------------------------------------------ #
        # Batched CFG: double the batch to compute both passes in one call
        # ------------------------------------------------------------------ #
        # Create null class labels for the unconditional pass.
        # null_class_id = 1000 (config.nfig.null_class_id = num_classes = 1000)
        null_labels: Tensor = torch.full(
            (B,),
            fill_value=self.config.null_class_id,
            dtype=torch.long,
            device=self.device,
        )

        # Concatenate conditional and unconditional labels along batch dim.
        # doubled_labels: shape [2B]
        # First B entries: real class labels (conditional)
        # Last B entries: null class labels (unconditional)
        doubled_labels: Tensor = torch.cat(
            [class_labels, null_labels], dim=0
        )  # [2B]

        # Duplicate token_seqs: each tensor [B, h_j*w_j] → [2B, h_j*w_j]
        # by concatenating with itself along the batch dimension.
        doubled_token_seqs: List[Tensor] = [
            torch.cat([t, t], dim=0) for t in token_seqs
        ]  # List of band_idx tensors, each [2B, h_j*w_j]

        # ------------------------------------------------------------------ #
        # Single transformer forward pass for both conditional and unconditional
        # ------------------------------------------------------------------ #
        # transformer.forward() returns logits for all token positions in the
        # current context (bands 0..band_idx-1) plus the next band (band_idx).
        # The block-wise causal mask inside the transformer ensures that logits
        # at band band_idx positions are conditioned only on bands 0..band_idx-1.
        #
        # Output shape: [2B, total_context_len, K]
        # where total_context_len = sum(band_sizes[0..band_idx]) includes
        # the current band's positions (the transformer predicts them given context).
        #
        # Note: The transformer is designed to predict the NEXT band given the
        # current context. When token_seqs has band_idx tensors (bands 0..band_idx-1),
        # the transformer outputs logits for all positions including band band_idx.
        # We extract the logits at band band_idx's positions.
        logits_all: Tensor = self.transformer(
            doubled_token_seqs, doubled_labels
        )  # [2B, total_context_len, K]

        # ------------------------------------------------------------------ #
        # Extract logits for band band_idx
        # ------------------------------------------------------------------ #
        # The transformer output includes logits for all token positions in the
        # sequence. Band band_idx's logits are at positions:
        #   [band_offsets[band_idx] : band_offsets[band_idx + 1]]
        # within the output sequence.
        #
        # However, when token_seqs has band_idx tensors (context = bands 0..band_idx-1),
        # the output sequence length is sum(band_sizes[0..band_idx]).
        # The last band_size positions correspond to band band_idx's predictions.
        #
        # We use the last band_size positions of the output for robustness:
        # logits_all[:, -band_size:, :] gives band band_idx's logits.
        # This works because the transformer outputs logits for all positions
        # including the "next" band positions at the end of the sequence.
        #
        # Shape: [2B, band_size, K]
        logits_band: Tensor = logits_all[:, -band_size:, :]  # [2B, band_size, K]

        # Split into conditional and unconditional logits.
        # First B entries: conditional (real class labels)
        # Last B entries: unconditional (null class labels)
        logits_cond: Tensor = logits_band[:B]    # [B, band_size, K]
        logits_uncond: Tensor = logits_band[B:]  # [B, band_size, K]

        # ------------------------------------------------------------------ #
        # Apply Classifier-Free Guidance
        # ------------------------------------------------------------------ #
        # CFG formula: logits_cfg = logits_uncond + scale * (logits_cond - logits_uncond)
        # cfg_scale = 4.5 from config.nfig.cfg_scale (paper Section 4.1)
        logits_cfg: Tensor = self._apply_cfg(
            logits_cond=logits_cond,
            logits_uncond=logits_uncond,
            scale=cfg_scale,
        )  # [B, band_size, K]

        # ------------------------------------------------------------------ #
        # Apply top-k filtering and sample
        # ------------------------------------------------------------------ #
        # Reshape to [B * band_size, K] for efficient per-position sampling.
        logits_flat: Tensor = logits_cfg.reshape(B * band_size, K)  # [B*band_size, K]

        # Apply top-k filtering and multinomial sampling.
        # Returns sampled indices of shape [B * band_size].
        sampled_flat: Tensor = self._top_k_sample(
            logits=logits_flat,
            k=top_k,
        )  # [B * band_size]

        # Reshape back to [B, band_size].
        new_tokens: Tensor = sampled_flat.reshape(B, band_size)  # [B, band_size]

        return new_tokens

    def _apply_cfg(
        self,
        logits_cond: Tensor,
        logits_uncond: Tensor,
        scale: float,
    ) -> Tensor:
        """Combine conditional and unconditional logits using CFG formula.

        Implements the standard Classifier-Free Guidance combination:
            logits_cfg = logits_uncond + scale * (logits_cond - logits_uncond)

        This is equivalent to:
            logits_cfg = (1 - scale) * logits_uncond + scale * logits_cond

        Both forms are mathematically identical. The first form is used here
        as it makes the guidance direction explicit.

        At scale=0.0: logits_cfg = logits_uncond (unconditional generation)
        At scale=1.0: logits_cfg = logits_cond (conditional, no guidance boost)
        At scale=4.5: strong class-conditional guidance (paper's optimal value)

        No softmax is applied here — top-k filtering operates on raw logits.
        The CFG combination is applied in logit space (before softmax), which
        is the standard approach for discrete token generation.

        Args:
            logits_cond: Conditional logits of shape (B, n_tokens, K).
                From transformer forward pass with real class labels.
            logits_uncond: Unconditional logits of shape (B, n_tokens, K).
                From transformer forward pass with null class labels (id=1000).
            scale: CFG scale factor. From config.nfig.cfg_scale = 4.5.
                Must be non-negative (validated in sample()).

        Returns:
            CFG-combined logits of shape (B, n_tokens, K).
            Same shape as inputs. Raw (unnormalized) logits.
        """
        # Standard CFG formula: logits_cfg = logits_uncond + scale * (logits_cond - logits_uncond)
        # Operates element-wise on the full logit tensor.
        logits_cfg: Tensor = logits_uncond + scale * (logits_cond - logits_uncond)
        return logits_cfg

    def _top_k_sample(
        self,
        logits: Tensor,
        k: int,
    ) -> Tensor:
        """Apply top-k filtering to logits and sample one token per position.

        Implements the top-k sampling strategy from paper Section 4.1
        (config.nfig.top_k = 990). Only the top-k logits are kept; all others
        are set to -inf before softmax, ensuring they have zero probability.

        Sampling procedure:
            1. Find the k-th largest logit per row (threshold).
            2. Mask all logits below threshold to -inf.
            3. Apply softmax to get probabilities.
            4. Sample one token per position via multinomial sampling.

        Args:
            logits: Raw logit tensor of shape (N, K) where:
                - N = B * band_size (batch × tokens per band)
                - K = codebook_size = 4096
                Values are CFG-combined logits (may have large magnitudes).
            k: Number of top logits to keep. From config.nfig.top_k = 990.
                Must be in [1, K]. With k=990 and K=4096, ~24% of logits are kept.

        Returns:
            Sampled token indices of shape (N,), dtype torch.long.
            Values in [0, K-1] = [0, 4095].

        Note:
            The -inf masking before softmax handles numerical stability correctly:
            softmax(-inf) = 0, so masked tokens have exactly zero probability.
            After CFG, logits may have large magnitudes, but the relative ordering
            (and thus top-k selection) is preserved.
        """
        N: int = logits.shape[0]
        K: int = logits.shape[1]

        # Clamp k to valid range [1, K] for safety.
        k_clamped: int = max(1, min(k, K))

        # ------------------------------------------------------------------ #
        # Step 1: Find the k-th largest logit value per row (threshold)
        # ------------------------------------------------------------------ #
        # torch.topk returns the top-k values and their indices.
        # We only need the values to determine the threshold.
        # top_k_values: shape [N, k_clamped]
        # The k-th largest value is at index k_clamped - 1 (0-indexed).
        top_k_values: Tensor = torch.topk(
            logits, k=k_clamped, dim=-1, largest=True, sorted=True
        ).values  # [N, k_clamped]

        # Threshold: the k-th largest value per row.
        # Shape: [N, 1] for broadcasting against [N, K].
        threshold: Tensor = top_k_values[:, -1:].contiguous()  # [N, 1]

        # ------------------------------------------------------------------ #
        # Step 2: Mask all logits below threshold to -inf
        # ------------------------------------------------------------------ #
        # logits < threshold: True for logits that should be masked out.
        # masked_fill: replaces True positions with -inf.
        # After masking, only the top-k logits have finite values.
        filtered_logits: Tensor = logits.masked_fill(
            logits < threshold,
            float("-inf"),
        )  # [N, K]

        # ------------------------------------------------------------------ #
        # Step 3: Convert to probabilities via softmax
        # ------------------------------------------------------------------ #
        # softmax(-inf) = 0 for masked positions (numerically exact).
        # softmax is applied over the vocabulary dimension (dim=-1).
        probs: Tensor = F.softmax(filtered_logits, dim=-1)  # [N, K]

        # ------------------------------------------------------------------ #
        # Step 4: Sample one token per position via multinomial sampling
        # ------------------------------------------------------------------ #
        # torch.multinomial samples one index per row according to probs.
        # num_samples=1: one token per position.
        # squeeze(-1): remove the trailing singleton dimension.
        # Shape: [N, 1] → [N]
        sampled: Tensor = torch.multinomial(
            probs,
            num_samples=1,
            replacement=True,
        ).squeeze(-1)  # [N]

        return sampled

    def extra_repr(self) -> str:
        """Return a human-readable string with key sampler configuration.

        Returns:
            String describing the sampler's key inference parameters.
        """
        total_tokens: int = sum(self.band_sizes)
        return (
            f"num_bands={self.num_bands}, "
            f"scale_factors={self.scale_factors}, "
            f"band_sizes={self.band_sizes}, "
            f"total_tokens={total_tokens}, "
            f"cfg_scale={self.config.cfg_scale}, "
            f"top_k={self.config.top_k}, "
            f"codebook_size={self.config.codebook_size}, "
            f"null_class_id={self.config.null_class_id}, "
            f"device={self.device}"
        )

    def __repr__(self) -> str:
        """Return a string representation of the sampler.

        Returns:
            String with class name and key configuration.
        """
        return f"NFIGSampler({self.extra_repr()})"

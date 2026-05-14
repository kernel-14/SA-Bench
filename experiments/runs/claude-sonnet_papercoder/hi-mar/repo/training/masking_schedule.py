## training/masking_schedule.py
"""Masking schedule utilities for Hi-MAR training and inference.

This module provides a stateless utility class ``MaskingSchedule`` with all
methods implemented as ``@staticmethod``. It covers every masking scenario
described in the paper:

- Phase 1 training (ImageNet): uniform ratio sampling in [0.7, 1.0]
- Phase 2 training (ImageNet/COCO): cosine ratio sampling following MaskGIT
- Phase 1 training (COCO): Beta(4, 1) ratio sampling following AutoNAT-L
- Inference (both phases): deterministic cosine progressive unmasking schedule

Masking convention (uniform throughout the codebase):
    True  = masked   (token is hidden / to be predicted)
    False = unmasked (token is visible / already known)

No project-internal imports. This module sits at the bottom of the dependency
graph and is safe to import from any other module without circular-import risk.
"""

import math
from typing import Optional

import torch
import torch.distributions as dist


class MaskingSchedule:
    """Stateless utility class for all masking operations in Hi-MAR.

    All methods are ``@staticmethod``; the class has no ``__init__`` and no
    instance state. Callers may use the class directly without instantiation::

        ratio = MaskingSchedule.sample_uniform_ratio(batch_size=8)
        mask  = MaskingSchedule.tokens_to_mask(n_tokens=64, ratio=ratio)

    Masking convention: ``True`` means the token is **masked** (hidden).
    """

    @staticmethod
    def sample_uniform_ratio(
        batch_size: int,
        r_min: float = 0.7,
        r_max: float = 1.0,
    ) -> torch.Tensor:
        """Samples per-sample masking ratios uniformly from [r_min, r_max].

        Used for Phase 1 training on ImageNet class-conditional generation.

        Paper reference (Section 4.2):
            "In the first phase, the masking ratio is randomly sampled in
            [0.7, 1.0] as MAR."

        Config reference:
            training_imagenet.masking.phase1.strategy = uniform
            training_imagenet.masking.phase1.ratio_min = 0.7
            training_imagenet.masking.phase1.ratio_max = 1.0

        Args:
            batch_size: Number of independent ratio samples to draw (one per
                sample in the batch).
            r_min: Lower bound of the uniform distribution. Defaults to 0.7
                per the paper's specification.
            r_max: Upper bound of the uniform distribution. Defaults to 1.0.

        Returns:
            Float tensor of shape ``[batch_size]`` with values in
            ``[r_min, r_max]``. The tensor is on CPU; callers move it to the
            target device as needed.
        """
        # torch.rand produces values in [0, 1); scale and shift to [r_min, r_max).
        ratios: torch.Tensor = torch.rand(batch_size) * (r_max - r_min) + r_min
        return ratios

    @staticmethod
    def sample_cosine_ratio(batch_size: int) -> torch.Tensor:
        """Samples per-sample masking ratios via the MaskGIT cosine schedule.

        Used for Phase 2 training on both ImageNet and MS-COCO.

        The cosine mapping ``r = cos(π/2 · u)`` with ``u ~ Uniform(0, 1)``
        produces ratios concentrated near 1.0 (high masking), which matches
        the inference-time cosine unmasking schedule and ensures the model
        trains on a distribution consistent with how it will be evaluated.

        Mapping behaviour:
            u = 0.0  →  r = cos(0)       = 1.0  (fully masked)
            u = 0.5  →  r = cos(π/4)    ≈ 0.71
            u = 1.0  →  r = cos(π/2)    = 0.0  (fully unmasked)

        Paper reference (Section 4.2):
            "the second phase uses the cosine masking strategy following
            MaskGIT."

        Config reference:
            training_imagenet.masking.phase2.strategy = cosine
            training_coco.masking.phase2.strategy = cosine

        Args:
            batch_size: Number of independent ratio samples to draw.

        Returns:
            Float tensor of shape ``[batch_size]`` with values in ``[0, 1]``.
        """
        u: torch.Tensor = torch.rand(batch_size)
        ratios: torch.Tensor = torch.cos(math.pi / 2.0 * u)
        return ratios

    @staticmethod
    def sample_beta_ratio(
        batch_size: int,
        alpha: float = 4.0,
        beta: float = 1.0,
    ) -> torch.Tensor:
        """Samples per-sample masking ratios from a Beta distribution.

        Used for Phase 1 training on MS-COCO text-to-image generation,
        following the AutoNAT-L protocol.

        ``Beta(α=4, β=1)`` has mean ``α/(α+β) = 0.8``, so on average 80 % of
        tokens are masked during COCO training. The distribution is skewed
        toward 1.0, biasing training toward harder prediction tasks where the
        model must generate most content from scratch given only text
        conditioning.

        Paper reference (Section 4.2):
            "we follow AutoNAT-L and randomly sample the masking ratio by Beta
            distribution (α=4, β=1)."

        Config reference:
            training_coco.masking.phase1.strategy = beta
            training_coco.masking.phase1.beta_alpha = 4.0
            training_coco.masking.phase1.beta_beta  = 1.0

        Args:
            batch_size: Number of independent ratio samples to draw.
            alpha: First shape parameter of the Beta distribution. Defaults to
                4.0 per the paper.
            beta: Second shape parameter of the Beta distribution. Defaults to
                1.0 per the paper.

        Returns:
            Float tensor of shape ``[batch_size]`` with values in ``[0, 1]``.
        """
        beta_dist = dist.Beta(
            torch.tensor(alpha, dtype=torch.float32),
            torch.tensor(beta, dtype=torch.float32),
        )
        ratios: torch.Tensor = beta_dist.sample((batch_size,))
        return ratios

    @staticmethod
    def tokens_to_mask(
        n_tokens: int,
        ratio: torch.Tensor,
    ) -> torch.Tensor:
        """Converts per-sample float ratios into boolean mask tensors.

        For each sample ``b`` in the batch, randomly selects
        ``⌈ratio[b] · n_tokens⌉`` token positions to mask (set to ``True``).
        The selection is uniformly random without replacement, implemented via
        a vectorised argsort trick that avoids Python-level loops.

        Paper reference (Section 3.1):
            "MAR randomly selects ⌈r·N⌉ visual tokens and replaces them with
            masked tokens, where r denotes the masking ratio."

        Masking convention: ``True = masked``.

        Vectorised implementation:
            1. Draw ``rand_scores`` of shape ``[B, n_tokens]`` from
               ``Uniform(0, 1)``.
            2. Compute ranks via ``argsort`` (ascending). Rank 0 is the
               position with the smallest random score.
            3. A position is masked iff its rank is strictly less than
               ``n_mask_per_sample[b]``. Because the scores are i.i.d.
               uniform, this selects exactly ``n_mask`` positions uniformly
               at random.

        Args:
            n_tokens: Total number of tokens in the sequence (e.g. 64 for
                low-resolution, 256 for high-resolution).
            ratio: Float tensor of shape ``[B]`` with per-sample masking
                ratios in ``[0, 1]``. Produced by one of the
                ``sample_*_ratio`` methods above.

        Returns:
            Boolean tensor of shape ``[B, n_tokens]`` on the same device as
            ``ratio``. ``True`` indicates a masked (hidden) position.

        Edge cases:
            - ``ratio = 0.0`` → ``n_mask = 0`` → all ``False`` (nothing masked).
            - ``ratio = 1.0`` → ``n_mask = n_tokens`` → all ``True``.
        """
        batch_size: int = ratio.shape[0]
        device: torch.device = ratio.device

        # Number of tokens to mask per sample: ⌈ratio * n_tokens⌉, clamped to
        # [0, n_tokens] to guard against floating-point edge cases.
        n_mask_per_sample: torch.Tensor = torch.ceil(
            ratio.float() * n_tokens
        ).long().clamp(0, n_tokens)  # shape: [B]

        # Random scores for each (sample, token) pair.
        rand_scores: torch.Tensor = torch.rand(
            batch_size, n_tokens, device=device
        )  # shape: [B, n_tokens]

        # Rank each token within its sample (ascending: rank 0 = smallest score).
        # Positions with rank < n_mask_per_sample[b] are selected for masking.
        ranks: torch.Tensor = rand_scores.argsort(dim=1)  # shape: [B, n_tokens]

        # Broadcast n_mask_per_sample for comparison: [B, 1] vs [B, n_tokens].
        mask: torch.Tensor = ranks < n_mask_per_sample.unsqueeze(1)  # BoolTensor [B, n_tokens]
        return mask

    @staticmethod
    def get_cosine_schedule_mask(
        n_tokens: int,
        step: int,
        total_steps: int,
    ) -> torch.Tensor:
        """Returns the number of tokens that should remain masked after a given step.

        Implements the cosine progressive unmasking schedule used at inference
        time. At each autoregressive step ``s`` (0-indexed), the fraction of
        tokens that should **still be masked** after that step is:

            r(s) = cos(π/2 · (s + 1) / total_steps)

        Boundary behaviour:
            s = 0            →  r ≈ cos(π/2 · 1/T)  (most tokens still masked)
            s = total_steps-1 →  r = cos(π/2)  = 0.0 (all tokens revealed)

        The returned boolean tensor has exactly ``n_still_masked`` positions
        set to ``True``. The specific positions are chosen randomly so that
        the ``Generator`` can intersect this with its current mask state to
        determine which tokens to newly predict at each step.

        Paper reference (Section 4.2 and 4.5):
            "we use 32 and 4 steps for the first and second phases with a
            cosine schedule."

        Config reference:
            inference.schedule = cosine
            inference.phase1_steps = 32
            inference.phase2_steps = 4

        Args:
            n_tokens: Total number of tokens in the sequence (64 or 256).
            step: Current autoregressive step index, 0-indexed. Must satisfy
                ``0 <= step < total_steps``.
            total_steps: Total number of autoregressive steps for this phase
                (32 for Phase 1, 4 for Phase 2 per the paper).

        Returns:
            Boolean tensor of shape ``[n_tokens]`` on CPU. ``True`` indicates
            a position that should **remain masked** after this step. The
            tensor contains exactly ``n_still_masked`` ``True`` values, where
            ``n_still_masked = floor(cos(π/2 · (step+1)/total_steps) * n_tokens)``.
        """
        # Fraction of tokens that should remain masked after this step.
        progress: float = (step + 1) / float(total_steps)
        ratio_after: float = math.cos(math.pi / 2.0 * progress)

        # Number of tokens to keep masked (floor to ensure we always reveal at
        # least one token per step when total_steps > 1).
        n_still_masked: int = math.floor(ratio_after * n_tokens)
        n_still_masked = max(0, min(n_still_masked, n_tokens))

        # Build a random boolean mask with exactly n_still_masked True values.
        # Using a random permutation ensures no positional bias.
        perm: torch.Tensor = torch.randperm(n_tokens)  # shape: [n_tokens]
        mask: torch.Tensor = torch.zeros(n_tokens, dtype=torch.bool)
        if n_still_masked > 0:
            mask[perm[:n_still_masked]] = True

        return mask

    @staticmethod
    def apply_mask(
        tokens: torch.Tensor,
        mask: torch.Tensor,
        mask_token: torch.Tensor,
    ) -> torch.Tensor:
        """Replaces masked token positions with the learnable mask token embedding.

        This is the operation described in Section 3.1 of the paper where
        masked positions in the input sequence are replaced with a shared
        learnable ``[MASK]`` embedding before being fed into the Transformer.

        The method is non-destructive: it returns a new tensor and never
        modifies ``tokens`` in-place. The original ``tokens`` tensor must be
        preserved by the caller for loss computation (comparing predictions
        against ground-truth token values at masked positions).

        Paper reference (Section 3.1):
            "The masked sequence X' = {x'_1, x'_2, ..., x'_N} is fed into
            the masked autoregressive Transformer."

        Masking convention: ``mask[b, i] = True`` means position ``i`` in
        sample ``b`` is masked and will be replaced by ``mask_token``.

        Args:
            tokens: Float tensor of shape ``[B, N, D]`` containing the
                original (unmasked) token embeddings. ``D`` is the latent
                dimension (16 for KL-16 VAE tokens) or the Transformer hidden
                size, depending on where in the pipeline this is called.
            mask: Boolean tensor of shape ``[B, N]``. ``True = masked``.
                Must be on the same device as ``tokens``.
            mask_token: Learnable mask embedding of shape ``[1, 1, D]`` or
                ``[D]``. This is ``ScaleAwareTransformer.mask_token``, an
                ``nn.Parameter``. Must be on the same device as ``tokens``.

        Returns:
            Float tensor of shape ``[B, N, D]`` on the same device as
            ``tokens``, with masked positions replaced by ``mask_token``.
        """
        # Ensure mask_token has shape [1, 1, D] for broadcasting.
        mt: torch.Tensor = mask_token
        if mt.dim() == 1:
            # Shape [D] → [1, 1, D]
            mt = mt.unsqueeze(0).unsqueeze(0)
        elif mt.dim() == 2:
            # Shape [1, D] → [1, 1, D]
            mt = mt.unsqueeze(0)
        # mt is now [1, 1, D]; expand to [B, N, D] for torch.where.
        mt_expanded: torch.Tensor = mt.expand_as(tokens)

        # mask has shape [B, N]; unsqueeze last dim to [B, N, 1] for broadcasting
        # across the D dimension.
        mask_expanded: torch.Tensor = mask.unsqueeze(-1)  # [B, N, 1]

        # torch.where selects mt_expanded where mask is True, tokens otherwise.
        # This is equivalent to:
        #   output = tokens.clone()
        #   output[mask] = mt_expanded[mask]
        # but avoids the clone + advanced indexing overhead.
        output: torch.Tensor = torch.where(mask_expanded, mt_expanded, tokens)
        return output

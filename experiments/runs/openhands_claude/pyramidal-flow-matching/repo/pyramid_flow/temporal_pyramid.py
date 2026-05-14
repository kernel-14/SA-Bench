"""Temporal pyramid for autoregressive video generation.

Implements the temporal pyramid design from the paper:
- History condition with progressively compressed lower-resolution frames
- Noise corruption of history during training to mitigate error accumulation
- Clean generated frames used as history during inference

Key design (Eq. 16-17):
    Training: ... -> Down(x_{t'}^{i-2}, 2^{k+1}) -> Down(x_{t'}^{i-1}, 2^k) -> x_hat_t^i
    Inference: ... -> Down(x_1^{i-2}, 2^{k+1}) -> Down(x_1^{i-1}, 2^k) -> x_hat_t^i

where t' indicates small noise added to history latents during training.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from pyramid_flow.spatial_pyramid import downsample_latent, upsample_latent


class TemporalPyramid:
    """Temporal pyramid for efficient autoregressive video generation.

    Compresses history frames at progressively lower resolutions:
    - Most distant frames: lowest resolution (1/2^K)
    - Most recent frames: higher resolution
    - Current generation: full resolution

    This reduces the number of history tokens by up to 1/4^K times.
    """

    def __init__(
        self,
        num_stages: int = 3,
        history_noise_max: float = 1 / 3,
        downsample_mode: str = "bilinear",
        upsample_mode: str = "nearest",
    ):
        self.num_stages = num_stages
        self.history_noise_max = history_noise_max
        self.downsample_mode = downsample_mode
        self.upsample_mode = upsample_mode

    def build_history_condition(
        self,
        history_latents: List[torch.Tensor],
        current_stage: int,
        training: bool = True,
    ) -> Tuple[torch.Tensor, List[int]]:
        """Build the temporal pyramid history condition.

        For a current generation at pyramid stage k, the history is:
        [..., Down(x^{i-2}, 2^{k+1}), Down(x^{i-1}, 2^k)]

        More distant frames are compressed more aggressively.

        Args:
            history_latents: list of (B, C, H, W) latent tensors, ordered oldest to newest
            current_stage: current pyramid stage (0=lowest, K-1=full res)
            training: if True, add corruption noise to history

        Returns:
            history_tokens: concatenated history at appropriate resolutions
            frame_indices: original frame indices for position encoding
        """
        if not history_latents:
            return None, []

        compressed_history = []
        frame_indices = []

        n_hist = len(history_latents)
        for i, latent in enumerate(history_latents):
            # Determine compression level for this history frame
            # Most recent frame: compressed by 2^current_stage
            # Older frames: compressed more aggressively
            age = n_hist - 1 - i  # 0 = most recent, n_hist-1 = oldest
            compression_level = current_stage + age
            compression_level = min(compression_level, self.num_stages + 1)
            factor = 2 ** compression_level

            # Downsample history frame
            compressed = downsample_latent(latent, factor, self.downsample_mode)

            # Add corruption noise during training (Eq. 16)
            if training:
                noise_strength = torch.rand(1).item() * self.history_noise_max
                noise = torch.randn_like(compressed)
                compressed = compressed + noise_strength * noise

            compressed_history.append(compressed)
            frame_indices.append(i)

        return compressed_history, frame_indices

    def build_pyramid_history_sequence(
        self,
        history_latents: List[torch.Tensor],
        current_stage: int,
        training: bool = True,
    ) -> Tuple[Optional[torch.Tensor], List[int]]:
        """Build the full temporal pyramid history as a flat sequence.

        Implements the temporal pyramid from Fig. 3a of the paper.
        At each pyramid stage k, the history is compressed:
        - Stage k=0 (lowest res): all history at 1/2^K resolution
        - Stage k=K-1 (full res): most recent at 1/2, older at 1/4, etc.

        Args:
            history_latents: list of (B, C, H, W) full-resolution latents
            current_stage: current generation stage
            training: whether to add corruption noise

        Returns:
            history_seq: (B, N_hist_tokens, C, H_k, W_k) or None
            frame_indices: list of frame indices for temporal RoPE
        """
        if not history_latents:
            return None, []

        compressed_frames = []
        frame_indices = []

        n_hist = len(history_latents)
        for i, latent in enumerate(history_latents):
            age = n_hist - 1 - i
            # Compression: most recent frame at 2^current_stage, older at higher powers
            compression = current_stage + age
            compression = min(compression, self.num_stages + 1)
            factor = 2 ** compression

            compressed = downsample_latent(latent, factor, self.downsample_mode)

            if training and self.history_noise_max > 0:
                noise_strength = torch.rand(1, device=latent.device).item() * self.history_noise_max
                compressed = compressed + noise_strength * torch.randn_like(compressed)

            compressed_frames.append(compressed)
            frame_indices.append(i)

        return compressed_frames, frame_indices

    def get_history_token_count(
        self,
        n_history_frames: int,
        base_tokens_per_frame: int,
        current_stage: int,
    ) -> int:
        """Compute total number of history tokens.

        Args:
            n_history_frames: number of history frames
            base_tokens_per_frame: tokens per frame at full resolution
            current_stage: current pyramid stage

        Returns:
            total history token count
        """
        total = 0
        for age in range(n_history_frames):
            compression = current_stage + age
            compression = min(compression, self.num_stages + 1)
            factor = 2 ** compression
            tokens = base_tokens_per_frame // (factor * factor)
            total += tokens
        return total

    def compute_efficiency_gain(
        self,
        n_history_frames: int,
        base_tokens_per_frame: int,
        current_stage: int,
    ) -> float:
        """Compute token reduction ratio vs full-resolution history."""
        full_res_tokens = n_history_frames * base_tokens_per_frame
        pyramid_tokens = self.get_history_token_count(
            n_history_frames, base_tokens_per_frame, current_stage
        )
        return full_res_tokens / max(pyramid_tokens, 1)


def pack_variable_length_sequences(
    sequences: List[torch.Tensor],
    max_tokens: int,
) -> Tuple[torch.Tensor, List[int]]:
    """Pack variable-length sequences into a fixed-length batch.

    Implements Patch n' Pack (Dehghani et al., 2023) for length-balanced batching.
    Sequences are packed together to form batches with approximately equal token counts.

    Args:
        sequences: list of (N_i, D) token sequences of varying lengths
        max_tokens: maximum tokens per packed sequence

    Returns:
        packed: (B, max_tokens, D) packed sequences with padding
        lengths: list of actual sequence lengths
    """
    packed = []
    lengths = []
    current_pack = []
    current_len = 0

    for seq in sequences:
        n = seq.shape[0]
        if current_len + n > max_tokens and current_pack:
            # Finalize current pack
            packed_seq = torch.cat(current_pack, dim=0)
            packed.append(packed_seq)
            lengths.append(current_len)
            current_pack = []
            current_len = 0

        current_pack.append(seq)
        current_len += n

    if current_pack:
        packed_seq = torch.cat(current_pack, dim=0)
        packed.append(packed_seq)
        lengths.append(current_len)

    return packed, lengths

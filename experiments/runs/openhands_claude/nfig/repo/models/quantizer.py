from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.frequency import upsample_feature


class VectorQuantizer(nn.Module):
    """
    Vector quantizer with a shared learnable codebook Z ∈ R^{K×C}.

    Implements Eq. 4 from the paper: nearest-neighbor lookup in codebook.
    Uses straight-through estimator for gradients.
    """

    def __init__(self, codebook_size: int, codebook_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment_cost = commitment_cost

        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            v: (B, C, h, w) continuous feature map

        Returns:
            v_q: (B, C, h, w) quantized feature map (straight-through)
            indices: (B, h*w) codebook indices
            loss: scalar VQ loss
        """
        B, C, h, w = v.shape
        # Flatten spatial dims: (B*h*w, C)
        flat = v.permute(0, 2, 3, 1).reshape(-1, C)

        # Compute distances to all codebook entries
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z·e
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            + self.codebook.weight.pow(2).sum(1)
            - 2 * flat @ self.codebook.weight.t()
        )  # (B*h*w, K)

        indices = dist.argmin(dim=1)  # (B*h*w,)
        z_q = self.codebook(indices)  # (B*h*w, C)
        z_q = z_q.view(B, h, w, C).permute(0, 3, 1, 2)  # (B, C, h, w)

        # VQ loss: codebook loss + commitment loss
        codebook_loss = F.mse_loss(z_q, v.detach())
        commitment_loss = F.mse_loss(v, z_q.detach())
        loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator
        z_q_st = v + (z_q - v).detach()

        indices = indices.view(B, h * w)
        return z_q_st, indices, loss

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices: (B, h*w) or (B, h, w) integer indices

        Returns:
            (B, C, h, w) or (B, h, w, C) quantized vectors
        """
        flat = indices.reshape(-1)
        z = self.codebook(flat)
        if indices.dim() == 2:
            B, L = indices.shape
            side = int(L ** 0.5)
            return z.view(B, side, side, self.codebook_dim).permute(0, 3, 1, 2)
        else:
            B, h, w = indices.shape
            return z.view(B, h, w, self.codebook_dim).permute(0, 3, 1, 2)


class FrequencyResidualQuantizer(nn.Module):
    """
    Frequency-guided Residual Quantizer (Section 3.1.2).

    For each frequency level i:
    - Downsample the i-th frequency component f_hat_i to (h_i, w_i)
    - Compute residual: R_i = R_{i-1} + (f_hat_i - upsample(v_i))
    - Quantize: v_i = argmin ||target_i - upsample(v_i)||^2

    where target_0 = f_hat_0, target_i = R_{i-1} + f_hat_i for i >= 1.
    """

    def __init__(
        self,
        codebook_size: int,
        codebook_dim: int,
        scale_factors: List[int],
        feature_map_size: int,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.scale_factors = scale_factors
        self.feature_map_size = feature_map_size
        self.n = len(scale_factors)

        # Shared codebook across all frequency levels
        self.quantizer = VectorQuantizer(codebook_size, codebook_dim, commitment_cost)

    def forward(
        self, components: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """
        Frequency-guided residual quantization (Eq. 3 in paper).

        For each level i:
          target_i = f_hat_i              (i == 0)
          target_i = R_{i-1} + f_hat_i   (i >= 1)

          v_i = argmin ||target_i - T(v_i, H', W')||^2
              ≈ Q(downsample(target_i, h_i, w_i))

          R_i = target_i - T(v_q_i, H', W')

        Args:
            components: list of n frequency components, each (B, C, H', W')

        Returns:
            v_q_list: list of quantized feature maps at each scale (B, C, h_i, w_i)
            indices_list: list of token index tensors (B, h_i*w_i)
            residuals: list of residual tensors (B, C, H', W')
            total_vq_loss: scalar
        """
        H = self.feature_map_size
        W = self.feature_map_size

        v_q_list = []
        indices_list = []
        residuals = []
        total_vq_loss = torch.tensor(0.0, device=components[0].device)

        R_prev = None  # accumulated residual at full resolution

        for i, (s, f_hat_i) in enumerate(zip(self.scale_factors, components)):
            h_i, w_i = s, s

            # Compute full-resolution target for this level
            if i == 0:
                target = f_hat_i  # (B, C, H', W')
            else:
                target = R_prev + f_hat_i  # (B, C, H', W')

            # Downsample target to scale resolution for quantization
            # This approximates argmin_v ||target - T(v, H', W')||^2
            target_down = F.interpolate(
                target, size=(h_i, w_i), mode="bilinear", align_corners=False
            )

            # Quantize downsampled target
            v_q_i, idx_i, vq_loss_i = self.quantizer(target_down)
            total_vq_loss = total_vq_loss + vq_loss_i

            # Upsample quantized back to full resolution for residual computation
            v_q_i_up = upsample_feature(v_q_i, H, W)

            # Residual: R_i = target_i - T(v_q_i, H', W')
            R_i = target - v_q_i_up

            v_q_list.append(v_q_i)
            indices_list.append(idx_i)
            residuals.append(R_i)
            R_prev = R_i

        return v_q_list, indices_list, residuals, total_vq_loss

    def decode_indices(self, indices_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Reconstruct feature map from token indices by summing upsampled quantized vectors.

        Args:
            indices_list: list of (B, h_i*w_i) index tensors

        Returns:
            (B, C, H', W') reconstructed feature map
        """
        H = self.feature_map_size
        W = self.feature_map_size
        result = None

        for i, (s, idx) in enumerate(zip(self.scale_factors, indices_list)):
            h_i, w_i = s, s
            B = idx.shape[0]
            # Reshape indices to (B, h_i, w_i) for lookup
            idx_2d = idx.view(B, h_i, w_i)
            z_q = self.quantizer.lookup(idx_2d)  # (B, C, h_i, w_i)
            z_q_up = upsample_feature(z_q, H, W)
            result = z_q_up if result is None else result + z_q_up

        return result

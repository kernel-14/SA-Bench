from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def pseudo_huber_distance(x: torch.Tensor, y: torch.Tensor, c: float = 0.00054) -> torch.Tensor:
    """
    Pseudo-Huber distance: D(x, y) = sqrt(||x - y||² + c²) - c

    Used as the distance function D in consistency losses.
    The constant c is set to 0.00054 * sqrt(d) where d is the data dimension,
    following Song and Dhariwal (2024).
    """
    diff = (x - y).view(x.shape[0], -1)
    return torch.sqrt((diff ** 2).sum(dim=-1) + c ** 2) - c


def lpips_distance(x: torch.Tensor, y: torch.Tensor, lpips_fn: nn.Module) -> torch.Tensor:
    """LPIPS perceptual distance."""
    return lpips_fn(x, y).squeeze()


class ConsistencyTrainingLoss(nn.Module):
    """
    Consistency Training (CT) loss with Independent Coupling (IC):

        L_CT(θ) = E_{q_I(x_*, z), i} [λ(σ_{t_i}) * D(sg(f_θ(x_{t_i}, σ_{t_i})),
                                                          f_θ(x_{t_{i+1}}, σ_{t_{i+1}}))]

    where x_{t_i} = x_* + σ_{t_i} * z and x_{t_{i+1}} = x_* + σ_{t_{i+1}} * z.
    """

    def __init__(self, distance_fn: Optional[Callable] = None):
        super().__init__()
        self.distance_fn = distance_fn or pseudo_huber_distance

    def forward(
        self,
        model: nn.Module,
        x_lower: torch.Tensor,
        x_upper: torch.Tensor,
        sigma_lower: torch.Tensor,
        sigma_upper: torch.Tensor,
        loss_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the CT loss.

        Args:
            model: consistency model f_θ
            x_lower: noisy samples at lower noise level (B, C, H, W)
            x_upper: noisy samples at upper noise level (B, C, H, W)
            sigma_lower: lower noise levels (B,)
            sigma_upper: upper noise levels (B,)
            loss_weights: per-sample loss weights λ(σ_{t_i}) (B,)

        Returns:
            Scalar loss value
        """
        # Stop-gradient on the lower noise level prediction
        with torch.no_grad():
            f_lower = model(x_lower, sigma_lower).detach()

        f_upper = model(x_upper, sigma_upper)

        dist = self.distance_fn(f_lower, f_upper)
        loss = (loss_weights * dist).mean()
        return loss


class ConsistencyDistillationLoss(nn.Module):
    """
    Consistency Distillation (CD) loss:

        L_CD(θ) = E_{q_I(x_*, z), i} [λ(σ_{t_i}) * D(sg(f_θ(x_{t_i}^Φ, σ_{t_i})),
                                                          f_θ(x_{t_{i+1}}, σ_{t_{i+1}}))]

    where x_{t_i}^Φ is computed by one Euler step of the PF-ODE using a pre-trained
    score model:
        x_{t_i}^Φ = x_{t_{i+1}} + (t_i - t_{i+1}) * v_{t_{i+1}}(x_{t_{i+1}})
    """

    def __init__(
        self,
        score_model: nn.Module,
        distance_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.score_model = score_model
        self.distance_fn = distance_fn or pseudo_huber_distance

    def euler_step(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
    ) -> torch.Tensor:
        """
        One Euler step of the PF-ODE using the score model.

        v_t(x) = -t * ∇_x log p_t(x) = (x - D(x, t)) / t  (EDM formulation)

        x_{t_i}^Φ = x_{t_{i+1}} + (σ_i - σ_{i+1}) * v_{t_{i+1}}(x_{t_{i+1}})
        """
        with torch.no_grad():
            # Score model predicts denoised x_0
            x_denoised = self.score_model(x, sigma)
            # Velocity field: v_t(x) = (x - x_denoised) / sigma
            sigma_bc = sigma[:, None, None, None]
            v = (x - x_denoised) / sigma_bc
            # Euler step
            dt = (sigma_next - sigma)[:, None, None, None]
            x_next = x + dt * v
        return x_next

    def forward(
        self,
        model: nn.Module,
        x_upper: torch.Tensor,
        sigma_lower: torch.Tensor,
        sigma_upper: torch.Tensor,
        loss_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the CD loss.

        Args:
            model: consistency model f_θ
            x_upper: noisy samples at upper noise level (B, C, H, W)
            sigma_lower: lower noise levels (B,)
            sigma_upper: upper noise levels (B,)
            loss_weights: per-sample loss weights λ(σ_{t_i}) (B,)

        Returns:
            Scalar loss value
        """
        # Compute x_{t_i}^Φ via Euler step
        x_lower_phi = self.euler_step(x_upper, sigma_upper, sigma_lower)

        with torch.no_grad():
            f_lower = model(x_lower_phi, sigma_lower).detach()

        f_upper = model(x_upper, sigma_upper)

        dist = self.distance_fn(f_lower, f_upper)
        loss = (loss_weights * dist).mean()
        return loss


class GCLoss(nn.Module):
    """
    Generator-Augmented Consistency (GC) loss:

        L_GC(θ) = E_{q(x_hat, z), i} [λ(σ_{t_i}) * D(sg(f_θ(x_tilde_{t_i}, σ_{t_i})),
                                                          f_θ(x_tilde_{t_{i+1}}, σ_{t_{i+1}}))]

    where:
        x_hat_{t_i} = sg(f_θ(x_{t_i}, σ_{t_i}))  (endpoint prediction)
        x_tilde_{t_i} = x_hat_{t_i} + σ_{t_i} * z
        x_tilde_{t_{i+1}} = x_hat_{t_i} + σ_{t_{i+1}} * z
    """

    def __init__(self, distance_fn: Optional[Callable] = None):
        super().__init__()
        self.distance_fn = distance_fn or pseudo_huber_distance

    def forward(
        self,
        model: nn.Module,
        x_tilde_lower: torch.Tensor,
        x_tilde_upper: torch.Tensor,
        sigma_lower: torch.Tensor,
        sigma_upper: torch.Tensor,
        loss_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the GC loss.

        Args:
            model: consistency model f_θ
            x_tilde_lower: GC noisy samples at lower noise level (B, C, H, W)
            x_tilde_upper: GC noisy samples at upper noise level (B, C, H, W)
            sigma_lower: lower noise levels (B,)
            sigma_upper: upper noise levels (B,)
            loss_weights: per-sample loss weights λ(σ_{t_i}) (B,)

        Returns:
            Scalar loss value
        """
        with torch.no_grad():
            f_lower = model(x_tilde_lower, sigma_lower).detach()

        f_upper = model(x_tilde_upper, sigma_upper)

        dist = self.distance_fn(f_lower, f_upper)
        loss = (loss_weights * dist).mean()
        return loss


class JointGCLoss(nn.Module):
    """
    Joint learning loss combining IC and GC trajectories:

        L_{GC-μ}(θ) = μ * L_GC(θ) + (1-μ) * L_CT(θ)

    In practice, implemented via a binary mask m ~ Binomial(μ, batch_size)
    that selects which samples use GC vs IC trajectories (Algorithm 1).
    """

    def __init__(self, mu: float = 0.5, distance_fn: Optional[Callable] = None):
        super().__init__()
        self.mu = mu
        self.distance_fn = distance_fn or pseudo_huber_distance

    def forward(
        self,
        model: nn.Module,
        x_lower: torch.Tensor,
        x_upper: torch.Tensor,
        sigma_lower: torch.Tensor,
        sigma_upper: torch.Tensor,
        loss_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the joint GC-μ loss.

        The x_lower and x_upper are already mixed IC/GC pairs (constructed by JointCoupling).

        Args:
            model: consistency model f_θ
            x_lower: mixed IC/GC noisy samples at lower noise level (B, C, H, W)
            x_upper: mixed IC/GC noisy samples at upper noise level (B, C, H, W)
            sigma_lower: lower noise levels (B,)
            sigma_upper: upper noise levels (B,)
            loss_weights: per-sample loss weights λ(σ_{t_i}) (B,)

        Returns:
            Scalar loss value
        """
        with torch.no_grad():
            f_lower = model(x_lower, sigma_lower).detach()

        f_upper = model(x_upper, sigma_upper)

        dist = self.distance_fn(f_lower, f_upper)
        loss = (loss_weights * dist).mean()
        return loss


def get_pseudo_huber_c(img_channels: int, img_resolution: int) -> float:
    """
    Compute the pseudo-Huber constant c = 0.00054 * sqrt(d)
    where d = img_channels * img_resolution^2.
    """
    import math
    d = img_channels * img_resolution * img_resolution
    return 0.00054 * math.sqrt(d)

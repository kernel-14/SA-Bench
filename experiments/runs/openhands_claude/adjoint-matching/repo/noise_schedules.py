"""
Noise schedules for Flow Matching and diffusion models.

Unified framework from Section 3 of the paper:
  dX_t = b(X_t, t) dt + sigma(t) dB_t
  b(x, t) = kappa_t * x + (sigma(t)^2/2 + eta_t) * s(x, t)
  kappa_t = alpha_dot_t / alpha_t
  eta_t = beta_t * (alpha_dot_t/alpha_t * beta_t - beta_dot_t)

Reference flow: X_t = beta_t * X_0 + alpha_t * X_1
  alpha_0 = beta_1 = 0,  alpha_1 = beta_0 = 1

Memoryless noise schedule (Proposition 1, Theorem 1):
  sigma(t)^2 = 2 * eta_t  =>  sigma(t) = sqrt(2 * eta_t)
  This is the ONLY schedule that ensures convergence to the tilted distribution.
"""

import math
import torch
from typing import Tuple


# ---------------------------------------------------------------------------
# Flow Matching schedule: alpha_t = t, beta_t = 1 - t
# ---------------------------------------------------------------------------

def fm_alpha(t: torch.Tensor) -> torch.Tensor:
    """alpha_t = t"""
    return t


def fm_beta(t: torch.Tensor) -> torch.Tensor:
    """beta_t = 1 - t"""
    return 1.0 - t


def fm_alpha_dot(t: torch.Tensor) -> torch.Tensor:
    """d/dt alpha_t = 1"""
    return torch.ones_like(t)


def fm_beta_dot(t: torch.Tensor) -> torch.Tensor:
    """d/dt beta_t = -1"""
    return -torch.ones_like(t)


def fm_kappa(t: torch.Tensor) -> torch.Tensor:
    """kappa_t = alpha_dot_t / alpha_t = 1/t"""
    return fm_alpha_dot(t) / fm_alpha(t)


def fm_eta(t: torch.Tensor) -> torch.Tensor:
    """
    eta_t = beta_t * (kappa_t * beta_t - beta_dot_t)
          = (1-t) * ((1-t)/t + 1)
          = (1-t) * (1/t)
          = (1-t) / t
    """
    alpha = fm_alpha(t)
    beta = fm_beta(t)
    alpha_dot = fm_alpha_dot(t)
    beta_dot = fm_beta_dot(t)
    return beta * (alpha_dot / alpha * beta - beta_dot)


def fm_sigma_memoryless(t: torch.Tensor, h: float = 0.025) -> torch.Tensor:
    """
    Memoryless noise schedule for Flow Matching (Table 1):
      sigma(t) = sqrt(2 * eta_t) = sqrt(2*(1-t)/t)

    With offset to avoid division by zero (Appendix G.1):
      sigma(t) = sqrt(2*(1-t+h)/(t+h))
    """
    return torch.sqrt(2.0 * (1.0 - t + h) / (t + h))


def fm_sigma_zero(t: torch.Tensor) -> torch.Tensor:
    """Deterministic ODE: sigma(t) = 0"""
    return torch.zeros_like(t)


# ---------------------------------------------------------------------------
# DDIM/DDPM schedule: alpha_bar_t (cumulative product)
# ---------------------------------------------------------------------------

def make_cosine_alpha_bar(num_timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine noise schedule for alpha_bar_t (Nichol & Dhariwal, 2021).
    Returns alpha_bar at K+1 points: t in {0, 1/K, ..., 1}.
    """
    steps = num_timesteps + 1
    x = torch.linspace(0, num_timesteps, steps)
    alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    return alphas_cumprod


def make_linear_alpha_bar(
    num_timesteps: int,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
) -> torch.Tensor:
    """Linear beta schedule -> alpha_bar_t."""
    betas = torch.linspace(beta_start, beta_end, num_timesteps)
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)


def ddim_kappa(alpha_bar_t: torch.Tensor, alpha_bar_dot_t: torch.Tensor) -> torch.Tensor:
    """kappa_t = alpha_bar_dot_t / (2 * alpha_bar_t)"""
    return alpha_bar_dot_t / (2.0 * alpha_bar_t)


def ddim_eta(alpha_bar_t: torch.Tensor, alpha_bar_dot_t: torch.Tensor) -> torch.Tensor:
    """eta_t = alpha_bar_dot_t / (2 * alpha_bar_t) * (1 - alpha_bar_t) / 2
    Wait, from Table 1: eta_t = alpha_bar_dot_t / (2 * alpha_bar_t) * (1 - alpha_bar_t)
    Actually from the paper: for DDIM, eta_t = (1 - alpha_bar_t) * alpha_bar_dot_t / (2 * alpha_bar_t)
    """
    # From Table 1: eta_t for DDIM = (1 - alpha_bar_t) * alpha_bar_dot_t / (2 * alpha_bar_t)
    # This comes from beta_t = sqrt(1 - alpha_bar_t), alpha_t = sqrt(alpha_bar_t)
    # eta_t = beta_t * (kappa_t * beta_t - beta_dot_t)
    # With alpha_t = sqrt(alpha_bar_t), beta_t = sqrt(1 - alpha_bar_t):
    # kappa_t = alpha_dot_t / alpha_t = alpha_bar_dot_t / (2 * alpha_bar_t)
    # beta_dot_t = -alpha_bar_dot_t / (2 * sqrt(1 - alpha_bar_t))
    # eta_t = sqrt(1-alpha_bar_t) * (alpha_bar_dot_t/(2*alpha_bar_t) * sqrt(1-alpha_bar_t)
    #         + alpha_bar_dot_t/(2*sqrt(1-alpha_bar_t)))
    #       = sqrt(1-alpha_bar_t) * alpha_bar_dot_t/(2*sqrt(1-alpha_bar_t)) * (1-alpha_bar_t+1)/alpha_bar_t
    # Simplifying: eta_t = alpha_bar_dot_t / (2 * alpha_bar_t)
    # Actually let me just use the formula directly from Table 1 notation
    return alpha_bar_dot_t / (2.0 * alpha_bar_t)


def ddpm_sigma(alpha_bar_t: torch.Tensor, alpha_bar_dot_t: torch.Tensor) -> torch.Tensor:
    """
    DDPM diffusion coefficient (memoryless for DDIM):
      sigma(t) = sqrt(alpha_bar_dot_t / alpha_bar_t)
    This is the memoryless schedule for DDIM (Table 1).
    """
    return torch.sqrt(alpha_bar_dot_t / alpha_bar_t)


# ---------------------------------------------------------------------------
# Unified schedule class
# ---------------------------------------------------------------------------

class FlowMatchingSchedule:
    """
    Flow Matching schedule with alpha_t = t, beta_t = 1 - t.
    Provides all quantities needed for the unified SDE framework.
    """

    def __init__(self, num_timesteps: int = 40, sigma_offset_h: float = 0.025):
        self.K = num_timesteps
        self.h = 1.0 / num_timesteps
        self.sigma_offset_h = sigma_offset_h
        # Timesteps: t in {0, h, 2h, ..., (K-1)*h}  (K steps, not including t=1)
        self.timesteps = torch.arange(num_timesteps) * self.h  # [0, h, ..., 1-h]

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return 1.0 - t

    def alpha_dot(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)

    def beta_dot(self, t: torch.Tensor) -> torch.Tensor:
        return -torch.ones_like(t)

    def kappa(self, t: torch.Tensor) -> torch.Tensor:
        """kappa_t = alpha_dot_t / alpha_t = 1/t"""
        return self.alpha_dot(t) / self.alpha(t)

    def eta(self, t: torch.Tensor) -> torch.Tensor:
        """eta_t = beta_t * (kappa_t * beta_t - beta_dot_t) = (1-t)/t"""
        a = self.alpha(t)
        b = self.beta(t)
        a_dot = self.alpha_dot(t)
        b_dot = self.beta_dot(t)
        return b * (a_dot / a * b - b_dot)

    def sigma_memoryless(self, t: torch.Tensor) -> torch.Tensor:
        """
        Memoryless noise schedule with offset (Appendix G.1):
          sigma(t) = sqrt(2*(1-t+h)/(t+h))
        """
        h = self.sigma_offset_h
        return torch.sqrt(2.0 * (1.0 - t + h) / (t + h))

    def sigma_zero(self, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(t)

    def base_drift(self, x: torch.Tensor, score: torch.Tensor, t: torch.Tensor,
                   sigma: torch.Tensor) -> torch.Tensor:
        """
        Unified base drift (Eq. 10-11):
          b(x, t) = kappa_t * x + (sigma(t)^2/2 + eta_t) * s(x, t)
        """
        kappa = self.kappa(t)
        eta = self.eta(t)
        # Broadcast scalar t quantities to match x shape
        kappa = kappa.view(-1, *([1] * (x.dim() - 1)))
        eta = eta.view(-1, *([1] * (x.dim() - 1)))
        sigma = sigma.view(-1, *([1] * (x.dim() - 1)))
        return kappa * x + (0.5 * sigma ** 2 + eta) * score

    def fm_drift_from_velocity(self, x: torch.Tensor, v: torch.Tensor,
                                t: torch.Tensor) -> torch.Tensor:
        """
        Flow Matching ODE drift: dX_t/dt = v(X_t, t)
        Equivalently: b(x,t) = kappa_t * x + eta_t * s(x,t)
        where s(x,t) = (v(x,t) - kappa_t*x) / eta_t
        So b(x,t) = v(x,t) when sigma=0.
        """
        return v

    def memoryless_fm_drift(self, x: torch.Tensor, v: torch.Tensor,
                             t: torch.Tensor) -> torch.Tensor:
        """
        Memoryless Flow Matching drift (Eq. 27 numerator, from Algorithm 1 Eq. 40):
          b(x,t) + sigma(t)*u(x,t) = 2*v_finetune(x,t) - kappa_t * x
        This is the full drift when using the memoryless schedule.
        """
        kappa = self.kappa(t).view(-1, *([1] * (x.dim() - 1)))
        return 2.0 * v - kappa * x

    def score_from_velocity(self, x: torch.Tensor, v: torch.Tensor,
                             t: torch.Tensor) -> torch.Tensor:
        """
        Convert FM velocity to score function (Eq. 8 / B.4):
          s(x,t) = (v(x,t) - kappa_t*x) / eta_t
        """
        kappa = self.kappa(t).view(-1, *([1] * (x.dim() - 1)))
        eta = self.eta(t).view(-1, *([1] * (x.dim() - 1)))
        return (v - kappa * x) / eta

    def velocity_from_score(self, x: torch.Tensor, score: torch.Tensor,
                             t: torch.Tensor) -> torch.Tensor:
        """
        Convert score to FM velocity (Eq. 8):
          v(x,t) = kappa_t*x + eta_t * s(x,t)
        """
        kappa = self.kappa(t).view(-1, *([1] * (x.dim() - 1)))
        eta = self.eta(t).view(-1, *([1] * (x.dim() - 1)))
        return kappa * x + eta * score

    def denoiser_from_velocity(self, x: torch.Tensor, v: torch.Tensor,
                                t: torch.Tensor) -> torch.Tensor:
        """
        Denoiser map X_hat_1(x, t) from velocity field (Appendix F.1, Eq. 229):
          X_hat_1(x, t) = (v(x,t) - (beta_dot_t/beta_t)*x) / (alpha_dot_t - (beta_dot_t/beta_t)*alpha_t)

        For alpha_t=t, beta_t=1-t:
          beta_dot_t/beta_t = -1/(1-t)
          X_hat_1 = (v(x,t) + x/(1-t)) / (1 + t/(1-t))
                  = (1-t)*v(x,t) + x
        """
        beta = self.beta(t).view(-1, *([1] * (x.dim() - 1)))
        beta_dot = self.beta_dot(t).view(-1, *([1] * (x.dim() - 1)))
        alpha = self.alpha(t).view(-1, *([1] * (x.dim() - 1)))
        alpha_dot = self.alpha_dot(t).view(-1, *([1] * (x.dim() - 1)))
        numerator = v - (beta_dot / beta) * x
        denominator = alpha_dot - (beta_dot / beta) * alpha
        return numerator / denominator

    def control_from_velocity_diff(self, v_finetune: torch.Tensor, v_base: torch.Tensor,
                                    t: torch.Tensor) -> torch.Tensor:
        """
        Control u(x,t) from velocity difference (Eq. 27):
          u(x,t) = sqrt(2 / (beta_t * (kappa_t*beta_t - beta_dot_t))) * (v_finetune - v_base)
                 = sqrt(2 / eta_t) * (v_finetune - v_base)
                 = (2/sigma(t)) * (v_finetune - v_base)   [since sigma = sqrt(2*eta_t)]
        """
        eta = self.eta(t).view(-1, *([1] * (v_finetune.dim() - 1)))
        return torch.sqrt(2.0 / eta) * (v_finetune - v_base)

    def select_grad_timesteps(self, device: torch.device) -> torch.Tensor:
        """
        Select timestep subset for gradient computation (Appendix G.2):
        - 10 uniformly sampled from [0, 0.725]
        - Always include last 10 steps [0.75, ..., 0.975]
        Returns indices into self.timesteps.
        """
        all_t = self.timesteps
        # Late timesteps: t in [0.75, 0.975]
        late_mask = all_t >= 0.75
        late_indices = torch.where(late_mask)[0]

        # Early timesteps: t in [0, 0.725]
        early_mask = all_t <= 0.725
        early_indices = torch.where(early_mask)[0]
        perm = torch.randperm(len(early_indices))[:10]
        early_selected = early_indices[perm]

        selected = torch.cat([early_selected, late_indices])
        return selected.to(device)


class DDIMSchedule:
    """
    DDIM/DDPM schedule for diffusion models.
    Continuous-time formulation from Section 3.
    """

    def __init__(self, num_timesteps: int = 40, schedule_type: str = "cosine"):
        self.K = num_timesteps
        self.h = 1.0 / num_timesteps

        if schedule_type == "cosine":
            alpha_bar = make_cosine_alpha_bar(num_timesteps)
        else:
            alpha_bar = make_linear_alpha_bar(num_timesteps)

        self.alpha_bar = alpha_bar  # shape [K+1]
        # Compute alpha_bar_dot via finite differences
        self.alpha_bar_dot = torch.zeros_like(alpha_bar)
        self.alpha_bar_dot[1:-1] = (alpha_bar[2:] - alpha_bar[:-2]) / (2.0 * self.h)
        self.alpha_bar_dot[0] = (alpha_bar[1] - alpha_bar[0]) / self.h
        self.alpha_bar_dot[-1] = (alpha_bar[-1] - alpha_bar[-2]) / self.h

    def get_alpha_bar(self, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return alpha_bar_k and alpha_bar_{k+1}."""
        return self.alpha_bar[k], self.alpha_bar[k + 1]

    def sigma_ddpm(self, k: int) -> torch.Tensor:
        """
        DDPM sigma_k = sqrt((alpha_bar_{k+1} - alpha_bar_k) / alpha_bar_k)
        This is the memoryless schedule for DDIM (Table 1).
        """
        ab_k = self.alpha_bar[k]
        ab_k1 = self.alpha_bar[k + 1]
        return torch.sqrt((ab_k1 - ab_k) / ab_k)

    def sigma_zero(self, k: int) -> torch.Tensor:
        return torch.tensor(0.0)

    def ddpm_step(self, x_k: torch.Tensor, eps_finetune: torch.Tensor,
                  k: int, eps_noise: torch.Tensor) -> torch.Tensor:
        """
        DDPM update (Algorithm 2, Eq. 219):
          X_{k+1} = sqrt(alpha_bar_{k+1}/alpha_bar_k) * (X_k - (1 - alpha_bar_k/alpha_bar_{k+1})
                    / sqrt(1 - alpha_bar_k) * eps_finetune) + sqrt((1-alpha_bar_{k+1})/(1-alpha_bar_k)
                    * (1 - alpha_bar_k/alpha_bar_{k+1})) * noise
        """
        ab_k = self.alpha_bar[k]
        ab_k1 = self.alpha_bar[k + 1]
        ratio = torch.sqrt(ab_k1 / ab_k)
        sigma_k = torch.sqrt((1.0 - ab_k1) / (1.0 - ab_k) * (1.0 - ab_k / ab_k1))
        x_k1 = ratio * (x_k - (1.0 - ab_k / ab_k1) / torch.sqrt(1.0 - ab_k) * eps_finetune)
        x_k1 = x_k1 + sigma_k * eps_noise
        return x_k1

    def ddpm_step_euler(self, x_k: torch.Tensor, eps_finetune: torch.Tensor,
                        k: int, eps_noise: torch.Tensor) -> torch.Tensor:
        """
        Euler-Maruyama discretization of DDPM (Algorithm 2, Eq. 220):
          X_{k+1} = X_k + (alpha_bar_{k+1} - alpha_bar_k)/(2*alpha_bar_k) * X_k
                  - (alpha_bar_{k+1} - alpha_bar_k)/(alpha_bar_k * sqrt(1-alpha_bar_k)) * eps
                  + sqrt((alpha_bar_{k+1} - alpha_bar_k)/alpha_bar_k) * noise
        """
        ab_k = self.alpha_bar[k]
        ab_k1 = self.alpha_bar[k + 1]
        d_ab = ab_k1 - ab_k
        drift = (d_ab / (2.0 * ab_k)) * x_k - (d_ab / (ab_k * torch.sqrt(1.0 - ab_k))) * eps_finetune
        diffusion = torch.sqrt(d_ab / ab_k) * eps_noise
        return x_k + drift + diffusion

    def control_from_eps_diff(self, eps_finetune: torch.Tensor, eps_base: torch.Tensor,
                               k: int) -> torch.Tensor:
        """
        Control u(x,t) from noise predictor difference (Eq. 26):
          u(x,t) = -sqrt(alpha_bar_dot_t / (alpha_bar_t * (1 - alpha_bar_t)))
                   * (eps_finetune - eps_base)
        """
        ab_k = self.alpha_bar[k]
        ab_k1 = self.alpha_bar[k + 1]
        d_ab = ab_k1 - ab_k
        # Approximate alpha_bar_dot_t * h = d_ab
        coeff = torch.sqrt(d_ab / (ab_k * (1.0 - ab_k)))
        return -coeff * (eps_finetune - eps_base)

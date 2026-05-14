import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from models.networks import ResidualMLP


class GaussianDiffusion(nn.Module):
    """DDPM with linear noise schedule.

    Implements the forward process q(x^{n+1} | x^n) and reverse process
    p_θ(x^{n-1} | x^n) for denoising diffusion probabilistic models
    (Ho et al., 2020; Sohl-Dickstein et al., 2015).
    """

    def __init__(
        self,
        n_diffusion_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        clip_denoised: bool = True,
    ):
        super().__init__()
        self.n_diffusion_steps = n_diffusion_steps
        self.clip_denoised = clip_denoised

        betas = self._make_beta_schedule(beta_schedule, n_diffusion_steps, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.register_buffer("betas", torch.FloatTensor(betas))
        self.register_buffer("alphas_cumprod", torch.FloatTensor(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", torch.FloatTensor(alphas_cumprod_prev))
        self.register_buffer("sqrt_alphas_cumprod", torch.FloatTensor(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.FloatTensor(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer("log_one_minus_alphas_cumprod", torch.FloatTensor(np.log(1.0 - alphas_cumprod)))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.FloatTensor(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.FloatTensor(np.sqrt(1.0 / alphas_cumprod - 1)))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", torch.FloatTensor(posterior_variance))
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.FloatTensor(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            torch.FloatTensor(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            torch.FloatTensor((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)),
        )

    @staticmethod
    def _make_beta_schedule(
        schedule: str, n_steps: int, beta_start: float, beta_end: float
    ) -> np.ndarray:
        if schedule == "linear":
            return np.linspace(beta_start, beta_end, n_steps)
        elif schedule == "cosine":
            steps = n_steps + 1
            x = np.linspace(0, n_steps, steps)
            alphas_cumprod = np.cos(((x / n_steps) + 0.008) / 1.008 * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return np.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {schedule}")

    def _extract(self, arr: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
        batch_size = t.shape[0]
        out = arr.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(
        self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward process: q(x^n | x^0) = N(sqrt(ᾱ_n) x^0, (1 - ᾱ_n) I)."""
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_start_from_noise(
        self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        sqrt_recip_alphas_cumprod_t = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod_t = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise

    def q_posterior(
        self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_noise = model(x, t, cond)
        x_recon = self.predict_start_from_noise(x, t, pred_noise)
        if self.clip_denoised:
            x_recon = x_recon.clamp(-1.0, 1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_recon, x, t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        model_mean, _, model_log_variance = self.p_mean_variance(model, x, t, cond)
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: Tuple,
        cond: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.n_diffusion_steps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, cond)
        return x

    def training_loss(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x_start.shape[0]
        t = torch.randint(0, self.n_diffusion_steps, (batch_size,), device=x_start.device)
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        pred_noise = model(x_noisy, t, cond)
        return F.mse_loss(pred_noise, noise)


class ConditionalDiffusion(nn.Module):
    """Conditional diffusion model with classifier-free guidance (CFG).

    Implements the PGR generative model G that learns p_D(τ | c) where
    c = F(τ) is the relevance condition.

    Training objective (Eq. 2 from paper):
        E[||ε_θ(x^n, n, (1-p)·y + p·∅)||²]
    where p ~ Bernoulli(p_uncond) randomly drops the condition.

    Sampling with CFG:
        ε_guided = ω · ε_θ(x^n, n, y) + (1-ω) · ε_θ(x^n, n, ∅)
    """

    NULL_COND_VALUE = 0.0

    def __init__(
        self,
        transition_dim: int,
        hidden_dim: int = 256,
        n_hidden_layers: int = 4,
        time_embed_dim: int = 128,
        cond_embed_dim: int = 128,
        n_diffusion_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        p_uncond: float = 0.25,
        guidance_scale: float = 1.5,
        clip_denoised: bool = True,
    ):
        super().__init__()
        self.transition_dim = transition_dim
        self.p_uncond = p_uncond
        self.guidance_scale = guidance_scale

        self.model = ResidualMLP(
            input_dim=transition_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            time_embed_dim=time_embed_dim,
            cond_embed_dim=cond_embed_dim,
        )
        self.diffusion = GaussianDiffusion(
            n_diffusion_steps=n_diffusion_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            clip_denoised=clip_denoised,
        )

        self.register_buffer("null_cond", torch.tensor([[self.NULL_COND_VALUE]]))

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """CFG-guided noise prediction at sampling time.

        ε_guided = ω · ε_cond + (1-ω) · ε_uncond
        """
        null_cond = self.null_cond.expand(x.shape[0], -1)
        eps_cond = self.model(x, t, cond)
        eps_uncond = self.model(x, t, null_cond)
        return self.guidance_scale * eps_cond + (1.0 - self.guidance_scale) * eps_uncond

    def loss(
        self,
        transitions: torch.Tensor,
        conditions: torch.Tensor,
    ) -> torch.Tensor:
        """CFG training loss (Eq. 2 from paper).

        Randomly drops condition with probability p_uncond.
        """
        batch_size = transitions.shape[0]
        drop_mask = torch.bernoulli(
            torch.full((batch_size, 1), self.p_uncond, device=transitions.device)
        )
        null_cond = self.null_cond.expand(batch_size, -1)
        effective_cond = (1.0 - drop_mask) * conditions + drop_mask * null_cond

        t = torch.randint(
            0, self.diffusion.n_diffusion_steps, (batch_size,), device=transitions.device
        )
        noise = torch.randn_like(transitions)
        x_noisy = self.diffusion.q_sample(transitions, t, noise)
        pred_noise = self.model(x_noisy, t, effective_cond)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        conditions: torch.Tensor,
        use_guidance: bool = True,
    ) -> torch.Tensor:
        """Generate synthetic transitions conditioned on relevance values.

        Uses CFG at sampling time when use_guidance=True.
        """
        device = next(self.parameters()).device
        shape = (n_samples, self.transition_dim)
        x = torch.randn(shape, device=device)

        model_fn = self.forward_with_cfg if use_guidance else self.model

        for i in reversed(range(self.diffusion.n_diffusion_steps)):
            t = torch.full((n_samples,), i, device=device, dtype=torch.long)
            x = self.diffusion.p_sample(model_fn, x, t, conditions)
        return x

    @torch.no_grad()
    def sample_ddim(
        self,
        n_samples: int,
        conditions: torch.Tensor,
        ddim_steps: int = 50,
        eta: float = 0.0,
        use_guidance: bool = True,
    ) -> torch.Tensor:
        """DDIM sampling for faster generation (Song et al., 2020)."""
        device = next(self.parameters()).device
        shape = (n_samples, self.transition_dim)
        x = torch.randn(shape, device=device)

        n_total = self.diffusion.n_diffusion_steps
        step_size = n_total // ddim_steps
        timesteps = list(reversed(range(0, n_total, step_size)))

        model_fn = self.forward_with_cfg if use_guidance else self.model

        for i, t_val in enumerate(timesteps):
            t = torch.full((n_samples,), t_val, device=device, dtype=torch.long)
            pred_noise = model_fn(x, t, conditions)
            alpha_cumprod_t = self.diffusion._extract(self.diffusion.alphas_cumprod, t, x.shape)
            x_start = (x - (1 - alpha_cumprod_t).sqrt() * pred_noise) / alpha_cumprod_t.sqrt()
            if self.diffusion.clip_denoised:
                x_start = x_start.clamp(-1.0, 1.0)

            if i < len(timesteps) - 1:
                t_prev_val = timesteps[i + 1]
                t_prev = torch.full((n_samples,), t_prev_val, device=device, dtype=torch.long)
                alpha_cumprod_prev = self.diffusion._extract(self.diffusion.alphas_cumprod, t_prev, x.shape)
            else:
                alpha_cumprod_prev = torch.ones_like(alpha_cumprod_t)

            sigma = eta * ((1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_prev)).sqrt()
            noise = torch.randn_like(x)
            x = alpha_cumprod_prev.sqrt() * x_start + (1 - alpha_cumprod_prev - sigma ** 2).clamp(0).sqrt() * pred_noise + sigma * noise

        return x


def build_transition_tensor(
    states: torch.Tensor,
    actions: torch.Tensor,
    next_states: torch.Tensor,
    rewards: torch.Tensor,
) -> torch.Tensor:
    """Concatenate (s, a, s', r) into a flat transition vector for diffusion."""
    if rewards.ndim == 1:
        rewards = rewards.unsqueeze(-1)
    return torch.cat([states, actions, next_states, rewards], dim=-1)


def unpack_transition_tensor(
    transitions: torch.Tensor,
    state_dim: int,
    action_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack flat transition vector back into (s, a, s', r)."""
    s = transitions[:, :state_dim]
    a = transitions[:, state_dim: state_dim + action_dim]
    sp = transitions[:, state_dim + action_dim: 2 * state_dim + action_dim]
    r = transitions[:, 2 * state_dim + action_dim:]
    return s, a, sp, r


class TransitionNormalizer:
    """Normalizes transitions to [-1, 1] for diffusion model training."""

    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, transitions: np.ndarray):
        self.mean = transitions.mean(axis=0)
        self.std = transitions.std(axis=0) + 1e-8

    def normalize(self, transitions: np.ndarray) -> np.ndarray:
        return (transitions - self.mean) / self.std

    def denormalize(self, transitions: np.ndarray) -> np.ndarray:
        return transitions * self.std + self.mean

    def normalize_tensor(self, transitions: torch.Tensor) -> torch.Tensor:
        mean = torch.FloatTensor(self.mean).to(transitions.device)
        std = torch.FloatTensor(self.std).to(transitions.device)
        return (transitions - mean) / std

    def denormalize_tensor(self, transitions: torch.Tensor) -> torch.Tensor:
        mean = torch.FloatTensor(self.mean).to(transitions.device)
        std = torch.FloatTensor(self.std).to(transitions.device)
        return transitions * std + mean

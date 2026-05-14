"""
1D toy experiment reproducing Figure 2 from the paper.

Demonstrates that:
(a) Base Flow Matching model generates from p_base
(b,c) Fine-tuning with constant σ(t) leads to biased distributions
(d) Fine-tuning with memoryless noise schedule converges to tilted distribution

Setup:
- 1D Flow Matching model: p_base = N(0, 1)
- Reward: r(x) = -0.5*(x-3)^2  (Gaussian centered at 3)
- Tilted distribution: p*(x) ∝ p_base(x) exp(r(x)) = N(1.5, 0.5)
- Reference flow: X_t = (1-t)*X_0 + t*X_1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Callable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 1D Flow Matching model
# ---------------------------------------------------------------------------

class VelocityNet1D(nn.Module):
    """Simple MLP velocity field for 1D experiments."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),  # input: (x, t)
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1]
            t: [B]
        Returns:
            velocity: [B, 1]
        """
        inp = torch.cat([x, t.unsqueeze(-1)], dim=-1)
        return self.net(inp)


# ---------------------------------------------------------------------------
# 1D noise schedules
# ---------------------------------------------------------------------------

def fm_sigma_memoryless_1d(t: torch.Tensor, h: float = 0.025) -> torch.Tensor:
    """σ(t) = √(2(1-t+h)/(t+h))"""
    return torch.sqrt(2.0 * (1.0 - t + h) / (t + h))


def fm_kappa_1d(t: torch.Tensor) -> torch.Tensor:
    """κ_t = 1/t"""
    return 1.0 / t


def fm_eta_1d(t: torch.Tensor) -> torch.Tensor:
    """η_t = (1-t)/t"""
    return (1.0 - t) / t


# ---------------------------------------------------------------------------
# 1D pre-training
# ---------------------------------------------------------------------------

def pretrain_fm_1d(
    model: VelocityNet1D,
    p_data_mean: float = 0.0,
    p_data_std: float = 1.0,
    num_steps: int = 5000,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: torch.device = None,
) -> VelocityNet1D:
    """
    Pre-train 1D Flow Matching model.
    Target: p_base = N(p_data_mean, p_data_std^2)
    """
    if device is None:
        device = torch.device("cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(num_steps):
        # Sample data
        x1 = torch.randn(batch_size, 1, device=device) * p_data_std + p_data_mean
        x0 = torch.randn(batch_size, 1, device=device)
        t = torch.rand(batch_size, device=device) * 0.98 + 0.01  # avoid t=0,1

        # Reference flow: X_t = (1-t)*X_0 + t*X_1
        t_view = t.unsqueeze(-1)
        x_t = (1.0 - t_view) * x0 + t_view * x1
        target = x1 - x0  # velocity target

        v_pred = model(x_t, t)
        loss = F.mse_loss(v_pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


# ---------------------------------------------------------------------------
# 1D fine-tuning with different noise schedules
# ---------------------------------------------------------------------------

def finetune_1d(
    model: VelocityNet1D,
    base_model: VelocityNet1D,
    reward_fn: Callable,
    sigma_fn: Callable,
    num_steps: int = 2000,
    batch_size: int = 256,
    lr: float = 1e-3,
    reward_lambda: float = 5.0,
    K: int = 40,
    device: torch.device = None,
    method: str = "adjoint_matching",
) -> VelocityNet1D:
    """
    Fine-tune 1D Flow Matching model with specified noise schedule.

    Args:
        model: Model to fine-tune
        base_model: Frozen base model
        reward_fn: r(x) -> scalar
        sigma_fn: σ(t) function
        method: "adjoint_matching" or "draft_1"
    """
    if device is None:
        device = torch.device("cpu")
    model = model.to(device)
    base_model = base_model.to(device)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    h = 1.0 / K

    for step in range(num_steps):
        x0 = torch.randn(batch_size, 1, device=device)

        if method == "adjoint_matching":
            loss = _adjoint_matching_loss_1d(
                model, base_model, reward_fn, sigma_fn, x0, K, h, reward_lambda, device
            )
        elif method == "draft_1":
            loss = _draft_loss_1d(model, reward_fn, sigma_fn, x0, K, h, reward_lambda, device)
        else:
            raise ValueError(f"Unknown method: {method}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def _adjoint_matching_loss_1d(
    model, base_model, reward_fn, sigma_fn, x0, K, h, reward_lambda, device
):
    """1D Adjoint Matching loss."""
    # Sample trajectory
    with torch.no_grad():
        x = x0.clone()
        xs = [x]
        for i in range(K):
            t_val = (i + 0.5) / K  # midpoint
            t = torch.full((x.shape[0],), t_val, device=device)
            v = model(x, t)
            kappa = fm_kappa_1d(t).unsqueeze(-1)
            sigma = sigma_fn(t).unsqueeze(-1)
            drift = 2.0 * v - kappa * x
            noise = torch.randn_like(x)
            x = x + h * drift + (h ** 0.5) * sigma * noise
            xs.append(x)

    # Noiseless final state
    with torch.no_grad():
        t_last = torch.full((xs[-2].shape[0],), (K - 0.5) / K, device=device)
        v_last = base_model(xs[-2], t_last)
        x1_hat = xs[-2] + h * v_last

    # Reward gradient
    x1_req = x1_hat.detach().requires_grad_(True)
    r = reward_lambda * reward_fn(x1_req)
    reward_grad = torch.autograd.grad(r.sum(), x1_req)[0]

    # Lean adjoint backwards
    a_tilde = -reward_grad
    a_tildes = [a_tilde]
    for i in range(K - 1, -1, -1):
        t_val = (i + 0.5) / K
        t = torch.full((xs[i].shape[0],), t_val, device=device)
        x_t = xs[i].detach()
        a_t = a_tildes[-1].detach()
        kappa = fm_kappa_1d(t).unsqueeze(-1)

        x_t_req = x_t.requires_grad_(True)
        v_base = base_model(x_t_req, t)
        b_base = 2.0 * v_base - kappa * x_t_req
        vjp = torch.autograd.grad(b_base, x_t_req, grad_outputs=a_t,
                                   create_graph=False)[0]
        a_prev = a_t + h * vjp
        a_tildes.append(a_prev)
    a_tildes.reverse()

    # Loss
    total_loss = torch.tensor(0.0, device=device)
    for i in range(K):
        t_val = (i + 0.5) / K
        t = torch.full((xs[i].shape[0],), t_val, device=device)
        sigma = sigma_fn(t).unsqueeze(-1)

        v_ft = model(xs[i].detach(), t)
        v_bs = base_model(xs[i].detach(), t)
        a_t = a_tildes[i].detach()

        term = (2.0 / sigma) * (v_ft - v_bs) + sigma * a_t
        total_loss = total_loss + (term ** 2).mean()

    return total_loss / K


def _draft_loss_1d(model, reward_fn, sigma_fn, x0, K, h, reward_lambda, device):
    """1D DRaFT-1 loss."""
    with torch.no_grad():
        x = x0.clone()
        for i in range(K - 1):
            t_val = (i + 0.5) / K
            t = torch.full((x.shape[0],), t_val, device=device)
            v = model(x, t)
            kappa = fm_kappa_1d(t).unsqueeze(-1)
            sigma = sigma_fn(t).unsqueeze(-1)
            drift = 2.0 * v - kappa * x
            noise = torch.randn_like(x)
            x = x + h * drift + (h ** 0.5) * sigma * noise

    # Last step with gradients
    t_val = (K - 0.5) / K
    t = torch.full((x.shape[0],), t_val, device=device)
    v = model(x, t)
    kappa = fm_kappa_1d(t).unsqueeze(-1)
    sigma = sigma_fn(t).unsqueeze(-1)
    drift = 2.0 * v - kappa * x
    with torch.no_grad():
        noise = torch.randn_like(x)
    x = x + h * drift + (h ** 0.5) * sigma * noise

    r = reward_fn(x)
    return -reward_lambda * r.mean()


# ---------------------------------------------------------------------------
# Sampling from 1D model
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_1d(
    model: VelocityNet1D,
    n_samples: int = 10000,
    K: int = 40,
    sigma_fn: Optional[Callable] = None,
    device: torch.device = None,
) -> np.ndarray:
    """Sample from 1D Flow Matching model."""
    if device is None:
        device = torch.device("cpu")
    h = 1.0 / K
    x = torch.randn(n_samples, 1, device=device)

    for i in range(K):
        t_val = (i + 0.5) / K
        t = torch.full((n_samples,), t_val, device=device)
        v = model(x, t)

        if sigma_fn is not None:
            kappa = fm_kappa_1d(t).unsqueeze(-1)
            sigma = sigma_fn(t).unsqueeze(-1)
            drift = 2.0 * v - kappa * x
            noise = torch.randn_like(x)
            x = x + h * drift + (h ** 0.5) * sigma * noise
        else:
            x = x + h * v

    return x.cpu().numpy().flatten()


# ---------------------------------------------------------------------------
# Tilted distribution (analytical)
# ---------------------------------------------------------------------------

def tilted_distribution_1d(
    x: np.ndarray,
    p_base_mean: float = 0.0,
    p_base_std: float = 1.0,
    reward_fn_np: Callable = None,
) -> np.ndarray:
    """
    Compute tilted distribution p*(x) ∝ p_base(x) exp(r(x)).

    For r(x) = -0.5*(x-3)^2 and p_base = N(0,1):
    p*(x) ∝ exp(-x^2/2) * exp(-0.5*(x-3)^2)
           = exp(-x^2 + 3x - 4.5)
           = exp(-(x - 1.5)^2 / 0.5 - const)
    So p*(x) = N(1.5, 0.5)
    """
    from scipy.stats import norm
    log_p_base = norm.logpdf(x, loc=p_base_mean, scale=p_base_std)
    if reward_fn_np is not None:
        log_reward = reward_fn_np(x)
    else:
        # Default: r(x) = -0.5*(x-3)^2
        log_reward = -0.5 * (x - 3.0) ** 2
    log_p_star = log_p_base + log_reward
    # Normalize
    log_p_star -= np.max(log_p_star)
    p_star = np.exp(log_p_star)
    dx = x[1] - x[0] if len(x) > 1 else 1.0
    p_star /= (p_star.sum() * dx)
    return p_star


# ---------------------------------------------------------------------------
# Full 1D experiment (Figure 2)
# ---------------------------------------------------------------------------

def run_toy_experiment(
    device: torch.device = None,
    num_pretrain_steps: int = 5000,
    num_finetune_steps: int = 2000,
    reward_lambda: float = 5.0,
    K: int = 40,
    n_samples: int = 10000,
    seed: int = 42,
) -> dict:
    """
    Reproduce Figure 2: comparison of noise schedules for fine-tuning.

    Returns dict with samples from:
    - base model
    - fine-tuned with constant σ=0.5 (biased)
    - fine-tuned with constant σ=1.0 (biased)
    - fine-tuned with memoryless σ(t) (correct)
    - analytical tilted distribution
    """
    if device is None:
        device = torch.device("cpu")
    torch.manual_seed(seed)

    # Reward: r(x) = -0.5*(x-3)^2
    def reward_fn(x: torch.Tensor) -> torch.Tensor:
        return -0.5 * (x - 3.0) ** 2

    h = 1.0 / K

    # Pre-train base model
    base_model = VelocityNet1D(hidden_dim=64)
    base_model = pretrain_fm_1d(base_model, num_steps=num_pretrain_steps, device=device)

    # Sample from base model (ODE)
    samples_base = sample_1d(base_model, n_samples=n_samples, K=K, device=device)

    results = {"base": samples_base}

    # Fine-tune with different noise schedules
    sigma_configs = {
        "constant_0.5": lambda t: torch.full_like(t, 0.5),
        "constant_1.0": lambda t: torch.full_like(t, 1.0),
        "memoryless": lambda t: fm_sigma_memoryless_1d(t, h=h),
    }

    for sigma_name, sigma_fn in sigma_configs.items():
        # Copy base model
        ft_model = VelocityNet1D(hidden_dim=64)
        ft_model.load_state_dict(base_model.state_dict())

        # Fine-tune
        ft_model = finetune_1d(
            model=ft_model,
            base_model=base_model,
            reward_fn=reward_fn,
            sigma_fn=sigma_fn,
            num_steps=num_finetune_steps,
            reward_lambda=reward_lambda,
            K=K,
            device=device,
            method="adjoint_matching",
        )

        # Sample (use same sigma for sampling as fine-tuning)
        samples = sample_1d(ft_model, n_samples=n_samples, K=K,
                            sigma_fn=sigma_fn, device=device)
        results[sigma_name] = samples

    # Analytical tilted distribution
    x_grid = np.linspace(-4, 6, 1000)
    p_star = tilted_distribution_1d(x_grid)
    results["tilted_analytical"] = (x_grid, p_star)

    return results


def plot_toy_experiment(results: dict, save_path: str = None):
    """Plot Figure 2 from the paper."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not available for plotting")
        return

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    x_grid, p_star = results["tilted_analytical"]

    configs = [
        ("base", "Base Flow Matching", "blue"),
        ("constant_0.5", "Constant σ=0.5 (biased)", "red"),
        ("constant_1.0", "Constant σ=1.0 (biased)", "orange"),
        ("memoryless", "Memoryless σ(t) (correct)", "green"),
    ]

    for ax, (key, title, color) in zip(axes, configs):
        samples = results[key]
        ax.hist(samples, bins=50, density=True, alpha=0.6, color=color, label="Generated")
        ax.plot(x_grid, p_star, "k--", linewidth=2, label="p*(x)")
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-4, 6)
        ax.legend(fontsize=8)
        ax.set_xlabel("x")
        ax.set_ylabel("Density")

    plt.suptitle("Figure 2: Effect of noise schedule on fine-tuning", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    return fig


if __name__ == "__main__":
    print("Running 1D toy experiment (Figure 2)...")
    results = run_toy_experiment(
        num_pretrain_steps=5000,
        num_finetune_steps=2000,
        reward_lambda=5.0,
    )
    plot_toy_experiment(results, save_path="figure2_toy_experiment.png")
    print("Done. Saved to figure2_toy_experiment.png")

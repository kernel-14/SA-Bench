r"""Training loop for score estimation (score matching).

This is a theoretical paper that does not train neural networks.
The experiments in Appendix A use exact score functions.

This file provides a placeholder for score matching on the Gaussian target
distribution, which could be used to verify the full pipeline with estimated
scores (Assumption 2 in the paper). The score matching loss is:

    L = E_{x_0 ~ p_data, t, x_t ~ N(sqrt(alpha_bar_t)*x_0, (1-alpha_bar_t)*I)}
        [|| s_t(x_t) - s_t^*(x_t) ||_2^2]

where s_t^* is the exact score and s_t is the estimate.
"""

import torch
import torch.nn as nn

from score_function import GaussianScoreFunction
from sampler import build_alpha_hat_schedule


class ScoreNetwork(nn.Module):
    """Simple MLP for estimating the score function s_t(x).

    The network takes (x, t_embedding) as input and outputs a score estimate.
    For d-dimensional data, we use a small residual network.
    """

    def __init__(self, d: int, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        self.d = d
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        layers = []
        layers.append(nn.Linear(d + hidden_dim, hidden_dim))
        for _ in range(num_layers - 2):
            layers.append(nn.SiLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, d))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute score estimate s_t(x).

        Args:
            x: Input samples, shape (batch, d).
            t: Time values in [0, 1] (tau), shape (batch,).
        """
        t_embed = self.time_mlp(t.unsqueeze(-1))
        combined = torch.cat([x, t_embed], dim=-1)
        return self.net(combined)


def score_matching_loss(
    x_0: torch.Tensor,
    score_net: ScoreNetwork,
    exact_score_fn: GaussianScoreFunction,
    alpha_bar_min: float = 0.0,
    alpha_bar_max: float = 0.999,
) -> torch.Tensor:
    """Compute the score matching loss (denoising score matching).

    Args:
        x_0: Samples from target distribution, shape (batch, d).
        score_net: Score estimation network.
        exact_score_fn: Ground truth score function.
        alpha_bar_min: Minimum noise level.
        alpha_bar_max: Maximum noise level.

    Returns:
        Scalar loss value.
    """
    batch_size = x_0.shape[0]
    device = x_0.device

    alpha_bar = alpha_bar_min + (
        alpha_bar_max - alpha_bar_min
    ) * torch.rand(batch_size, device=device)

    noise = torch.randn_like(x_0)
    sqrt_alpha = torch.sqrt(alpha_bar).unsqueeze(-1)
    sqrt_one_minus = torch.sqrt(1.0 - alpha_bar).unsqueeze(-1)
    x_t = sqrt_alpha * x_0 + sqrt_one_minus * noise

    tau = 1.0 - alpha_bar
    estimated_score = score_net(x_t, tau)
    exact_score = exact_score_fn.score_batch(x_t, alpha_bar)

    loss = ((estimated_score - exact_score) ** 2).sum(dim=-1).mean()
    return loss


def train_score_network(
    d: int,
    k: int,
    sigma_max: float,
    num_steps: int = 10000,
    batch_size: int = 256,
    lr: float = 1e-3,
):
    """Train a score network on the Gaussian target distribution.

    Args:
        d: Data dimension.
        k: Number of non-zero variance components.
        sigma_max: Maximum variance.
        num_steps: Number of training steps.
        batch_size: Batch size per step.
        lr: Learning rate.
    """
    score_fn = GaussianScoreFunction(d=d, k=k, sigma_max=sigma_max)
    score_net = ScoreNetwork(d=d)
    optimizer = torch.optim.Adam(score_net.parameters(), lr=lr)

    for step in range(num_steps):
        x_0 = torch.randn(batch_size, d)
        x_0 = x_0 * torch.sqrt(score_fn.sigma_diag.unsqueeze(0))

        loss = score_matching_loss(x_0, score_net, score_fn)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 1000 == 0:
            print(f"Step {step:6d}: loss = {loss.item():.6f}")

    return score_net


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=10)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--sigma_max", type=float, default=10.0)
    parser.add_argument("--num_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    score_net = train_score_network(
        d=args.d,
        k=args.k,
        sigma_max=args.sigma_max,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
    )
    torch.save(score_net.state_dict(), "score_net.pt")
    print("Score network saved to score_net.pt")

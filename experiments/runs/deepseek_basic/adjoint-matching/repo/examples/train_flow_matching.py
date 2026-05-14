"""
Example training script for Adjoint Matching fine-tuning of Flow Matching models.

This demonstrates the complete fine-tuning pipeline described in Algorithm 1:
1. Setup memoryless noise schedule
2. Initialize fine-tuned model from base model
3. Train with Adjoint Matching loss
4. Generate samples after fine-tuning with arbitrary noise schedule
"""

import torch
import torch.nn as nn
import argparse
import math

from adjoint_matching.noise_schedule import FlowMatchingNoiseSchedule
from adjoint_matching.adjoint_matching import AdjointMatchingTrainer
from adjoint_matching.baselines import (
    ContinuousAdjointLoss,
    DiscreteAdjointLoss,
    DRaFTLoss,
    RefLLoss,
)
from adjoint_matching.evaluation import EvaluationMetrics, ClassifierFreeGuidance


# Placeholder Flow Matching model (velocity field v(x, t))
class SimpleFlowMatchingModel(nn.Module):
    """
    Simple Flow Matching velocity field model.

    In a real application, this would be a U-Net or DiT architecture
    trained on latent representations of images.
    """

    def __init__(self, dim: int = 512, hidden_dim: int = 1024):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),  # +1 for time
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute velocity v(x, t).

        Args:
            x: State [B, D].
            t: Time [B] or scalar.

        Returns:
            Velocity [B, D].
        """
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        t = t.unsqueeze(-1)
        xt = torch.cat([x, t], dim=-1)
        return self.net(xt)


# Placeholder reward model
class SimpleRewardModel(nn.Module):
    """Simple reward model (placeholder for ImageReward)."""

    def __init__(self, dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute reward r(x).

        Args:
            x: Generated sample [B, D].

        Returns:
            Reward scalar per sample [B].
        """
        return self.net(x).squeeze(-1)


def main():
    parser = argparse.ArgumentParser(description="Adjoint Matching fine-tuning")
    parser.add_argument("--model_type", type=str, default="flow_matching",
                        choices=["flow_matching", "ddim"])
    parser.add_argument("--method", type=str, default="adjoint_matching",
                        choices=["adjoint_matching", "continuous_adjoint",
                                 "discrete_adjoint", "draft", "refl"])
    parser.add_argument("--lambda_reg", type=float, default=12500,
                        help="Reward scaling factor")
    parser.add_argument("--num_steps", type=int, default=40,
                        help="Number of discretization steps")
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--num_iterations", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dim = 512  # Latent dimension

    # Initialize models
    base_model = SimpleFlowMatchingModel(dim=dim).to(device)
    fine_tuned_model = SimpleFlowMatchingModel(dim=dim).to(device)
    reward_model = SimpleRewardModel(dim=dim).to(device)

    # Setup noise schedule
    if args.model_type == "flow_matching":
        noise_schedule = FlowMatchingNoiseSchedule(
            num_steps=args.num_steps, offset=True
        )

        def reward_fn(x):
            return args.lambda_reg * reward_model(x)

    else:
        from adjoint_matching.noise_schedule import DDIMMemorylessNoiseSchedule
        noise_schedule = DDIMMemorylessNoiseSchedule(num_steps=args.num_steps)

        def reward_fn(x):
            return args.lambda_reg * reward_model(x)

    print(f"Training {args.method} with λ={args.lambda_reg}, K={args.num_steps}")

    if args.method == "adjoint_matching":
        trainer = AdjointMatchingTrainer(
            base_model=base_model,
            fine_tuned_model=fine_tuned_model,
            noise_schedule=noise_schedule,
            reward_fn=reward_fn,
            model_type=args.model_type,
            lambda_reg=args.lambda_reg,
            lr=args.lr,
        )

        for iteration in range(args.num_iterations):
            metrics = trainer.train_step(args.batch_size, device)
            if iteration % 50 == 0:
                print(f"Iter {iteration}: loss = {metrics['loss']:.4f}")

    elif args.method == "draft":
        # DRaFT-1 training
        draft_loss = DRaFTLoss(
            base_model=base_model,
            noise_schedule=noise_schedule,
            reward_fn=reward_fn,
            lambda_reg=args.lambda_reg,
            K_draft=1,
        )
        optimizer = torch.optim.Adam(
            fine_tuned_model.parameters(),
            lr=args.lr,
            betas=(0.95, 0.999),
        )

        for iteration in range(args.num_iterations):
            optimizer.zero_grad()
            loss = draft_loss.compute_loss(
                fine_tuned_model, args.batch_size, device
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                fine_tuned_model.parameters(), max_norm=1.0
            )
            optimizer.step()
            if iteration % 50 == 0:
                print(f"Iter {iteration}: loss = {loss.item():.4f}")

    # Generate samples after fine-tuning
    print("\nGenerating samples with σ=0 (ODE sampler)...")
    with torch.no_grad():
        samples = trainer.generate(
            batch_size=16, device=device, sigma_sampling=0.0
        )
    print(f"Generated {samples.shape[0]} samples of shape {samples.shape[1:]}")

    # Compute metrics
    print("\nComputing evaluation metrics...")
    eval_metrics = EvaluationMetrics(device=str(device))
    prompts = ["test prompt"] * 16
    metrics = eval_metrics.compute_all_metrics(samples, prompts)
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    # Save fine-tuned model
    torch.save(fine_tuned_model.state_dict(), "fine_tuned_model.pt")
    print("\nModel saved to fine_tuned_model.pt")


if __name__ == "__main__":
    main()

"""
Main training script for Adjoint Matching fine-tuning.

Usage:
    python scripts/train_adjoint_matching.py --config configs/adjoint_matching.yaml

This script implements the full training loop from Algorithm 1 in the paper.
"""

import argparse
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import MLPVelocityModel, ConditionalMLPVelocityModel
from src.adjoint_matching import (
    AdjointMatchingTrainer,
    compute_lean_adjoint,
    adjoint_matching_loss_fm,
    select_gradient_timesteps,
)
from src.noise_schedules import get_sigma_memoryless_fm
from src.baselines import draft_loss, refl_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Adjoint Matching Fine-tuning")
    parser.add_argument("--config", type=str, default="configs/adjoint_matching.yaml")
    parser.add_argument("--method", type=str, default="adjoint_matching",
                        choices=["adjoint_matching", "draft_1", "draft_40", "refl", "cont_adjoint", "disc_adjoint"])
    parser.add_argument("--lambda_reward", type=float, default=12500.0)
    parser.add_argument("--num_steps", type=int, default=40)
    parser.add_argument("--num_iterations", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def create_model(config: dict, device: torch.device) -> nn.Module:
    """Create velocity model based on config."""
    model_type = config.get("model_type", "mlp")
    
    if model_type == "mlp":
        model = MLPVelocityModel(
            data_dim=config.get("data_dim", 2),
            hidden_dim=config.get("hidden_dim", 256),
            num_layers=config.get("num_layers", 4),
        )
    elif model_type == "conditional_mlp":
        model = ConditionalMLPVelocityModel(
            data_dim=config.get("data_dim", 2),
            condition_dim=config.get("condition_dim", 64),
            hidden_dim=config.get("hidden_dim", 256),
            num_layers=config.get("num_layers", 4),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)


def create_reward_fn(config: dict, device: torch.device):
    """Create reward function based on config."""
    reward_type = config.get("reward_type", "quadratic")
    
    if reward_type == "quadratic":
        target = torch.tensor(config.get("target", [0.0, 0.0]), device=device)
        def reward_fn(x):
            return -((x - target) ** 2).sum(dim=-1)
        return reward_fn
    elif reward_type == "norm":
        def reward_fn(x):
            return -x.norm(dim=-1)
        return reward_fn
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")


def train_adjoint_matching(
    finetune_model: nn.Module,
    base_model: nn.Module,
    reward_fn,
    optimizer: optim.Optimizer,
    num_iterations: int,
    batch_size: int,
    num_steps: int,
    lambda_reward: float,
    data_dim: int,
    device: torch.device,
    output_dir: str,
):
    """Main training loop for Adjoint Matching."""
    h = 1.0 / num_steps
    lct = 1.6 * (lambda_reward ** 2)
    
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_history = []
    
    for iteration in range(num_iterations):
        finetune_model.train()
        optimizer.zero_grad()
        
        # Sample initial noise
        x0 = torch.randn(batch_size, data_dim, device=device)
        
        # Step 1: Sample trajectory with memoryless noise schedule
        states = [x0.detach()]
        x = x0.clone()
        
        with torch.no_grad():
            for k in range(num_steps):
                t = k * h
                t_tensor = torch.full((batch_size,), t, device=device)
                sigma_t = get_sigma_memoryless_fm(torch.tensor(t, device=device), h=h).item()
                
                v = finetune_model(x, t_tensor)
                kappa_t = 1.0 / (t + h)
                drift = 2.0 * v - kappa_t * x
                noise = torch.randn_like(x)
                x = x + h * drift + (h ** 0.5) * sigma_t * noise
                states.append(x.detach())
        
        # Step 2: Compute lean adjoint
        scaled_reward_fn = lambda x: lambda_reward * reward_fn(x)
        adjoint_states = compute_lean_adjoint(
            states=states,
            base_velocity_fn=base_model,
            reward_fn=scaled_reward_fn,
            num_steps=num_steps,
            use_noiseless_final=True,
        )
        
        # Step 3: Select gradient timesteps
        grad_timesteps = select_gradient_timesteps(
            num_steps=num_steps,
            num_early=10,
            num_late=10,
        )
        
        # Step 4: Compute Adjoint Matching loss
        loss = adjoint_matching_loss_fm(
            finetune_velocity_fn=finetune_model,
            base_velocity_fn=base_model,
            states=states,
            adjoint_states=adjoint_states,
            num_steps=num_steps,
            lct=lct,
            gradient_timesteps=grad_timesteps,
        )
        
        # Step 5: Update
        loss.backward()
        torch.nn.utils.clip_grad_norm_(finetune_model.parameters(), 1.0)
        optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            x1 = states[-1]
            reward = reward_fn(x1).mean().item()
        
        metrics = {
            "iteration": iteration,
            "loss": loss.item(),
            "reward": reward,
        }
        metrics_history.append(metrics)
        
        if (iteration + 1) % 100 == 0:
            print(f"Iter {iteration+1}/{num_iterations}: loss={loss.item():.4f}, reward={reward:.4f}")
        
        # Save checkpoint
        if (iteration + 1) % 500 == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_{iteration+1}.pt")
            torch.save({
                "iteration": iteration,
                "model_state_dict": finetune_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    return metrics_history


def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Override config with command line args
    config.update({
        "method": args.method,
        "lambda_reward": args.lambda_reward,
        "num_steps": args.num_steps,
        "num_iterations": args.num_iterations,
        "batch_size": args.batch_size,
        "lr": args.lr,
    })
    
    # Set seed
    torch.manual_seed(args.seed)
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create models
    data_dim = config.get("data_dim", 2)
    base_model = create_model(config, device)
    finetune_model = create_model(config, device)
    
    # Initialize finetune model from base (in practice, load pre-trained weights)
    # For demo, we use random initialization
    finetune_model.load_state_dict(base_model.state_dict())
    
    # Freeze base model
    for param in base_model.parameters():
        param.requires_grad_(False)
    base_model.eval()
    
    # Create reward function
    reward_fn = create_reward_fn(config, device)
    
    # Create optimizer
    optimizer = optim.Adam(
        finetune_model.parameters(),
        lr=args.lr,
        betas=(0.95, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
    )
    
    print(f"Training with method: {args.method}")
    print(f"Lambda reward: {args.lambda_reward}")
    print(f"Num steps: {args.num_steps}")
    print(f"Num iterations: {args.num_iterations}")
    
    if args.method == "adjoint_matching":
        metrics = train_adjoint_matching(
            finetune_model=finetune_model,
            base_model=base_model,
            reward_fn=reward_fn,
            optimizer=optimizer,
            num_iterations=args.num_iterations,
            batch_size=args.batch_size,
            num_steps=args.num_steps,
            lambda_reward=args.lambda_reward,
            data_dim=data_dim,
            device=device,
            output_dir=args.output_dir,
        )
    else:
        # Other methods
        print(f"Method {args.method} training loop not fully implemented in this demo.")
        print("See src/baselines.py for the loss functions.")
    
    print("Training complete!")


if __name__ == "__main__":
    main()

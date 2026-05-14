"""
Toy 1D experiment to demonstrate the memoryless noise schedule.

Reproduces Figure 2 from the paper:
"Visualization of Theorem 1 showing that fine-tuning must be done with the 
memoryless noise schedule to ensure convergence to the tilted distribution."

Setup:
- Base model: Flow Matching on a 1D Gaussian mixture
- Reward: r(x) = -||x - target||^2 (encourages samples near target)
- Compare: constant sigma vs memoryless sigma
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple, List
from .models import MLPVelocityModel
from .adjoint_matching import AdjointMatchingTrainer, compute_lean_adjoint, adjoint_matching_loss_fm, select_gradient_timesteps
from .noise_schedules import get_sigma_memoryless_fm
from .baselines import draft_loss


def create_gaussian_mixture_data(
    n_samples: int,
    means: List[float] = [-2.0, 2.0],
    stds: List[float] = [0.5, 0.5],
    weights: List[float] = [0.5, 0.5],
) -> torch.Tensor:
    """Create samples from a 1D Gaussian mixture."""
    samples = []
    n_components = len(means)
    
    for i in range(n_samples):
        # Sample component
        comp = np.random.choice(n_components, p=weights)
        x = np.random.normal(means[comp], stds[comp])
        samples.append(x)
    
    return torch.tensor(samples, dtype=torch.float32).unsqueeze(-1)


def train_base_flow_matching(
    data: torch.Tensor,
    model: nn.Module,
    num_epochs: int = 1000,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> nn.Module:
    """
    Pre-train a Flow Matching model on data.
    
    Flow Matching objective:
    L = E[||v(X_t, t) - (X_1 - X_0)||^2]
    where X_t = t*X_1 + (1-t)*X_0, X_0 ~ N(0,1), X_1 ~ p_data
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    n_data = data.shape[0]
    
    for epoch in range(num_epochs):
        # Sample batch
        idx = torch.randperm(n_data)[:batch_size]
        x1 = data[idx]
        
        # Sample noise and time
        x0 = torch.randn_like(x1)
        t = torch.rand(batch_size, 1)
        
        # Interpolate
        x_t = t * x1 + (1 - t) * x0
        
        # Target velocity: x1 - x0
        target_v = x1 - x0
        
        # Predict velocity
        t_flat = t.squeeze(-1)
        v_pred = model(x_t, t_flat)
        
        # Flow Matching loss
        loss = ((v_pred - target_v) ** 2).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    
    return model


def quadratic_reward(x: torch.Tensor, target: float = 3.0) -> torch.Tensor:
    """
    Quadratic reward: r(x) = -(x - target)^2
    Encourages samples near target.
    """
    return -((x - target) ** 2).sum(dim=-1)


def run_toy_experiment(
    num_steps: int = 40,
    lambda_reward: float = 1.0,
    num_finetune_steps: int = 500,
    device: str = "cpu",
) -> dict:
    """
    Run the 1D toy experiment comparing different noise schedules.
    
    Returns:
        Dictionary with samples from each method
    """
    device = torch.device(device)
    
    # Create data
    print("Creating data...")
    data = create_gaussian_mixture_data(10000, means=[-2.0, 2.0])
    data = data.to(device)
    
    # Train base model
    print("Training base Flow Matching model...")
    base_model = MLPVelocityModel(data_dim=1, hidden_dim=128, num_layers=4)
    base_model = base_model.to(device)
    base_model = train_base_flow_matching(data, base_model, num_epochs=2000)
    
    # Define reward
    target = 3.0
    reward_fn = lambda x: lambda_reward * quadratic_reward(x, target)
    
    # Generate base samples
    print("Generating base samples...")
    base_model.eval()
    with torch.no_grad():
        x0 = torch.randn(1000, 1, device=device)
        x = x0.clone()
        h = 1.0 / num_steps
        for k in range(num_steps):
            t = k * h
            t_tensor = torch.full((1000,), t, device=device)
            v = base_model(x, t_tensor)
            x = x + h * v
        base_samples = x.cpu().numpy()
    
    results = {"base": base_samples}
    
    # Fine-tune with memoryless noise schedule (Adjoint Matching)
    print("Fine-tuning with Adjoint Matching (memoryless schedule)...")
    ft_model_am = MLPVelocityModel(data_dim=1, hidden_dim=128, num_layers=4)
    ft_model_am.load_state_dict(base_model.state_dict())
    ft_model_am = ft_model_am.to(device)
    
    optimizer_am = optim.Adam(ft_model_am.parameters(), lr=1e-4)
    
    for step in range(num_finetune_steps):
        ft_model_am.train()
        optimizer_am.zero_grad()
        
        x0 = torch.randn(64, 1, device=device)
        
        # Sample trajectory with memoryless schedule
        states = [x0.detach()]
        x = x0.clone()
        
        with torch.no_grad():
            for k in range(num_steps):
                t = k * h
                t_tensor = torch.full((64,), t, device=device)
                sigma_t = get_sigma_memoryless_fm(torch.tensor(t, device=device), h=h).item()
                v = ft_model_am(x, t_tensor)
                kappa_t = 1.0 / (t + h)
                drift = 2.0 * v - kappa_t * x
                noise = torch.randn_like(x)
                x = x + h * drift + (h ** 0.5) * sigma_t * noise
                states.append(x.detach())
        
        # Compute lean adjoint
        adjoint_states = compute_lean_adjoint(
            states=states,
            base_velocity_fn=base_model,
            reward_fn=reward_fn,
            num_steps=num_steps,
            use_noiseless_final=True,
        )
        
        # Select gradient timesteps
        grad_timesteps = select_gradient_timesteps(num_steps, num_early=10, num_late=10)
        
        # Compute loss
        lct = 1.6 * (lambda_reward ** 2)
        loss = adjoint_matching_loss_fm(
            finetune_velocity_fn=ft_model_am,
            base_velocity_fn=base_model,
            states=states,
            adjoint_states=adjoint_states,
            num_steps=num_steps,
            lct=lct,
            gradient_timesteps=grad_timesteps,
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ft_model_am.parameters(), 1.0)
        optimizer_am.step()
        
        if (step + 1) % 100 == 0:
            with torch.no_grad():
                r = reward_fn(states[-1]).mean().item()
            print(f"AM Step {step+1}/{num_finetune_steps}, Loss: {loss.item():.4f}, Reward: {r:.4f}")
    
    # Generate samples from fine-tuned model (ODE)
    ft_model_am.eval()
    with torch.no_grad():
        x0 = torch.randn(1000, 1, device=device)
        x = x0.clone()
        for k in range(num_steps):
            t = k * h
            t_tensor = torch.full((1000,), t, device=device)
            v = ft_model_am(x, t_tensor)
            x = x + h * v
        results["adjoint_matching_ode"] = x.cpu().numpy()
    
    # Fine-tune with constant sigma (to show bias)
    print("Fine-tuning with constant sigma (biased)...")
    ft_model_const = MLPVelocityModel(data_dim=1, hidden_dim=128, num_layers=4)
    ft_model_const.load_state_dict(base_model.state_dict())
    ft_model_const = ft_model_const.to(device)
    
    optimizer_const = optim.Adam(ft_model_const.parameters(), lr=1e-4)
    sigma_const = 1.0
    
    for step in range(num_finetune_steps):
        ft_model_const.train()
        optimizer_const.zero_grad()
        
        x0 = torch.randn(64, 1, device=device)
        
        # Sample trajectory with constant sigma
        states_const = [x0.detach()]
        x = x0.clone()
        
        with torch.no_grad():
            for k in range(num_steps):
                t = k * h
                t_tensor = torch.full((64,), t, device=device)
                v = ft_model_const(x, t_tensor)
                noise = torch.randn_like(x)
                x = x + h * v + (h ** 0.5) * sigma_const * noise
                states_const.append(x.detach())
        
        # Compute lean adjoint with constant sigma base drift
        # For constant sigma, base drift is just v_base
        adjoint_states_const = [None] * (num_steps + 1)
        
        x_last = states_const[-2].detach()
        t_last = torch.full((64,), 1.0 - h, device=device)
        with torch.no_grad():
            v_last = base_model(x_last, t_last)
        x_hat_1 = x_last + h * v_last
        
        x_hat_1_req = x_hat_1.detach().requires_grad_(True)
        r = reward_fn(x_hat_1_req)
        r_grad = torch.autograd.grad(r.sum(), x_hat_1_req)[0]
        a = -r_grad.detach()
        adjoint_states_const[num_steps] = a
        
        for k in range(num_steps - 1, -1, -1):
            t = k * h
            t_tensor = torch.full((64,), t, device=device)
            x_k = states_const[k].detach()
            x_k_req = x_k.requires_grad_(True)
            
            with torch.enable_grad():
                v_base = base_model(x_k_req, t_tensor)
                vjp = torch.autograd.grad(v_base, x_k_req, grad_outputs=a, create_graph=False)[0]
            
            a = (a + h * vjp).detach()
            adjoint_states_const[k] = a
        
        # Compute loss with constant sigma
        total_loss = torch.tensor(0.0, device=device)
        for k in range(num_steps):
            t = k * h
            t_tensor = torch.full((64,), t, device=device)
            x_k = states_const[k].detach()
            a_k = adjoint_states_const[k].detach()
            
            v_ft = ft_model_const(x_k, t_tensor)
            with torch.no_grad():
                v_base = base_model(x_k, t_tensor)
            
            control = (1.0 / sigma_const) * (v_ft - v_base)
            target = sigma_const * a_k
            residual = control + target
            loss_k = (residual ** 2).sum(dim=-1)
            total_loss = total_loss + loss_k.mean()
        
        loss = total_loss / num_steps
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ft_model_const.parameters(), 1.0)
        optimizer_const.step()
        
        if (step + 1) % 100 == 0:
            with torch.no_grad():
                r = reward_fn(states_const[-1]).mean().item()
            print(f"Const Step {step+1}/{num_finetune_steps}, Loss: {loss.item():.4f}, Reward: {r:.4f}")
    
    # Generate samples from constant sigma model
    ft_model_const.eval()
    with torch.no_grad():
        x0 = torch.randn(1000, 1, device=device)
        x = x0.clone()
        for k in range(num_steps):
            t = k * h
            t_tensor = torch.full((1000,), t, device=device)
            v = ft_model_const(x, t_tensor)
            x = x + h * v
        results["constant_sigma_ode"] = x.cpu().numpy()
    
    # Compute tilted distribution (ground truth)
    # p*(x) ∝ p_base(x) * exp(r(x))
    # For Gaussian mixture base and quadratic reward, we can compute this analytically
    x_grid = np.linspace(-5, 6, 1000)
    
    # Base density (Gaussian mixture)
    p_base = 0.5 * np.exp(-0.5 * ((x_grid + 2) / 0.5) ** 2) / (0.5 * np.sqrt(2 * np.pi))
    p_base += 0.5 * np.exp(-0.5 * ((x_grid - 2) / 0.5) ** 2) / (0.5 * np.sqrt(2 * np.pi))
    
    # Reward
    r_grid = lambda_reward * (-(x_grid - target) ** 2)
    
    # Tilted distribution
    p_tilted = p_base * np.exp(r_grid)
    p_tilted = p_tilted / (p_tilted.sum() * (x_grid[1] - x_grid[0]))
    
    results["x_grid"] = x_grid
    results["p_base"] = p_base / (p_base.sum() * (x_grid[1] - x_grid[0]))
    results["p_tilted"] = p_tilted
    
    return results


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("Running 1D toy experiment...")
    results = run_toy_experiment(
        num_steps=40,
        lambda_reward=2.0,
        num_finetune_steps=500,
    )
    
    # Plot results
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    x_grid = results["x_grid"]
    
    # (a) Base model
    axes[0].hist(results["base"].flatten(), bins=50, density=True, alpha=0.7, label="Samples")
    axes[0].plot(x_grid, results["p_base"], 'r-', label="True density")
    axes[0].set_title("(a) Base Flow Matching")
    axes[0].legend()
    
    # (b) Constant sigma (biased)
    axes[1].hist(results["constant_sigma_ode"].flatten(), bins=50, density=True, alpha=0.7, label="Samples")
    axes[1].plot(x_grid, results["p_tilted"], 'r-', label="Tilted distribution")
    axes[1].set_title("(b) Constant σ (biased)")
    axes[1].legend()
    
    # (c) Constant sigma (biased) - same as (b) but different view
    axes[2].hist(results["constant_sigma_ode"].flatten(), bins=50, density=True, alpha=0.7, label="Samples")
    axes[2].plot(x_grid, results["p_tilted"], 'r-', label="Tilted distribution")
    axes[2].set_title("(c) Constant σ (biased)")
    axes[2].legend()
    
    # (d) Memoryless schedule (correct)
    axes[3].hist(results["adjoint_matching_ode"].flatten(), bins=50, density=True, alpha=0.7, label="Samples")
    axes[3].plot(x_grid, results["p_tilted"], 'r-', label="Tilted distribution")
    axes[3].set_title("(d) Memoryless σ (correct)")
    axes[3].legend()
    
    plt.tight_layout()
    plt.savefig("toy_experiment_figure2.png", dpi=150, bbox_inches="tight")
    print("Saved figure to toy_experiment_figure2.png")

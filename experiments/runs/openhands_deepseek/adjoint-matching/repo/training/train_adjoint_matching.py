"""
Training loop for Adjoint Matching fine-tuning of Flow Matching models.

Algorithm 1 from the paper:
1. Sample trajectories with memoryless noise schedule
2. Solve lean adjoint ODE backwards
3. Compute Adjoint Matching objective
4. Update parameters using gradient descent
"""
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast
from typing import Optional, List, Dict, Any
import random
import numpy as np
from tqdm import tqdm

from ..soc.memoryless_schedule import MemorylessNoiseSchedule
from ..soc.adjoint_matching import (
    LeanAdjointSolver,
    AdjointMatchingLoss,
    stop_grad,
)
from ..models.flow_matching import FlowMatchingModel


def train_adjoint_matching(
    base_model: FlowMatchingModel,
    finetune_model: FlowMatchingModel,
    reward_model: nn.Module,  # ImageReward or similar
    dataloader: torch.utils.data.DataLoader,
    config: Dict[str, Any],
    num_epochs: int = 25,
    lambda_reward: float = 12500.0,
    num_steps: int = 40,
    lr: float = 2e-5,
    adam_betas: tuple = (0.95, 0.999),
    weight_decay: float = 1e-2,
    grad_clip: float = 1.0,
    precision: str = "bfloat16",
    lct_constant: float = 1.6,
    grad_timesteps_first: int = 10,
    grad_timesteps_last: int = 10,
    dt_offset: bool = True,
    device: str = "cuda",
    log_every: int = 50,
    eval_every: int = 200,
    save_every: int = 500,
    save_dir: str = "./checkpoints",
):
    """
    Train Flow Matching model with Adjoint Matching.
    
    Args:
        base_model: pre-trained Flow Matching model (frozen)
        finetune_model: model to be fine-tuned
        reward_model: reward model (e.g., ImageReward)
        dataloader: dataloader for text prompts
        config: full configuration dict
    """
    base_model.eval()
    base_model.to(device)
    for p in base_model.parameters():
        p.requires_grad = False
    
    finetune_model.train()
    finetune_model.to(device)
    
    reward_model.eval()
    reward_model.to(device)
    for p in reward_model.parameters():
        p.requires_grad = False
    
    dt = 1.0 / num_steps
    
    # Memoryless noise schedule
    schedule = MemorylessNoiseSchedule(
        alpha_fn=lambda t: t,
        beta_fn=lambda t: 1.0 - t,
        dt=dt if dt_offset else 0.0,
        use_offset=dt_offset,
    )
    
    # Lean adjoint solver
    adjoint_solver = LeanAdjointSolver()
    
    # Adjoint Matching loss
    lct = lct_constant * (lambda_reward ** 2) if lct_constant is not None else None
    loss_fn = AdjointMatchingLoss(
        base_model=base_model,
        schedule=schedule,
        model_type="flow_matching",
        lct=lct,
        lambda_reward=lambda_reward,
    )
    
    # Optimizer
    optimizer = Adam(
        finetune_model.parameters(),
        lr=lr,
        betas=adam_betas,
        eps=1e-8,
        weight_decay=weight_decay,
    )
    
    if precision == "bfloat16":
        scaler = GradScaler()
    else:
        scaler = None
    
    # Times for trajectory
    times = [i * dt for i in range(num_steps + 1)]
    
    global_step = 0
    
    for epoch in range(num_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0.0
        epoch_control_cost = 0.0
        epoch_reward = 0.0
        
        for batch_idx, batch in enumerate(pbar):
            context = batch["text_embeddings"].to(device)
            batch_size = context.shape[0]
            
            # Step 1: Sample trajectories with memoryless noise schedule
            with torch.no_grad():
                trajectory = sample_trajectory_memoryless(
                    base_model=base_model,
                    finetune_model=finetune_model,
                    context=context,
                    schedule=schedule,
                    num_steps=num_steps,
                    dt=dt,
                )
            
            # Step 2: Solve lean adjoint ODE backwards
            with torch.no_grad():
                a_tilde_list = adjoint_solver.solve_backward(
                    trajectory=trajectory,
                    times=times,
                    base_drift_fn=lambda x, t_val: compute_base_drift(
                        base_model, x, t_val, context, schedule
                    ),
                    reward_grad_fn=lambda x: compute_reward_scaled(
                        reward_model, x, lambda_reward
                    ),
                    dt=dt,
                )
            
            # Step 3: Select gradient evaluation timesteps
            timestep_indices = select_gradient_timesteps(
                num_steps, grad_timesteps_first, grad_timesteps_last
            )
            
            # Step 4: Compute Adjoint Matching objective
            optimizer.zero_grad()
            
            if precision == "bfloat16":
                with autocast():
                    loss = loss_fn(
                        finetune_model=finetune_model,
                        trajectory=trajectory,
                        a_tilde_list=a_tilde_list,
                        times=times,
                        context=context,
                        timestep_indices=timestep_indices,
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(finetune_model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = loss_fn(
                    finetune_model=finetune_model,
                    trajectory=trajectory,
                    a_tilde_list=a_tilde_list,
                    times=times,
                    context=context,
                    timestep_indices=timestep_indices,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(finetune_model.parameters(), grad_clip)
                optimizer.step()
            
            # Logging
            with torch.no_grad():
                X_final = trajectory[-1]
                reward_val = compute_reward_scaled(reward_model, X_final, lambda_reward).mean()
                
                # Compute control cost
                control_cost_total = 0.0
                for k in range(num_steps):
                    X_t = trajectory[k]
                    t_val = times[k]
                    t = torch.full((batch_size,), t_val, device=device)
                    scale = schedule.compute_control_scaling(t)
                    v_f = finetune_model(X_t, t * 1000, context)
                    v_b = base_model(X_t, t * 1000, context)
                    u = scale[:, None, None, None] * (v_f - v_b)
                    control_cost_total += 0.5 * (u ** 2).sum(dim=(1, 2, 3)).mean() * dt
            
            epoch_loss += loss.item()
            epoch_control_cost += control_cost_total.item()
            epoch_reward += reward_val.item()
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ctrl": f"{control_cost_total.item():.4f}",
                "reward": f"{reward_val.item():.4f}",
            })
            
            if global_step % log_every == 0 and global_step > 0:
                print(f"\nStep {global_step}: loss={epoch_loss/(batch_idx+1):.4f}, "
                      f"control={epoch_control_cost/(batch_idx+1):.4f}, "
                      f"reward={epoch_reward/(batch_idx+1):.4f}")
            
            if global_step % save_every == 0 and global_step > 0:
                torch.save({
                    "step": global_step,
                    "model_state_dict": finetune_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, f"{save_dir}/adjoint_matching_step_{global_step}.pt")
            
            global_step += 1
        
        avg_loss = epoch_loss / len(dataloader)
        avg_ctrl = epoch_control_cost / len(dataloader)
        avg_reward = epoch_reward / len(dataloader)
        print(f"Epoch {epoch + 1} completed: loss={avg_loss:.4f}, "
              f"control={avg_ctrl:.4f}, reward={avg_reward:.4f}")
    
    return finetune_model


def sample_trajectory_memoryless(
    base_model: FlowMatchingModel,
    finetune_model: FlowMatchingModel,
    context: torch.Tensor,
    schedule: MemorylessNoiseSchedule,
    num_steps: int = 40,
    dt: float = 0.025,
) -> List[torch.Tensor]:
    """
    Sample trajectory with memoryless noise schedule.
    
    X_{t+h} = X_t + h * (2*v_finetune(X_t, t) - kappa_t * X_t) + sqrt(h) * sigma(t) * eps
    """
    batch_size = context.shape[0]
    device = context.device
    
    x0 = torch.randn(batch_size, base_model.unet.in_channels,
                    base_model.unet.in_channels,
                    base_model.unet.in_channels, device=device)
    
    trajectory = [x0]
    x = x0
    
    for i in range(num_steps):
        t_val = i * dt
        t = torch.full((batch_size,), t_val, device=device)
        
        sigma = schedule.sigma(t)
        kappa = schedule.kappa_t(t)
        
        v_finetune = finetune_model(x, t * 1000, context)
        
        drift = 2.0 * v_finetune - kappa * x
        
        noise = torch.randn_like(x)
        x_next = x + dt * drift + (dt ** 0.5) * sigma[:, None, None, None] * noise
        
        trajectory.append(x_next.clone())
        x = x_next
    
    return trajectory


def compute_base_drift(
    base_model: FlowMatchingModel,
    x: torch.Tensor,
    t_val: float,
    context: torch.Tensor,
    schedule: MemorylessNoiseSchedule,
) -> torch.Tensor:
    """
    Compute base drift b(x,t) for Flow Matching.
    
    b(x,t) = 2 * v_base(x,t) - kappa_t * x
    """
    batch_size = x.shape[0]
    device = x.device
    t = torch.full((batch_size,), t_val, device=device)
    v_base = base_model(x, t * 1000, context)
    kappa = schedule.kappa_t(t)
    return 2.0 * v_base - kappa * x


def compute_reward_scaled(
    reward_model: nn.Module,
    x: torch.Tensor,
    lambda_reward: float,
) -> torch.Tensor:
    """
    Compute scaled reward: r(x) = lambda * RewardModel(x).
    
    x is a latent representation; must be decoded before passing to reward model.
    For ImageReward, we need actual images.
    
    This is a placeholder - actual implementation needs a VAE decoder.
    """
    return lambda_reward * reward_model(x)


def select_gradient_timesteps(
    num_steps: int,
    first_n: int = 10,
    last_n: int = 10,
) -> List[int]:
    """
    Select subset of timesteps for gradient computation.
    
    Samples first_n uniformly from first 75% of timesteps,
    and always includes the last last_n timesteps.
    """
    cutoff = int(num_steps * 0.75)  # first 75%
    first_subset = random.sample(range(cutoff), min(first_n, cutoff))
    last_subset = list(range(num_steps - last_n, num_steps))
    
    all_indices = sorted(set(first_subset + last_subset))
    return all_indices

"""
Training loops for baseline fine-tuning methods.

Baselines:
- DRaFT-K (Clark et al., 2024): Direct reward fine-tuning through K steps
- ReFL (Xu et al., 2023): Reward Feedback Learning
- DPO (Wallace et al., 2023a): Direct Preference Optimization
- Continuous Adjoint: Differentiate-then-discretize adjoint method
- Discrete Adjoint: Discretize-then-differentiate adjoint method
"""
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast
from typing import Optional, List, Dict, Any
import random
from tqdm import tqdm

from ..soc.memoryless_schedule import MemorylessNoiseSchedule
from ..soc.adjoint_matching import (
    LeanAdjointSolver,
    FullAdjointSolver,
    AdjointMatchingLoss,
    ContinuousAdjointLoss,
    stop_grad,
)
from ..models.flow_matching import FlowMatchingModel


def train_draft(
    base_model: FlowMatchingModel,
    finetune_model: FlowMatchingModel,
    reward_model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    config: Dict[str, Any],
    num_epochs: int = 25,
    K: int = 1,  # DRaFT-K: number of steps to backprop through
    num_steps: int = 40,
    lr: float = 2e-5,
    adam_betas: tuple = (0.95, 0.999),
    weight_decay: float = 1e-2,
    grad_clip: float = 1.0,
    precision: str = "bfloat16",
    device: str = "cuda",
    log_every: int = 50,
    eval_every: int = 200,
    save_every: int = 500,
    save_dir: str = "./checkpoints",
):
    """
    DRaFT-K: Directly backpropagate reward through K sampling steps.
    
    For DRaFT-1, only backprop through last step.
    For DRaFT-40 (full), backprop through all steps.
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
    schedule = MemorylessNoiseSchedule(
        use_offset=True, dt=dt
    )
    
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
    
    global_step = 0
    
    for epoch in range(num_epochs):
        pbar = tqdm(dataloader, desc=f"DRaFT-{K} Epoch {epoch + 1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            context = batch["text_embeddings"].to(device)
            batch_size = context.shape[0]
            
            # Sample forward K steps with gradient tracking
            x0 = torch.randn(batch_size, finetune_model.unet.in_channels,
                           finetune_model.unet.in_channels,
                           finetune_model.unet.in_channels, device=device)
            
            # First N-K steps no gradient
            x = x0
            start_grad_step = max(0, num_steps - K)
            
            with torch.no_grad():
                for i in range(start_grad_step):
                    t_val = i * dt
                    t = torch.full((batch_size,), t_val, device=device)
                    sigma = schedule.sigma(t)
                    kappa = schedule.kappa_t(t)
                    v_f = finetune_model(x, t * 1000, context)
                    drift = 2.0 * v_f - kappa * x
                    noise = torch.randn_like(x)
                    x = x + dt * drift + (dt ** 0.5) * sigma[:, None, None, None] * noise
            
            # Last K steps with gradient
            for i in range(start_grad_step, num_steps):
                t_val = i * dt
                t = torch.full((batch_size,), t_val, device=device)
                sigma = schedule.sigma(t)
                kappa = schedule.kappa_t(t)
                v_f = finetune_model(x, t * 1000, context)
                drift = 2.0 * v_f - kappa * x
                noise = torch.randn_like(x)
                x = x + dt * drift + (dt ** 0.5) * sigma[:, None, None, None] * noise
            
            # Compute reward and loss (negative reward maximization)
            optimizer.zero_grad()
            
            if precision == "bfloat16":
                with autocast():
                    reward = reward_model(x)
                    loss = -reward.mean()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(finetune_model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                reward = reward_model(x)
                loss = -reward.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(finetune_model.parameters(), grad_clip)
                optimizer.step()
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "reward": f"{reward.mean().item():.4f}",
            })
            
            if global_step % save_every == 0 and global_step > 0:
                torch.save({
                    "step": global_step,
                    "model_state_dict": finetune_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, f"{save_dir}/draft{K}_step_{global_step}.pt")
            
            global_step += 1
    
    return finetune_model


def train_refl(
    base_model: FlowMatchingModel,
    finetune_model: FlowMatchingModel,
    reward_model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    config: Dict[str, Any],
    num_epochs: int = 25,
    num_steps: int = 40,
    lr: float = 2e-5,
    adam_betas: tuple = (0.95, 0.999),
    weight_decay: float = 1e-2,
    grad_clip: float = 1.0,
    precision: str = "bfloat16",
    device: str = "cuda",
    log_every: int = 50,
    save_every: int = 500,
    save_dir: str = "./checkpoints",
):
    """
    ReFL (Reward Feedback Learning, Xu et al. 2023).
    
    Uses denoiser map: hat{X}_1(X_t) = (v(X_t, t) - (beta_dot/beta)*X_t) / (alpha_dot - (beta_dot/beta)*alpha)
    Then maximizes reward on predicted clean sample.
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
    
    global_step = 0
    
    for epoch in range(num_epochs):
        pbar = tqdm(dataloader, desc=f"ReFL Epoch {epoch + 1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            context = batch["text_embeddings"].to(device)
            batch_size = context.shape[0]
            
            # Sample trajectory without gradient
            x0 = torch.randn(batch_size, finetune_model.unet.in_channels,
                           finetune_model.unet.in_channels,
                           finetune_model.unet.in_channels, device=device)
            
            x = x0
            # Sample random timestep for denoising
            k = random.randint(0, num_steps - 1)
            t_val = k * dt
            
            with torch.no_grad():
                schedule = MemorylessNoiseSchedule(use_offset=True, dt=dt)
                for i in range(k + 1):
                    t_i = torch.full((batch_size,), i * dt, device=device)
                    sigma = schedule.sigma(t_i)
                    kappa = schedule.kappa_t(t_i)
                    v_f = finetune_model(x, t_i * 1000, context)
                    drift = 2.0 * v_f - kappa * x
                    noise = torch.randn_like(x)
                    x = x + dt * drift + (dt ** 0.5) * sigma[:, None, None, None] * noise
            
            # Denoise at chosen timestep
            t = torch.full((batch_size,), t_val, device=device)
            v_f = finetune_model(x, t * 1000, context)
            
            # Flow Matching denoiser:
            # hat{X}_1 = (v - (beta_dot/beta)*x) / (alpha_dot - (beta_dot/beta)*alpha)
            alpha = t_val
            beta = 1.0 - t_val
            alpha_dot = 1.0  # derivative of t
            beta_dot = -1.0  # derivative of 1-t
            beta_ratio = beta_dot / (beta + 1e-8)
            x_denoised = (v_f - beta_ratio * x) / (alpha_dot - beta_ratio * alpha + 1e-8)
            
            optimizer.zero_grad()
            
            if precision == "bfloat16":
                with autocast():
                    reward = reward_model(x_denoised)
                    loss = -reward.mean()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(finetune_model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                reward = reward_model(x_denoised)
                loss = -reward.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(finetune_model.parameters(), grad_clip)
                optimizer.step()
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "reward": f"{reward.mean().item():.4f}"})
            
            if global_step % save_every == 0 and global_step > 0:
                torch.save({
                    "step": global_step,
                    "model_state_dict": finetune_model.state_dict(),
                }, f"{save_dir}/refl_step_{global_step}.pt")
            
            global_step += 1
    
    return finetune_model


def train_dpo(
    base_model: FlowMatchingModel,
    finetune_model: FlowMatchingModel,
    reward_model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    config: Dict[str, Any],
    num_epochs: int = 25,
    num_steps: int = 40,
    beta_dpo: float = 5000.0,  # KL penalty coefficient
    lr: float = 2e-5,
    adam_betas: tuple = (0.95, 0.999),
    weight_decay: float = 1e-2,
    grad_clip: float = 1.0,
    precision: str = "bfloat16",
    device: str = "cuda",
    log_every: int = 50,
    save_every: int = 500,
    save_dir: str = "./checkpoints",
):
    """
    DPO (Direct Preference Optimization) adapted for Flow Matching.
    
    Uses pairs of samples with reward-based preference weights.
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
    
    global_step = 0
    sigmoid = nn.Sigmoid()
    
    for epoch in range(num_epochs):
        pbar = tqdm(dataloader, desc=f"DPO Epoch {epoch + 1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            context = batch["text_embeddings"].to(device)
            batch_size = context.shape[0]
            
            # Sample two trajectories (a and b) from current model
            with torch.no_grad():
                schedule = MemorylessNoiseSchedule(use_offset=True, dt=dt)
                x0_a = torch.randn_like(torch.randn(batch_size, finetune_model.unet.in_channels,
                                                     finetune_model.unet.in_channels,
                                                     finetune_model.unet.in_channels, device=device))
                x0_b = torch.randn_like(x0_a)
                
                x_a = x0_a
                x_b = x0_b
                for i in range(num_steps):
                    t_i = torch.full((batch_size,), i * dt, device=device)
                    sigma = schedule.sigma(t_i)
                    kappa = schedule.kappa_t(t_i)
                    v_a = finetune_model(x_a, t_i * 1000, context)
                    v_b = finetune_model(x_b, t_i * 1000, context)
                    drift_a = 2.0 * v_a - kappa * x_a
                    drift_b = 2.0 * v_b - kappa * x_b
                    noise_a = torch.randn_like(x_a)
                    noise_b = torch.randn_like(x_b)
                    x_a = x_a + dt * drift_a + (dt ** 0.5) * sigma[:, None, None, None] * noise_a
                    x_b = x_b + dt * drift_b + (dt ** 0.5) * sigma[:, None, None, None] * noise_b
            
            # Compute rewards for preference weights
            with torch.no_grad():
                r_a = reward_model(x_a)
                r_b = reward_model(x_b)
                # Preference probability: P(a > b) = sigmoid(r_a - r_b)
                pref_weight_a = sigmoid(r_a - r_b)
                pref_weight_b = sigmoid(r_b - r_a)
            
            # Sample random timestep for DPO
            k = random.randint(0, num_steps - 1)
            t_val = k * dt
            t = torch.full((batch_size,), t_val, device=device)
            
            # Compute intermediate state for the chosen timestep
            # Re-sample the state at time t for a and b
            with torch.no_grad():
                x_a_t = x0_a
                x_b_t = x0_b
                for i in range(k + 1):
                    t_i = torch.full((batch_size,), i * dt, device=device)
                    sigma = schedule.sigma(t_i)
                    kappa = schedule.kappa_t(t_i)
                    v_a = finetune_model(x_a_t, t_i * 1000, context)
                    v_b = finetune_model(x_b_t, t_i * 1000, context)
                    drift_a = 2.0 * v_a - kappa * x_a_t
                    drift_b = 2.0 * v_b - kappa * x_b_t
                    noise_a = torch.randn_like(x_a_t)
                    noise_b = torch.randn_like(x_b_t)
                    x_a_t = x_a_t + dt * drift_a + (dt ** 0.5) * sigma[:, None, None, None] * noise_a
                    x_b_t = x_b_t + dt * drift_b + (dt ** 0.5) * sigma[:, None, None, None] * noise_b
            
            # DPO loss for Flow Matching
            alpha = t_val
            beta = 1.0 - t_val
            alpha_dot = 1.0
            beta_dot = -1.0
            beta_ratio = beta_dot / (beta + 1e-8)
            
            v_a_f = finetune_model(x_a_t.detach(), t * 1000, context)
            v_b_f = finetune_model(x_b_t.detach(), t * 1000, context)
            v_a_base = base_model(x_a_t.detach(), t * 1000, context)
            v_b_base = base_model(x_b_t.detach(), t * 1000, context)
            
            # Distance to final sample
            dist_a_f = ((v_a_f - beta_ratio * x_a_t.detach()) / (alpha_dot - beta_ratio * alpha + 1e-8) - x_a).pow(2).sum(dim=(1, 2, 3))
            dist_a_base = ((v_a_base - beta_ratio * x_a_t.detach()) / (alpha_dot - beta_ratio * alpha + 1e-8) - x_a).pow(2).sum(dim=(1, 2, 3))
            dist_b_f = ((v_b_f - beta_ratio * x_b_t.detach()) / (alpha_dot - beta_ratio * alpha + 1e-8) - x_b).pow(2).sum(dim=(1, 2, 3))
            dist_b_base = ((v_b_base - beta_ratio * x_b_t.detach()) / (alpha_dot - beta_ratio * alpha + 1e-8) - x_b).pow(2).sum(dim=(1, 2, 3))
            
            # DPO objective
            log_ratio_a = -beta_dpo / 2.0 * (dist_a_f - dist_a_base + dist_b_f - dist_b_base)
            log_ratio_b = -beta_dpo / 2.0 * (dist_b_f - dist_b_base + dist_a_f - dist_a_base)
            
            loss = -(
                pref_weight_a * torch.log(sigmoid(log_ratio_a) + 1e-8) +
                pref_weight_b * torch.log(sigmoid(log_ratio_b) + 1e-8)
            ).mean()
            
            optimizer.zero_grad()
            
            if precision == "bfloat16":
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(finetune_model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(finetune_model.parameters(), grad_clip)
                optimizer.step()
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            if global_step % save_every == 0 and global_step > 0:
                torch.save({
                    "step": global_step,
                    "model_state_dict": finetune_model.state_dict(),
                }, f"{save_dir}/dpo_step_{global_step}.pt")
            
            global_step += 1
    
    return finetune_model


def train_continuous_adjoint(
    base_model: FlowMatchingModel,
    finetune_model: FlowMatchingModel,
    reward_model: nn.Module,
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
    lct_constant: float = 1600.0,
    dt_offset: bool = True,
    device: str = "cuda",
    log_every: int = 50,
    save_every: int = 500,
    save_dir: str = "./checkpoints",
):
    """
    Continuous Adjoint method.
    
    Uses full adjoint ODE (differentiate-then-discretize).
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
    schedule = MemorylessNoiseSchedule(use_offset=dt_offset, dt=dt)
    
    adjoint_solver = FullAdjointSolver()
    loss_fn = ContinuousAdjointLoss(
        base_model=base_model,
        schedule=schedule,
        model_type="flow_matching",
        lct=lct_constant * (lambda_reward ** 2) if lct_constant else None,
        lambda_reward=lambda_reward,
    )
    
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
    
    times = [i * dt for i in range(num_steps + 1)]
    global_step = 0
    
    for epoch in range(num_epochs):
        pbar = tqdm(dataloader, desc=f"ContAdjoint Epoch {epoch + 1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            context = batch["text_embeddings"].to(device)
            batch_size = context.shape[0]
            
            with torch.no_grad():
                trajectory = sample_trajectory_memoryless_static(
                    base_model, finetune_model, context, schedule, num_steps, dt
                )
            
            with torch.no_grad():
                a_list = adjoint_solver.solve_backward(
                    trajectory=trajectory,
                    times=times,
                    base_drift_fn=lambda x, t_val: (2.0 * base_model(
                        x, torch.full((batch_size,), t_val, device=device) * 1000, context
                    ) - schedule.kappa_t(torch.full((batch_size,), t_val, device=device)) * x),
                    control_fn=lambda x, t_val: compute_control(
                        base_model, finetune_model, x, t_val, context, schedule
                    ),
                    sigma_fn=lambda t_val: schedule.sigma(
                        torch.full((batch_size,), t_val, device=device)
                    ),
                    reward_grad_fn=lambda x: lambda_reward * reward_model(x),
                    dt=dt,
                )
            
            timestep_indices = select_gradient_timesteps(
                num_steps, 10, 10
            )
            
            optimizer.zero_grad()
            
            if precision == "bfloat16":
                with autocast():
                    loss = loss_fn.compute_gradient(
                        finetune_model=finetune_model,
                        trajectory=trajectory,
                        a_list=a_list,
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
                loss = loss_fn.compute_gradient(
                    finetune_model=finetune_model,
                    trajectory=trajectory,
                    a_list=a_list,
                    times=times,
                    context=context,
                    timestep_indices=timestep_indices,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(finetune_model.parameters(), grad_clip)
                optimizer.step()
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            if global_step % save_every == 0 and global_step > 0:
                torch.save({
                    "step": global_step,
                    "model_state_dict": finetune_model.state_dict(),
                }, f"{save_dir}/cont_adj_step_{global_step}.pt")
            
            global_step += 1
    
    return finetune_model


def train_discrete_adjoint(
    base_model: FlowMatchingModel,
    finetune_model: FlowMatchingModel,
    reward_model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    config: Dict[str, Any],
    num_epochs: int = 25,
    lambda_reward: float = 12500.0,
    num_steps: int = 40,
    lr: float = 1e-5,  # lower LR for discrete adjoint stability
    adam_betas: tuple = (0.95, 0.999),
    weight_decay: float = 1e-2,
    grad_clip: float = 1.0,
    precision: str = "bfloat16",
    dt_offset: bool = True,
    device: str = "cuda",
    log_every: int = 50,
    save_every: int = 500,
    save_dir: str = "./checkpoints",
):
    """
    Discrete Adjoint method.
    
    Stores full computation graph and differentiates through the solver.
    Uses gradient checkpointing to reduce memory.
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
    schedule = MemorylessNoiseSchedule(use_offset=dt_offset, dt=dt)
    
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
    
    global_step = 0
    
    for epoch in range(num_epochs):
        pbar = tqdm(dataloader, desc=f"DiscAdjoint Epoch {epoch + 1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            context = batch["text_embeddings"].to(device)
            batch_size = context.shape[0]
            
            # Sample full trajectory with gradient tracking
            x0 = torch.randn(batch_size, finetune_model.unet.in_channels,
                           finetune_model.unet.in_channels,
                           finetune_model.unet.in_channels, device=device)
            
            x = x0
            total_control_cost = 0.0
            
            for i in range(num_steps):
                t_val = i * dt
                t = torch.full((batch_size,), t_val, device=device)
                sigma = schedule.sigma(t)
                kappa = schedule.kappa_t(t)
                
                v_f = finetune_model(x, t * 1000, context)
                v_b = base_model(x.detach(), t * 1000, context)
                
                drift = 2.0 * v_f - kappa * x
                
                # Control cost
                scale = schedule.compute_control_scaling(t)
                u = scale[:, None, None, None] * (v_f - v_b)
                total_control_cost += 0.5 * (u ** 2).sum(dim=(1, 2, 3)).mean() * dt
                
                noise = torch.randn_like(x)
                x = x + dt * drift + (dt ** 0.5) * sigma[:, None, None, None] * noise
            
            # Terminal reward
            reward = lambda_reward * reward_model(x)
            
            # SOC objective: control cost - reward
            loss = total_control_cost - reward.mean()
            
            optimizer.zero_grad()
            
            if precision == "bfloat16":
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(finetune_model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(finetune_model.parameters(), grad_clip)
                optimizer.step()
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "reward": f"{reward.mean().item():.4f}",
            })
            
            if global_step % save_every == 0 and global_step > 0:
                torch.save({
                    "step": global_step,
                    "model_state_dict": finetune_model.state_dict(),
                }, f"{save_dir}/disc_adj_step_{global_step}.pt")
            
            global_step += 1
    
    return finetune_model


# Helper functions

def sample_trajectory_memoryless_static(
    base_model, finetune_model, context, schedule, num_steps, dt
):
    """Non-gradient version of trajectory sampling."""
    batch_size = context.shape[0]
    device = context.device
    x0 = torch.randn(batch_size, finetune_model.unet.in_channels,
                    finetune_model.unet.in_channels,
                    finetune_model.unet.in_channels, device=device)
    trajectory = [x0]
    x = x0
    for i in range(num_steps):
        t_val = i * dt
        t = torch.full((batch_size,), t_val, device=device)
        sigma = schedule.sigma(t)
        kappa = schedule.kappa_t(t)
        v_f = finetune_model(x, t * 1000, context)
        drift = 2.0 * v_f - kappa * x
        noise = torch.randn_like(x)
        x_next = x + dt * drift + (dt ** 0.5) * sigma[:, None, None, None] * noise
        trajectory.append(x_next.clone())
        x = x_next
    return trajectory


def compute_control(base_model, finetune_model, x, t_val, context, schedule):
    """Compute control u(x,t)."""
    device = x.device
    batch_size = x.shape[0]
    t = torch.full((batch_size,), t_val, device=device)
    v_f = finetune_model(x, t * 1000, context)
    v_b = base_model(x.detach(), t * 1000, context)
    scale = schedule.compute_control_scaling(t)
    return scale[:, None, None, None] * (v_f - v_b)


def select_gradient_timesteps(num_steps, first_n=10, last_n=10):
    """Select subset of timesteps for gradient computation."""
    cutoff = int(num_steps * 0.75)
    first_subset = random.sample(range(cutoff), min(first_n, cutoff))
    last_subset = list(range(num_steps - last_n, num_steps))
    return sorted(set(first_subset + last_subset))

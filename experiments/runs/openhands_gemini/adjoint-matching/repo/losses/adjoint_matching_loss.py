
import torch
import torch.nn as nn
from typing import Callable, Tuple, List
import random

from adjoint_matching.config import Config
from adjoint_matching.utils import MemorylessFlowMatchingSDE, solve_sde_fwd, solve_adjoint_ode_bwd
from adjoint_matching.models.model import FlowMatchingModel, RewardModel

class AdjointMatchingLoss(nn.Module):
    def __init__(self, config: Config, base_model: FlowMatchingModel, reward_model: RewardModel):
        super().__init__()
        self.config = config
        self.base_model = base_model # v^base
        self.reward_model = reward_model
        self.memoryless_sde = MemorylessFlowMatchingSDE(config)
        self.lambda_reward = config.LAMBDA_REWARD_SCALING
        self.lct = config.LCT_ADJOINT_MATCHING

    def _sample_gradient_timesteps(self, num_total_steps: int) -> List[int]:
        """
        Samples timesteps for gradient evaluation as described in Appendix G.2.
        """
        sampled_timesteps = set()

        # Sample uniform timesteps from [0, 0.725]
        num_uniform = self.config.NUM_UNIFORM_TIMESTEPS_FOR_GRAD
        max_uniform_t_idx = int(self.config.UNIFORM_TIMESTEPS_RANGE_END * num_total_steps)
        if max_uniform_t_idx > 0:
            uniform_indices = random.sample(range(max_uniform_t_idx), min(num_uniform, max_uniform_t_idx))
            sampled_timesteps.update(uniform_indices)

        # Always sample the last 10 timesteps from [0.75, 0.975]
        num_last = self.config.NUM_LAST_TIMESTEPS_FOR_GRAD
        min_last_t_idx = int(self.config.LAST_TIMESTEPS_RANGE_START * num_total_steps)
        last_indices = range(min_last_t_idx, num_total_steps)
        sampled_timesteps.update(last_indices)
        
        return sorted(list(sampled_timesteps))

    def forward(self, finetuned_model: FlowMatchingModel, x_0: torch.Tensor, text_conditioning: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes the Adjoint Matching loss.
        finetuned_model: The model being fine-tuned (v^finetune)
        x_0: Initial noise for SDE (N(0,I))
        text_conditioning: Text embeddings for conditional generation
        """
        
        # 1. Simulate forward SDE to get trajectories X_t (Algorithm 1, Line 3)
        # The score_fn for SDE solver needs to be v_finetune
        # The SDE uses Memoryless Noise Schedule sigma(t) = sqrt(2 * eta_t)
        
        # We need a wrapper for finetuned_model.forward that matches the signature expected by SDE.b
        # SDE.b expects score_fn(x, t), and in FM, v is the vector field, so we use v as score
        
        def finetuned_v_field(x: torch.Tensor, t: float):
            # t is a float, convert to tensor for model input
            t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
            return finetuned_model(x, t_tensor, text_conditioning)

        # x_0 is typically N(0,I). In the paper, it's mentioned x_0 ~ N(0,I)
        x_0_sde = torch.randn_like(x_0) # Initialize with noise
        
        # Solve forward SDE for X_t
        # This gives a trajectory X_0, X_h, ..., X_{1-h}
        x_trajectory_tensors = []
        x_current = x_0_sde.to(self.config.DEVICE, self.config.DTYPE)
        
        times = torch.linspace(0, 1.0 - self.config.H, self.config.K_TIMESTEPS, device=self.config.DEVICE, dtype=self.config.DTYPE)
        
        for i in range(self.config.K_TIMESTEPS):
            t = times[i].item()
            x_current = self.memoryless_sde.sde_step(x_current, t, self.config.H, finetuned_v_field)
            x_trajectory_tensors.append(x_current.detach()) # Detach for adjoint calculation

        # 2. Solve the lean adjoint ODE backwards (Algorithm 1, Line 6)
        # The base_score_fn here needs to be v_base
        def base_v_field(x: torch.Tensor, t: float):
            t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
            return self.base_model(x, t_tensor, text_conditioning)
        
        adjoint_trajectory = solve_adjoint_ode_bwd(
            x_trajectory_tensors,
            self.memoryless_sde,
            base_v_field,
            self.reward_model,
            self.config.H,
            self.config.K_TIMESTEPS,
            self.lambda_reward
        )
        
        # 3. Compute Adjoint Matching Objective (Algorithm 1, Line 9, Eq. 37)
        total_loss = torch.tensor(0.0, device=self.config.DEVICE, dtype=self.config.DTYPE)
        
        # Sample timesteps for gradient calculation (Appendix G.2)
        grad_timesteps_indices = self._sample_gradient_timesteps(self.config.K_TIMESTEPS)
        
        for i in grad_timesteps_indices:
            t_idx = i
            t = times[t_idx].item()
            
            x_t = x_trajectory_tensors[t_idx].clone().detach().requires_grad_(True)
            tilde_a_t = adjoint_trajectory[t_idx].clone().detach() # Detach as per Algo 1

            # Get v_finetune(X_t, t) and v_base(X_t, t)
            t_tensor = torch.full((x_t.shape[0],), t, device=x.device, dtype=x.dtype)
            v_finetune_xt = finetuned_model(x_t, t_tensor, text_conditioning)
            v_base_xt = self.base_model(x_t, t_tensor, text_conditioning)

            # Get sigma(t)
            sigma_t = self.memoryless_sde.sigma(t)
            
            # Loss term: || (2/sigma(t)) * (v_finetune - v_base) + sigma(t) * tilde_a_t ||^2 (Eq. 37, modified)
            # The form in Algo 1 is for DDIM/DDPM, for Flow Matching it's (2/sigma(t)) * (v_finetune - v_base) + sigma(t) * tilde_a_t
            # This is derived from u = ... (v_finetune - v_base) + sigma(t)^T tilde_a
            
            # Reconcile Eq 27 and Eq 37 and Algo 1
            # From Eq 27, u(x,t) = sqrt(2 / (beta_t * (kappa_t * beta_t - dot(beta_t)))) * (v_finetune - v_base)
            # This is u = C * (v_finetune - v_base)
            # And in the Adjoint Matching objective (Eq 37) it's || u + sigma(t)^T tilde_a ||^2
            # Substituting u, we get || C * (v_finetune - v_base) + sigma(t)^T tilde_a ||^2
            # Here, sigma(t) = sqrt(2 * eta_t) (Memoryless noise schedule)
            # eta_t = beta_t * (kappa_t * beta_t - dot(beta_t))
            # So, C = sqrt(2 / eta_t) = 2 / sigma(t)
            # Thus, the term is || (2/sigma(t)) * (v_finetune - v_base) + sigma(t) * tilde_a_t ||^2
            
            loss_term = ((2.0 / sigma_t) * (v_finetune_xt - v_base_xt) + sigma_t * tilde_a_t).norm(dim=-1)**2
            
            # Apply loss clipping (Appendix G.3)
            loss_term_clipped = torch.min(loss_term, torch.full_like(loss_term, self.lct))
            total_loss += loss_term_clipped.mean() # Mean over batch

        return total_loss / len(grad_timesteps_indices) # Average over sampled timesteps


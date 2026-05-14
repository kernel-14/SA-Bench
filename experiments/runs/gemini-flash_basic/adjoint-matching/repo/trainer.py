import torch
import torch.nn as nn
import math

from model import BaseModel, FlowMatchingModel, DDPMModel
from networks import VectorField, RewardModel

class SDEIntegrator:
    """
    Integrates an SDE using the Euler-Maruyama method.
    """
    def __init__(self, model_params: BaseModel, num_steps: int = 100):
        self.model_params = model_params
        self.num_steps = num_steps
        self.dt = 1.0 / num_steps
        self.ts = torch.linspace(0, 1, num_steps + 1)

    def _get_drift_and_diffusion(self, x: torch.Tensor, t: float,
                                 vector_field: VectorField) -> tuple[torch.Tensor, float]:
        """
        Returns the drift and diffusion coefficients for the SDE.
        The drift depends on the model type (FlowMatching or DDPM).
        The diffusion is always the memoryless sigma(t).
        """
        sigma_t = self.model_params.memoryless_sigma_t(t)

        if isinstance(self.model_params, FlowMatchingModel):
            # For Flow Matching, the drift from Eq 341 or 251 (for finetuned process)
            # The drift is (2 * v_finetune - kappa_t * x)
            v_output = vector_field(x, t)
            # Handle potential inf in kappa_t at t=0
            kappa_t_val = self.model_params.kappa_t(t)
            if math.isinf(kappa_t_val):
                # At t=0, kappa_t is inf, but drift should be well-behaved or starting point is noise.
                # The forward SDE starts from X_0, where usually kappa_t is well-defined or term vanishes.
                # If kappa_t * x is present, this needs careful handling or assuming t > 0 for dynamics.
                # Given X_0 ~ N(0,I), x is initially zero, so kappa_t * x is 0*inf = NaN.
                # The problem definition might imply starting at t=epsilon or handling this edge case.
                # For now, if t=0 and kappa_t is inf, we'll assume the term kappa_t * x effectively doesn't contribute if x=0.
                # However, it's safer to avoid exact t=0 where kappa_t is infinite. Let's assume t is slightly > 0.
                # The integration loop samples t from self.ts, which includes 0. Will handle this in loop.
                if t == 0: # This means we are at the initial point, X_0
                    # Drift at t=0 for FlowMatching, from Lipman et al. 2023, is v(X_0, 0) = alpha_dot(0) X_0 + beta(0) X_1
                    # which implies v(0,0) = 0 for X_0 ~ N(0,I)
                    # The simplified SDE in (3) or (4) does not directly provide this.
                    # Let's assume that the terms with kappa_t are handled by starting integration at t=dt or by definition of v.
                    # For this implementation, I will assume the kappa_t(t) * x term is only active for t > 0
                    # and that at t=0, drift is purely from v_output if x=0. But v_output will depend on t.
                    # A simpler approach is to use the formula and let the numerical stability dictate it, or ensure t > 0.
                    # The prompt implies running the code, so a robust numerical handling for t=0 is needed.
                    # For now, if kappa_t is inf, and x is finite, then kappa_t * x will be inf. This is a problem.
                    # Let's assume a small epsilon to avoid t=0 problem for kappa_t/eta_t.
                    pass # Will handle this in the trainer or main loop
            drift = 2 * v_output - kappa_t_val * x
            return drift, sigma_t
        elif isinstance(self.model_params, DDPMModel):
            # For DDPM, the drift from Eq 245 (for finetuned process)
            epsilon_output = vector_field(x, t)
            alpha_bar_t = self.model_params.alpha_bar_t_schedule(t)
            kappa_t_val = self.model_params.kappa_t(t)

            # Handle potential division by zero for 1.0 - alpha_bar_t
            sqrt_one_minus_alpha_bar_t = math.sqrt(1.0 - alpha_bar_t) if (1.0 - alpha_bar_t) > 1e-6 else 1e-6

            drift = (kappa_t_val / 2) * x - 
                    kappa_t_val * epsilon_output / sqrt_one_minus_alpha_bar_t
            return drift, sigma_t
        else:
            raise NotImplementedError("Unsupported BaseModel type.")

    def integrate(self, x_0: torch.Tensor,
                  vector_field: VectorField) -> list[torch.Tensor]:
        """
        Forward simulation of the SDE.
        """
        trajectory = []
        x_t = x_0
        for i in range(self.num_steps + 1): # Include t=1 in loop
            t = self.ts[i].item()

            if i == 0:
                # For t=0, special handling to avoid division by zero in kappa_t for FM
                # and ensure X_0 is just the initial noise.
                trajectory.append(x_t)
                continue

            # Use detach() to ensure x_t is not part of the gradient graph for SDE integration
            # The gradients for fine_tuned_vf are computed during loss calculation.
            x_t_detached = x_t.detach()
            with torch.no_grad():
                drift, sigma_t = self._get_drift_and_diffusion(x_t_detached, t, vector_field)
                
                # SDE step
                noise = torch.randn_like(x_t)
                x_t = x_t + drift * self.dt + sigma_t * math.sqrt(self.dt) * noise
                trajectory.append(x_t)
        return trajectory


class AdjointMatchingTrainer(nn.Module):
    def __init__(self, 
                 model_params: BaseModel,
                 base_vector_field: VectorField, 
                 finetuned_vector_field: VectorField, 
                 reward_model: RewardModel,
                 num_sde_steps: int = 100,
                 num_adjoint_steps: int = 100):
        super().__init__()
        self.model_params = model_params
        self.base_vf = base_vector_field
        self.finetuned_vf = finetuned_vector_field
        self.reward_model = reward_model
        
        self.sde_integrator = SDEIntegrator(model_params, num_sde_steps)
        self.adjoint_dt = 1.0 / num_adjoint_steps
        self.adjoint_ts = torch.linspace(1, 0, num_adjoint_steps + 1) # Solve backwards from 1 to 0

    def _get_base_drift_for_adjoint(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """
        Returns the drift component for the base model, specifically for the lean adjoint calculation.
        This uses the base_vector_field.
        """
        if isinstance(self.model_params, FlowMatchingModel):
            # For Flow Matching, from Eq 347, the term inside gradient is 2 * v_base - kappa_t * x
            v_base_output = self.base_vf(x, t)
            kappa_t_val = self.model_params.kappa_t(t)
            # See comments in SDEIntegrator for t=0 handling of kappa_t.
            # For adjoint, we also need to ensure numerical stability.
            return 2 * v_base_output - kappa_t_val * x
        elif isinstance(self.model_params, DDPMModel):
            epsilon_base_output = self.base_vf(x, t)
            kappa_t_val = self.model_params.kappa_t(t)
            alpha_bar_t = self.model_params.alpha_bar_t_schedule(t)

            # Handle potential division by zero for 1.0 - alpha_bar_t
            sqrt_one_minus_alpha_bar_t = math.sqrt(1.0 - alpha_bar_t) if (1.0 - alpha_bar_t) > 1e-6 else 1e-6

            # This is the drift b(x,t) from general form (Eq 10-11)
            # Derived in previous turn, simplified to: kappa_t * x - kappa_t * epsilon_base / sqrt(1-alpha_bar_t)
            drift = kappa_t_val * x - 
                    kappa_t_val * epsilon_base_output / sqrt_one_minus_alpha_bar_t
            return drift
        else:
            raise NotImplementedError("Unsupported BaseModel type.")

    def _solve_lean_adjoint(self, trajectory: list[torch.Tensor], reward_grad_x1: torch.Tensor) -> list[torch.Tensor]:
        """
        Backward simulation of the lean adjoint ODE.
        """
        adjoint_trajectory_reversed = [reward_grad_x1] # a_1 = -nabla_x1 r(X_1) is the terminal condition
        a_t = reward_grad_x1
        
        # Iterate backwards from t=1 to t=0
        for i in range(self.adjoint_ts.shape[0] - 1): # loop num_adjoint_steps times
            # t_current is the time at which a_t is known, we want to find a_{t-h}
            t_current_idx_for_sde_ts = self.sde_integrator.num_steps - i # Corresponding forward time index
            t_current = self.sde_integrator.ts[t_current_idx_for_sde_ts].item()
            
            # X_t from the forward trajectory, requires grad for computing JVP
            # Ensure it is detached to break graph through SDE, but requires_grad for computing its jacobian
            x_t = trajectory[t_current_idx_for_sde_ts].detach().requires_grad_(True)

            # Compute gradient of base_drift w.r.t. x_t (JVP: a_t.T * nabla_Xt(drift_base))
            # base_drift_val will involve base_vf, which is fixed, so no grad through it
            base_drift_val = self._get_base_drift_for_adjoint(x_t, t_current)
            
            # grad_outputs needs to be a tensor of the same shape as base_drift_val
            # and contains the 'a_t' values. Detach a_t to not form graph through previous adjoint steps.
            grad_outputs_for_jvp = a_t.detach()

            grad_drift_x_product = torch.autograd.grad(
                base_drift_val,
                x_t,
                grad_outputs=grad_outputs_for_jvp,
                retain_graph=True, # May need to retain graph if x_t is part of a larger graph for current base_vf call
                allow_unused=True # In case base_drift_val doesn't depend on x_t directly for some models
            )[0]
            
            if grad_drift_x_product is None:
                grad_drift_x_product = torch.zeros_like(x_t)

            # Adjoint ODE step: a_{t-h} = a_t + h * a_t.T * nabla_Xt(drift_base)
            a_t = a_t + self.adjoint_dt * grad_drift_x_product
            adjoint_trajectory_reversed.append(a_t)
        
        return adjoint_trajectory_reversed[::-1] # Reverse to be in chronological order from t=0 to t=1


    def _get_control_u(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """
        Computes the control u(x,t) based on the difference between finetuned and base vector fields.
        """
        finetuned_output = self.finetuned_vf(x, t)
        base_output = self.base_vf(x, t)

        if isinstance(self.model_params, FlowMatchingModel):
            # From Eq 251: u(x,t) = sqrt(2 / eta_t) * (v_finetune - v_base)
            # And we know sqrt(2 / eta_t) = 2 / sigma(t) for memoryless sigma(t)
            sigma_t_val = self.model_params.memoryless_sigma_t(t)
            coeff = 2.0 / sigma_t_val if sigma_t_val != 0 else 0.0 # Handle division by zero
            return coeff * (finetuned_output - base_output)
        elif isinstance(self.model_params, DDPMModel):
            # From Eq 245: u(x,t) = - sqrt(dot_alpha_bar_t / (alpha_bar_t * (1 - alpha_bar_t))) * (eps_finetune - eps_base)
            alpha_bar_t = self.model_params.alpha_bar_t_schedule(t)
            dot_alpha_bar_t = self.model_params.dot_alpha_bar_t_schedule(t)
            
            # Handle potential division by zero for alpha_bar_t or 1-alpha_bar_t at endpoints
            # Add a small epsilon to denominator to prevent division by zero for schedules that hit 0 or 1.
            denominator = alpha_bar_t * (1.0 - alpha_bar_t)
            if denominator < 1e-6: # Also covers cases where alpha_bar_t is near 0 or 1
                return torch.zeros_like(finetuned_output)
                
            coeff = -1.0 * math.sqrt(dot_alpha_bar_t / denominator)
            return coeff * (finetuned_output - base_output)
        else:
            raise NotImplementedError("Unsupported BaseModel type.")

    def forward(self, x_0: torch.Tensor) -> torch.Tensor:
        """
        Computes the Adjoint Matching loss for a batch of initial noise x_0.
        """
        # 1. Simulate forward SDE to get trajectory X_t (without gradients from SDE itself)
        # The trajectory elements are detached by the SDEIntegrator
        trajectory = self.sde_integrator.integrate(x_0, self.finetuned_vf)
        X_1 = trajectory[-1]

        # 2. Compute terminal gradient for adjoint ODE: -nabla_X1 r(X_1)
        # X_1 needs to have gradients enabled to compute its gradient with respect to reward.
        X_1_requires_grad = X_1.detach().requires_grad_(True)
        r_X1 = self.reward_model(X_1_requires_grad)
        
        # Calculate gradients with respect to X_1. Sum is used as reward is scalar for each element in batch.
        reward_grad_x1 = torch.autograd.grad(r_X1.sum(), X_1_requires_grad, create_graph=False)[0] 
        # The paper uses -nabla_X1 r(X_1)
        reward_grad_x1 = -reward_grad_x1

        # 3. Solve lean adjoint ODE backwards to get a_t (without gradients from ODE itself)
        adjoint_trajectory = self._solve_lean_adjoint(trajectory, reward_grad_x1)

        # 4. Compute Adjoint Matching objective (Loss)
        total_loss = 0.0
        num_loss_terms = 0

        # Iterate through trajectory and adjoint_trajectory (which are in chronological order)
        for i in range(self.sde_integrator.num_steps + 1): # Iterate over all time steps including t=0 and t=1
            t = self.sde_integrator.ts[i].item()
            X_t = trajectory[i] # X_t is already detached
            a_t = adjoint_trajectory[i] # a_t is already detached

            # Skip calculation if sigma_t is problematic (e.g., zero or inf) or if at t=0 for kappa_t/eta_t issues
            # A small epsilon check for sigma_t is prudent.
            current_sigma_t = self.model_params.memoryless_sigma_t(t)
            if abs(current_sigma_t) < 1e-6 or math.isinf(current_sigma_t):
                continue
            
            # The control u(X_t, t) needs gradients with respect to finetuned_vf parameters
            # so we re-compute it here, with X_t detached.
            control_u_val = self._get_control_u(X_t.detach(), t) # X_t is already detached from SDE integrator

            # Adjoint Matching objective term: || u(X_t, t) + sigma(t) * a_t ||^2
            # The term a_t is derived from reward_grad_x1, which was detached. So a_t itself has no graph.
            # The loss needs to backpropagate through control_u_val, which depends on finetuned_vf.
            loss_term = torch.mean(torch.sum((control_u_val + current_sigma_t * a_t)**2, dim=-1)) * self.sde_integrator.dt
            total_loss += loss_term
            num_loss_terms += 1
        
        if num_loss_terms == 0:
            # This case implies no valid time steps were found, which might be an issue with dt or sigma_t definitions.
            return torch.tensor(0.0, device=x_0.device)
        
        # The sum of squared differences is an integral, so we multiply by dt for each term
        # The division by num_loss_terms normalizes over the number of valid time steps.
        # The integral already implies a sum over dt, so total_loss is already scaled by dt.
        # If we want a mean across batch and time, then it's sum_over_time(mean_over_batch(loss_term)) / num_timesteps
        return total_loss # This is sum over timesteps. If we want mean over time, divide by num_loss_terms

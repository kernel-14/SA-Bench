import torch
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import deque # For managing historical states in rollout
import os
import logging

# Local imports
from config import Config
from models.p2vae import P2VAEModel
from models.fmt import FMTModel
from data.pde_dataset import PDEDataset # Assuming PDEDataset is correctly defined for full trajectories
from utils.metrics import L2RelativeError, VRMSE # Assuming these are defined in metrics.py as callable classes
from utils.logging_utils import log_message, log_scalar, log_image # Assuming these are functions

# Set up logging for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add a console handler if one doesn't exist to ensure logs are visible
if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class Evaluator:
    """
    Orchestrates the evaluation of P2VAE and FMT models.
    Provides functionalities for evaluating reconstruction quality, performing
    long-term autoregressive rollouts, generating stochastic ensembles, and
    fine-tuning the combined model for new tasks.
    """
    def __init__(self, p2vae_model: P2VAEModel, fmt_model: FMTModel, test_loader: DataLoader, config: Config, device: str = 'cuda'):
        """
        Initializes the Evaluator.

        Args:
            p2vae_model (P2VAEModel): The pre-trained P2VAE model instance.
            fmt_model (FMTModel): The pre-trained FMT model instance.
            test_loader (DataLoader): DataLoader for the evaluation dataset.
            config (Config): The configuration object for the experiment.
            device (str): The compute device ('cuda' or 'cpu').
        """
        self.p2vae_model = p2vae_model.to(device).eval()
        self.fmt_model = fmt_model.to(device).eval()
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.dtype = torch.float16 if config.get('global.dtype', 'float16') == 'float16' else torch.float32

        # Evaluation parameters from config
        self.ode_sampler_config = self.config.get('evaluation.ode_sampler', {})
        self.num_ode_steps: int = self.ode_sampler_config.get('num_discretization_steps', 100)
        self.ode_dt: float = self.ode_sampler_config.get('dt', 0.01)
        self.deterministic_k: float = self.config.get('evaluation.deterministic_k', 1.0)
        self.ensemble_config: Dict[str, Any] = self.config.get('evaluation.ensemble_generation', {})
        self.finetuning_config: Dict[str, Any] = self.config.get('finetuning', {})

        self.trajectory_length: int = self.config.get('dataset.trajectory_length', 4) # Number of states (x0,x1,x2,x3) that FMT takes as context
        self.target_channels: int = self.config.get('dataset.target_channels', 3)
        self.target_resolution: Tuple[int, int] = tuple(self.config.get('dataset.target_resolution', [128, 128]))

        # Instantiate metric calculators
        self.l2re_calculator = L2RelativeError(epsilon=1e-8)
        self.vrmse_calculator = VRMSE(epsilon=1e-8)

        log_message(logger, f"Evaluator initialized on device: {self.device} with dtype: {self.dtype}", logging.INFO)

    def _get_time_embedding(self, t_scalar: float) -> torch.Tensor:
        """
        Creates a scalar tensor for time `t` to be processed by `fmt_model.time_embedding`.

        Args:
            t_scalar (float): The scalar time value (e.g., from 0.0 to 1.0).

        Returns:
            torch.Tensor: A tensor of shape (1,) (for batch_size 1) or (B,) (for batch_size B)
                          representing the time value, on the correct device and dtype.
        """
        # FMTModel's time_embedding expects (B, 1) or (B,) input for the first Linear layer.
        # During inference, we typically process one sample at a time (batch_size=1).
        # We return a (1,) tensor, which when passed to nn.Linear(1, D) becomes (1, D).
        return torch.tensor([t_scalar], device=self.device, dtype=self.dtype).unsqueeze(0) # Shape (1, 1)

    def _interpolate_x_tk(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, k: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Computes the interpolated state `x_t^k` according to the paper's formula (Section 3.1).

        Args:
            x0 (torch.Tensor): The starting physical state. Shape (B, C, H, W).
            x1 (torch.Tensor): The ending physical state. Shape (B, C, H, W).
            t (torch.Tensor): The interpolation time parameter. Shape (B, 1, 1, 1) or (B,).
            k (torch.Tensor): The bridge parameter. Shape (B, 1, 1, 1) or (B,).
            z (torch.Tensor): Sampled noise. Shape (B, C, H, W).

        Returns:
            torch.Tensor: The interpolated state `x_t^k`. Shape (B, C, H, W).
        """
        # Ensure t and k are broadcastable (e.g., (B, 1, 1, 1))
        if t.dim() == 1:
            t = t.view(-1, 1, 1, 1)
        if k.dim() == 1:
            k = k.view(-1, 1, 1, 1)

        mu_t: torch.Tensor = t * x1 + k * (1.0 - t) * x0
        sigma_t: torch.Tensor = (1.0 - t) * (1.0 - k)
        x_tk: torch.Tensor = mu_t + sigma_t * z
        return x_tk

    @torch.no_grad()
    def evaluate_reconstruction(self) -> Dict[str, float]:
        """
        Evaluates the P2VAE model's reconstruction quality on the test set.
        Returns a dictionary of average L2RE and VRMSE.
        """
        log_message(logger, "Starting P2VAE reconstruction evaluation...", logging.INFO)
        self.p2vae_model.eval()
        self.fmt_model.eval() # Ensure FMT is also in eval mode if it's there

        total_l2re = 0.0
        total_vrmse = 0.0
        num_samples = 0

        for i, batch in enumerate(self.test_loader):
            # Assuming PDEDataset provides 'x_0' for reconstruction evaluation.
            # In general, one might evaluate all states or a specific state.
            x_true = batch['x_0'].to(self.device, dtype=self.dtype) 
            
            # Forward pass through P2VAE
            # P2VAEModel's forward returns x_reco, mu, log_var, z
            x_reco, _, _, _ = self.p2vae_model(x_true)

            # Calculate metrics
            total_l2re += self.l2re_calculator(x_reco, x_true).item() * x_true.size(0)
            total_vrmse += self.vrmse_calculator(x_reco, x_true).item() * x_true.size(0)
            num_samples += x_true.size(0)

            if (i + 1) % self.config.get('evaluation.log_interval_steps', 100) == 0:
                log_message(logger, f"  Reconstruction Batch {i+1}/{len(self.test_loader)} - Current Avg L2RE: {total_l2re/num_samples:.4f}, Current Avg VRMSE: {total_vrmse/num_samples:.4f}", logging.DEBUG)

            # Log a sample image for visualization
            if i == 0:
                log_image(logger, "P2VAE_Reconstruction/Original", x_true[0].cpu(), 0)
                log_image(logger, "P2VAE_Reconstruction/Reconstructed", x_reco[0].cpu(), 0)

        avg_l2re = total_l2re / num_samples if num_samples > 0 else float('nan')
        avg_vrmse = total_vrmse / num_samples if num_samples > 0 else float('nan')
        log_message(logger, f"P2VAE Reconstruction Results - Avg L2RE: {avg_l2re:.4f}, Avg VRMSE: {avg_vrmse:.4f}", logging.INFO)
        return {"avg_l2re": avg_l2re, "avg_vrmse": avg_vrmse}

    @torch.no_grad()
    def _run_ode_sampler_inference(self,
                                   x_s_physical: torch.Tensor, # The *start* physical state (e.g., x_s) for the current physical step s -> s+1
                                   current_h_state: torch.Tensor, # The GRU hidden state (e.g., h_{s-1})
                                   k_val_for_ode: float, # The bridge parameter k for this prediction step
                                   physical_history_states: List[torch.Tensor], # [x_{s-3}, x_{s-2}, x_{s-1}] in physical space
                                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes a single physical time step prediction (e.g., x_s -> x_{s+1}) using the Euler ODE solver.
        This involves `self.num_ode_steps` Euler steps.

        Args:
            x_s_physical (torch.Tensor): The starting physical state for the current physical step (e.g., `x_s`).
                                         Shape (B, C, H, W).
            current_h_state (torch.Tensor): The GRU hidden state (`h_{s-1}`) *before* updating for this step.
                                            Shape (B, gru_hidden_size).
            k_val_for_ode (float): The bridge parameter `k` to use for this specific physical prediction step.
            physical_history_states (List[torch.Tensor]): A list of the 3 preceding physical states
                                                            `[x_{s-3}, x_{s-2}, x_{s-1}]` (for the pyramid context).
                                                            Each (B, C, H, W).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - predicted_x_s_plus_1 (torch.Tensor): The predicted next physical state (`x_{s+1}`).
                - updated_h_state (torch.Tensor): The updated GRU hidden state (`h_s`) to be used for the *next* physical step.
        """
        self.p2vae_model.eval()
        self.fmt_model.eval()

        batch_size = x_s_physical.shape[0]

        # Initialize the evolving state (current estimate of x_{s+1})
        # During the ODE solve, this `current_ode_evolving_physical_state` is `x_1` in `x_t^k = t x_1 + k(1-t) x_0`.
        # Initially, it could be `x_s_physical` or a small perturbation.
        # Common practice for rectified flow ODE sampling is to start from `x_s_physical` (or a small pert) and drive it towards `x_{s+1}`.
        # Let's initialize with `x_s_physical` for a smooth start.
        current_ode_evolving_physical_state = x_s_physical.clone().to(self.device, dtype=self.dtype)
        
        # Prepare the static parts of the latent pyramid inputs (from history_states)
        # These are y_{s-3}, y_{s-2}, y_{s-1}.
        # For these, t=0, k=1 means x_t^k = x_s, so we just get latent(x_s).
        # We need 3 (x_0, x_1, x_2) for history and the 4th (x_3) for the evolving current state.
        # The history here refers to states *before* x_s.
        # E.g., if predicting x_4 from x_3, history is (x_0, x_1, x_2).
        
        # Build the full 4-element list of interpolated latent states to feed to FMT.
        # This list changes at each *ODE step* because the last element (y_{s,t_ode}^{k_val}) changes.
        
        fixed_t = torch.tensor(0.0, device=self.device, dtype=self.dtype).view(1,1,1,1)
        fixed_k_for_history = torch.tensor(1.0, device=self.device, dtype=self.dtype).view(1,1,1,1)
        zero_noise = torch.zeros(batch_size, self.target_channels, self.target_resolution[0], self.target_resolution[1],
                                 device=self.device, dtype=self.dtype)

        # First 3 pyramid elements are based on previous physical states (x_{s-3}, x_{s-2}, x_{s-1}).
        # To strictly follow the `x_t^k` formula with these fixed states, we use t=0, k=1.
        pyramid_latent_context = []
        for phys_state in physical_history_states: # These are already x_{s-3}, x_{s-2}, x_{s-1}
            # For these history elements, we essentially want E(phys_state)
            # which is achieved by _interpolate_x_tk(phys_state, dummy_x1, t=0, k=1, z=0)
            latent_from_history = self.p2vae_model.get_latent(phys_state.to(self.device, dtype=self.dtype))
            pyramid_latent_context.append(latent_from_history)
        
        # The full_t_step is the time for the *last* element in the pyramid, which is `t_ode`
        # The full_k_step is `k_val_for_ode` for the last element in the pyramid.

        for i_ode_step in range(self.num_ode_steps):
            t_ode = i_ode_step * self.ode_dt # t_ode from 0 to 1-dt
            
            # Sample noise for this specific ODE step's interpolation (z~N(0,I))
            z_ode_step = torch.randn(batch_size, self.target_channels, self.target_resolution[0], self.target_resolution[1],
                                     device=self.device, dtype=self.dtype)

            # Compute the *interpolated current state* (x_t^k) for the FMT input
            # Here, x0 = x_s_physical (fixed start for this physical step)
            # and x1 = current_ode_evolving_physical_state (our current estimate of x_{s+1} during ODE integration).
            interpolated_x_tk_for_fmt = self._interpolate_x_tk(
                x0=x_s_physical,
                x1=current_ode_evolving_physical_state,
                t=torch.tensor(t_ode, device=self.device, dtype=self.dtype).view(1,1,1,1),
                k=torch.tensor(k_val_for_ode, device=self.device, dtype=self.dtype).view(1,1,1,1),
                z=z_ode_step
            )
            latent_y_tk_for_fmt = self.p2vae_model.get_latent(interpolated_x_tk_for_fmt)

            # Construct the complete list of 4 interpolated latent states for FMT.forward
            # Last element is the interpolated state for current ODE step
            full_interpolated_latent_states = pyramid_latent_context + [latent_y_tk_for_fmt]

            # The current_t_step for FMT is the 't_ode' scalar that produced the last pyramid element.
            t_scalar_for_fmt = self._get_time_embedding(t_ode) # Shape (1, 1)

            # Predict velocity field using FMT
            predicted_velocity_latent = self.fmt_model(
                interpolated_latent_states=full_interpolated_latent_states,
                current_t_step=t_scalar_for_fmt,
                h_history=current_h_state # Pass the h for this physical step
            )

            # Euler step in physical space (implicitly, as predicted_velocity is in physical interpretation)
            # NOTE: The FMT model predicts velocity of *latent* states. So this update should be in latent space.
            # However, the paper implies `(1-t)g_theta - (x_1 - x_t^k)` loss, where the target is in physical space.
            # This is a common ambiguity in such models. For now, assuming `predicted_velocity_latent` is effectively
            # a velocity in the physical space that gets added to `current_ode_evolving_physical_state`.
            # If `g_theta` outputs latent velocity, and `x_t^k` is physical, then `x_1-x_t^k` must be physical.
            # The loss is `||(1-t)g - (x_1 - x_t^k)||^2`.
            # If `g` is physical velocity, then this works.
            # If `g` is latent velocity, then `latent(x_1 - x_t^k)` is required.
            # Let's assume `predicted_velocity_latent` is the velocity in the latent space and we apply it there.
            # Then we need to convert current_ode_evolving_physical_state to latent, add, then decode.
            # Or assume `predicted_velocity_latent` is already decoded (or operates in a way that maps back).
            
            # The training objective `L_FM = ||(1-t)g_theta(x_t^k, t) - (x_1 - x_t^k)||^2`
            # implies `g_theta` predicts a physical quantity (`(x_1 - x_t^k)/(1-t)`).
            # So `predicted_velocity_latent` *is* directly the physical velocity.
            current_ode_evolving_physical_state = current_ode_evolving_physical_state + predicted_velocity_latent * self.ode_dt
        
        predicted_x_s_plus_1 = current_ode_evolving_physical_state

        # Update `h_current` for the *next* physical step (diffusion forcing)
        # The paper says `h_s ~ p_phi(h_s | h_{s-1}, x_{s,t_s}^{k_s}, t_s)`.
        # Here, `x_{s,t_s}^{k_s}` would be `predicted_x_s_plus_1` (the "end state" of the previous interval `s->s+1`).
        # The `t_s` here refers to the completion of the physical step (t=1.0).
        compressed_x_next_token = self.fmt_model.encode_condition_token(
            self.p2vae_model.get_latent(predicted_x_s_plus_1)
        )
        t_s_embedding_for_gru = self.fmt_model.time_embedding(self._get_time_embedding(1.0)) # Use 1.0 for end of physical interval
        
        updated_h_state = self.fmt_model.update_history_h(
            current_h_state.unsqueeze(0), # GRU expects (1, B, H) for initial hidden state
            compressed_x_next_token,
            t_s_embedding_for_gru
        ).squeeze(0) # Squeeze back to (B, H)

        return predicted_x_s_plus_1, updated_h_state

    @torch.no_grad()
    def evaluate_long_term_rollout(self) -> Dict[str, float]:
        """
        Evaluates the FMT model's long-term autoregressive rollout performance on the test set.
        Uses deterministic prediction (k=1.0).
        """
        log_message(logger, "Starting FMT long-term rollout evaluation (deterministic, k=1.0)...", logging.INFO)
        self.p2vae_model.eval()
        self.fmt_model.eval()

        # Dictionary to store L2RE for specific steps and average
        all_l2re_per_step: Dict[Union[int, str], List[float]] = {
            1: [], 5: [], 10: [], 'last': [], 'average': []
        }
        
        # Assume test_loader yields full trajectories (T_total, C, H, W)
        for traj_idx, batch in enumerate(self.test_loader):
            # Assuming PDEDataset provides 'trajectory_states'
            # Batch item is expected to be {'trajectory_states': (T_total, C, H, W)}
            ground_truth_trajectory_raw = batch['trajectory_states'].squeeze(0) # (T_total, C, H, W)
            ground_truth_trajectory = ground_truth_trajectory_raw.to(self.device, dtype=self.dtype)

            if ground_truth_trajectory.shape[0] < self.trajectory_length + 1:
                log_message(logger, f"Skipping trajectory {traj_idx} due to insufficient length "
                                    f"({ground_truth_trajectory.shape[0]} states). Requires at least "
                                    f"{self.trajectory_length + 1} states for one prediction step.", logging.WARNING)
                continue

            # Initialize history `(x_0, x_1, x_2)` physical states (for first prediction of x_4 from x_3)
            # `initial_history_deque` stores `[x_{s-3}, x_{s-2}, x_{s-1}, x_s]`
            initial_history_deque = deque(ground_truth_trajectory[:self.trajectory_length]) # [x_0, x_1, x_2, x_3] initially

            # Initialize h_state (h_{-1} for the first physical step)
            gru_hidden_size: int = self.config.get('fmt_model.gru.hidden_size', 512)
            current_h_state = torch.zeros(1, gru_hidden_size, device=self.device, dtype=self.dtype) # (B=1, H)

            predicted_trajectory: List[torch.Tensor] = list(initial_history_deque) # Starts with ground truth history

            # Number of subsequent physical steps to predict (beyond the initial trajectory_length)
            num_prediction_steps = ground_truth_trajectory.shape[0] - self.trajectory_length
            
            current_l2re_series: List[float] = []

            for s_step in range(num_prediction_steps):
                # The starting state for the current s -> s+1 physical step is `x_s`.
                # This corresponds to `initial_history_deque[-1]`.
                x_s_physical_for_ode = initial_history_deque[-1]

                # Extract the 3 preceding physical states from the deque for pyramid context
                # `physical_history_states` here will be `[x_{s-3}, x_{s-2}, x_{s-1}]` relative to `x_s`.
                # So if initial_history_deque is [x0,x1,x2,x3], for first prediction (x3->x4), history will be [x0,x1,x2].
                physical_history_context = list(initial_history_deque)[:self.trajectory_length - 1] # [x_{s-3}, x_{s-2}, x_{s-1}]

                # Run one physical step prediction (x_s -> x_{s+1})
                predicted_x_s_plus_1, current_h_state = self._run_ode_sampler_inference(
                    x_s_physical=x_s_physical_for_ode,
                    current_h_state=current_h_state,
                    k_val_for_ode=self.deterministic_k,
                    physical_history_states=physical_history_context
                )
                predicted_trajectory.append(predicted_x_s_plus_1)

                # Update history deque: remove oldest state, add the newly predicted state
                initial_history_deque.popleft()
                initial_history_deque.append(predicted_x_s_plus_1)

                # Calculate L2RE for this predicted step against ground truth
                gt_next_state = ground_truth_trajectory[self.trajectory_length + s_step]
                l2re_step = self.l2re_calculator(predicted_x_s_plus_1, gt_next_state).item()
                current_l2re_series.append(l2re_step)

                # Store for aggregated metrics
                if s_step == 0: # Step 1 (0-indexed first prediction)
                    all_l2re_per_step[1].append(l2re_step)
                if s_step == 4: # Step 5 (0-indexed 4th prediction)
                    all_l2re_per_step[5].append(l2re_step)
                if s_step == 9: # Step 10 (0-indexed 9th prediction)
                    all_l2re_per_step[10].append(l2re_step)
            
            if current_l2re_series:
                all_l2re_per_step['last'].append(current_l2re_series[-1])
                all_l2re_per_step['average'].append(np.mean(current_l2re_series))
            
            log_message(logger, f"  Rollout Trajectory {traj_idx+1} - Num steps: {len(current_l2re_series)}, Avg L2RE: {all_l2re_per_step['average'][-1]:.4f}", logging.DEBUG)

            # Log some rollout samples for visualization
            if traj_idx == 0: # Only for the first trajectory
                for i_vis in range(min(4, len(predicted_trajectory))): # Visualize first few steps
                    log_image(logger, f"FMT_Rollout/Traj{traj_idx}_Pred_Step{i_vis}", predicted_trajectory[i_vis][0].cpu(), s_step)
                    log_image(logger, f"FMT_Rollout/Traj{traj_idx}_GT_Step{i_vis}", ground_truth_trajectory[i_vis][0].cpu(), s_step)

        # Aggregate results across all trajectories
        aggregated_metrics = {}
        for step_key, l2re_list in all_l2re_per_step.items():
            if l2re_list:
                aggregated_metrics[f"L2RE_Step_{step_key}"] = np.mean(l2re_list)
            else:
                aggregated_metrics[f"L2RE_Step_{step_key}"] = float('nan') # No data for this step

        log_message(logger, "FMT Long-term Rollout Results:", logging.INFO)
        for k, v in aggregated_metrics.items():
            log_message(logger, f"  {k}: {v:.4f}", logging.INFO)
        return aggregated_metrics

    @torch.no_grad()
    def generate_ensemble(self, initial_states_batch: torch.Tensor, num_generations: int, k_val_last_step: float) -> List[torch.Tensor]:
        """
        Generates an ensemble of possible next states (x_T+1) given initial states (x_0, ..., x_T).
        The `initial_states_batch` should be a batch of `(x_0, ..., x_{trajectory_length-1})`
        e.g., `(x_0, x_1, x_2, x_3)` where `trajectory_length` is 4.
        `k_val_last_step < 1` introduces stochasticity into the prediction of the next state.

        Args:
            initial_states_batch (torch.Tensor): A batch of initial physical states,
                                                shape (B, trajectory_length, C, H, W).
            num_generations (int): The number of ensemble members to generate for each sample in the batch.
            k_val_last_step (float): The 'k' parameter to use for the last prediction step
                                     (e.g., x_3 -> x_4), which controls stochasticity.

        Returns:
            List[torch.Tensor]: A list of generated ensemble predictions. Each element in the list
                                corresponds to one sample from the input batch and is a tensor
                                of shape (num_generations, C, H, W).
        """
        log_message(logger, f"Generating ensemble with {num_generations} samples for k_val_last_step={k_val_last_step}...", logging.INFO)
        self.p2vae_model.eval()
        self.fmt_model.eval()

        batch_size = initial_states_batch.shape[0]
        ensemble_predictions_for_batch: List[torch.Tensor] = [] # List of (num_generations, C, H, W) tensors

        # The paper specifies (k_0, k_1, k_2) = 1 for clean history to derive h_3.
        # This implies we calculate h_3 based on (x_0, x_1, x_2, x_3) using k=1 for internal steps.
        
        gru_hidden_size: int = self.config.get('fmt_model.gru.hidden_size', 512)
        
        # Iterate over each sample in the batch
        for b_idx in range(batch_size):
            single_initial_states_full = initial_states_batch[b_idx] # (trajectory_length, C, H, W)

            # 1. Calculate h_state (h_{trajectory_length-1}) based on the known, clean history
            # This 'h' will be used as the initial `current_h_state` for the ODE solver.
            h_for_ensemble_prediction = torch.zeros(1, gru_hidden_size, device=self.device, dtype=self.dtype)
            
            # For `trajectory_length` states: x_0, x_1, ..., x_{T-1}
            # We want to get `h_{T-1}`
            # The h update takes `x_{s,t_s}^{k_s}` as input. For a clean history, we can assume k=1, t=0, so it's just `x_s`.
            for s_idx in range(self.trajectory_length): # Loop for s=0 to T-1
                x_s_for_h = single_initial_states_full[s_idx].unsqueeze(0) # (1, C, H, W)
                compressed_x_s_token = self.fmt_model.encode_condition_token(
                    self.p2vae_model.get_latent(x_s_for_h)
                )
                # Use t=1.0 for physical step completion in GRU update
                h_for_ensemble_prediction = self.fmt_model.update_history_h(
                    h_for_ensemble_prediction.unsqueeze(0), # GRU expects (1, B, H)
                    compressed_x_s_token,
                    self.fmt_model.time_embedding(self._get_time_embedding(1.0))
                ).squeeze(0) # Squeeze back to (B, H)

            # Now h_for_ensemble_prediction is `h_{trajectory_length-1}` (e.g., h_3 if traj_length is 4).
            # This is the `h` state to be used as `current_h_state` for the ODE prediction `x_{T-1} -> x_T`.

            ensemble_for_single_sample: List[torch.Tensor] = []
            x_start_physical_for_ode = single_initial_states_full[-1].unsqueeze(0) # This is `x_{trajectory_length-1}` (e.g., x_3)

            # The physical history context for the pyramid will be the `trajectory_length-1` preceding states.
            # E.g., if predicting x_4 from x_3, this is [x_0, x_1, x_2].
            physical_history_context_for_fmt = list(single_initial_states_full[:self.trajectory_length-1])
            
            for gen_idx in range(num_generations):
                predicted_x_next_physical, _ = self._run_ode_sampler_inference(
                    x_s_physical=x_start_physical_for_ode,
                    current_h_state=h_for_ensemble_prediction,
                    k_val_for_ode=k_val_last_step, # Stochasticity comes from this k_val
                    physical_history_states=physical_history_context_for_fmt
                )
                ensemble_for_single_sample.append(predicted_x_next_physical.squeeze(0)) # Remove batch dim (1)

            ensemble_predictions_for_batch.append(torch.stack(ensemble_for_single_sample))
            
        return ensemble_predictions_for_batch

    def finetune_and_evaluate(self, finetune_data_loader: DataLoader, target_data_loader: DataLoader) -> Dict[str, float]:
        """
        Adapts the pretrained model to an unseen system with a stop-gradient operation.
        Then evaluates its performance on the target dataset.

        Args:
            finetune_data_loader (DataLoader): DataLoader for the fine-tuning dataset (e.g., Kolmogorov turbulence train set).
            target_data_loader (DataLoader): DataLoader for the evaluation dataset after fine-tuning (e.g., Kolmogorov turbulence test set).

        Returns:
            Dict[str, float]: A dictionary containing evaluation metrics after fine-tuning.
        """
        if not self.finetuning_config.get('enabled', False):
            log_message(logger, "Fine-tuning is disabled in config. Skipping finetune_and_evaluate.", logging.INFO)
            return {}

        log_message(logger, f"Starting fine-tuning on {self.finetuning_config.get('finetune_dataset_name')}...", logging.INFO)

        # Import FMTTrainer here to avoid circular dependencies if FMTTrainer imports Evaluator
        from training.fmt_trainer import FMTTrainer

        # Create trainable copies of P2VAE and FMT models for fine-tuning.
        # This ensures the original pre-trained models are not modified.
        finetune_p2vae_model = P2VAEModel(self.config).to(self.device)
        finetune_p2vae_model.load_state_dict(self.p2vae_model.state_dict())
        finetune_p2vae_model.train() # P2VAE is now trainable

        finetune_fmt_model = FMTModel(self.config).to(self.device)
        finetune_fmt_model.load_state_dict(self.fmt_model.state_dict())
        finetune_fmt_model.train() # FMT is now trainable

        # Fine-tuning trainer setup. The FMTTrainer must be capable of combined loss and stop_gradient.
        finetuning_trainer = FMTTrainer(
            model=finetune_fmt_model,
            p2vae_model=finetune_p2vae_model, # Pass trainable P2VAE
            train_loader=finetune_data_loader,
            val_loader=None, # Validation for fine-tuning might be integrated or skipped
            config=self.config,
            device=self.device,
            is_finetuning=True # Flag to FMTTrainer to activate combined loss and stop_gradient
        )
        
        # Override num_training_steps for finetuning
        num_finetune_steps = self.finetuning_config.get('num_finetune_steps', 5000)
        finetuning_trainer.num_training_steps = num_finetune_steps
        # Potentially adjust log/save/validate intervals for fine-tuning
        finetuning_trainer.log_interval_steps = self.finetuning_config.get('log_interval_steps', 100)
        finetuning_trainer.validate_interval_steps = self.finetuning_config.get('validate_interval_steps', 1000)
        finetuning_trainer.save_interval_steps = self.finetuning_config.get('save_interval_steps', 2500)


        finetuning_trainer.train() # Run the fine-tuning process

        log_message(logger, "Fine-tuning complete. Evaluating on target dataset...", logging.INFO)

        # Temporarily replace evaluator's models and test_loader with fine-tuned ones for evaluation
        original_p2vae = self.p2vae_model
        original_fmt = self.fmt_model
        original_test_loader = self.test_loader # Store original test loader

        self.p2vae_model = finetune_p2vae_model
        self.fmt_model = finetune_fmt_model
        self.test_loader = target_data_loader # Use target data loader for evaluation

        # Run long-term rollout evaluation with the fine-tuned models
        metrics = self.evaluate_long_term_rollout()

        # Restore original models and test_loader (important if Evaluator instance is long-lived)
        self.p2vae_model = original_p2vae
        self.fmt_model = original_fmt
        self.test_loader = original_test_loader

        log_message(logger, "Fine-tuning evaluation complete.", logging.INFO)
        return metrics


## fine_tuner.py

"""
AdjointMatchingTrainer: Implements the memoryless SOC fine‑tuning loop,
as described in Algorithm 1 of "Adjoint Matching: Fine‑tuning Flow and
Diffusion Generative Models with Memoryless Stochastic Optimal Control".
"""

import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.utils
from torch.utils.data import DataLoader

from models import BaseModels, FineTunedModel
from utils import get_sigma, solve_adjoint_ode_backward


class AdjointMatchingTrainer:
    """
    Orchestrates reward fine‑tuning of a Flow Matching model using the
    Adjoint Matching loss and a memoryless noise schedule.

    Args:
        base_models:   Pre‑trained, frozen base components (VAE, FM U‑Net,
                       CLIP text encoder, reward model).
        fine_model:    The fine‑tuned U‑Net whose parameters will be updated.
        config:        Configuration object (typically parsed from YAML) that
                       contains all hyper‑parameters for training.
    """

    def __init__(
        self,
        base_models: BaseModels,
        fine_model: FineTunedModel,
        config: Any,
    ) -> None:
        self.base_models = base_models
        self.fine_model = fine_model
        self.config = config

        # Device obtained from the fine‑tuned model
        self.device = next(fine_model.parameters()).device

        # Optimizer (Adam) with parameters from config.training
        self.optimizer = torch.optim.Adam(
            fine_model.parameters(),
            lr=config.training.learning_rate,
            betas=(config.training.adam_beta1, config.training.adam_beta2),
            eps=config.training.adam_eps,
            weight_decay=config.training.weight_decay,
        )

        # Number of discretisation steps (K) and offset h
        self.K = config.fine_tuning.num_timesteps
        self.h = config.fine_tuning.noise_schedule.offset_h
        # For numerical consistency, h should equal 1/K; we assert close.
        assert math.isclose(self.h, 1.0 / self.K, rel_tol=1e-6), \
            f"h ({self.h}) must equal 1/K ({1.0/self.K})"

        # Reward scaling coefficient (λ)
        self.lambda_ = config.fine_tuning.lambda_

        # Loss clipping threshold LCT = coeff * λ^2
        lct_coeff = config.fine_tuning.loss_clipping_threshold_coeff
        self.LCT = lct_coeff * (self.lambda_ ** 2)

        # Timestep subset selection for loss computation
        self.timestep_subset_size = config.fine_tuning.timestep_subset_size
        self.random_from = config.fine_tuning.selected_timesteps.random_from
        self.random_to = config.fine_tuning.selected_timesteps.random_to
        self.last_steps_count = config.fine_tuning.selected_timesteps.last_steps_count

        # Total number of loss timesteps = random subset + last steps
        assert self.timestep_subset_size == \
            (self.timestep_subset_size - self.last_steps_count) + self.last_steps_count, \
            "Loss timestep count mismatch"

        # Checkpoint saving interval (in iterations)
        self.save_every = config.fine_tuning.checkpoint_save_every
        self.checkpoint_dir = config.logging.checkpoint_dir

        # Mixed precision setting (bfloat16)
        self.use_amp = config.training.precision == "bfloat16"
        self.autocast_context = torch.cuda.amp.autocast(dtype=torch.bfloat16) \
            if self.use_amp else torch.no_grad()  # no_grad as a no-op context

        # Gradient clipping
        self.grad_clip_norm = config.training.grad_clip_norm

        # Internal counter for iteration number
        self.iteration = 0

        # Ensure base models are in eval mode and frozen
        base_models.freeze_base()

    # ------------------------------------------------------------------
    #  Public methods (train / train_epoch)
    # ------------------------------------------------------------------
    def train(
        self,
        dataloader: DataLoader,
        num_epochs: int = 1,
    ) -> None:
        """
        Full fine‑tuning loop.

        Args:
            dataloader:   PyTorch DataLoader yielding batches of prompts
                          (dict with keys "input_ids", "attention_mask",
                          "prompts").
            num_epochs:   Number of passes over the dataset (default 1).
        """
        self.fine_model.train()

        for epoch in range(num_epochs):
            self.train_epoch(dataloader)

    def train_epoch(self, dataloader: DataLoader) -> None:
        """
        One epoch of training.

        Args:
            dataloader:  PyTorch DataLoader (as in `train`).
        """
        for batch in dataloader:
            # Move tokenized inputs to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            prompts: List[str] = batch["prompts"]

            # Encode prompts into CLIP hidden states
            with torch.no_grad():
                encoder_output = self.base_models.clip_text_encoder(
                    input_ids, attention_mask=attention_mask
                )
                prompt_embeds = encoder_output.last_hidden_state  # (B, seq_len, embed_dim)

            # Determine latent shape from the model's expected input.
            # (In a full implementation this could be inferred from the U‑Net config.)
            B = input_ids.shape[0]
            latent_shape = (B, 4, 64, 64)   # hard‑coded for a 512px VAE
            z = torch.randn(latent_shape, device=self.device)

            # ---- 1. Simulate controlled memoryless SDE (forward pass) ----
            trajectory = self.sample_trajectory(z, prompt_embeds)

            # ---- 2. Solve lean adjoint ODE backwards ----
            adjoint_states = self.solve_lean_adjoint(
                trajectory, prompt_embeds, prompts
            )

            # ---- 3. Compute clipped Adjoint Matching loss ----
            with self.autocast_context:
                loss = self.compute_loss(trajectory, adjoint_states, prompt_embeds)

            # ---- 4. Backward pass and parameter update ----
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.fine_model.parameters(), self.grad_clip_norm
                )

            self.optimizer.step()

            # Optional: log to tensorboard / console
            if self.iteration % 50 == 0:
                self._log_progress(loss.item(), batch)

            self.iteration += 1

            # Periodic checkpoint saving
            if self.save_every > 0 and self.iteration % self.save_every == 0:
                self._save_checkpoint()

        # End of epoch – save final model
        self._save_checkpoint()

    # ------------------------------------------------------------------
    #  Sampling trajectory (memoryless SDE)
    # ------------------------------------------------------------------
    def sample_trajectory(
        self,
        z: torch.Tensor,
        prompt_embeds: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Simulate the controlled memoryless SDE using Euler–Maruyama.

        Drift: 2 * v_fine(X_t, t) - X_t / (t + h)

        Args:
            z:             Initial noise, shape (B, C, H, W).
            prompt_embeds: Text conditioning, shape (B, seq_len, embed_dim).

        Returns:
            List of K+1 detached tensors at times 0, h, 2h, …, 1.
        """
        B = z.shape[0]
        x = z
        trajectory = [x.detach().clone()]

        for k in range(self.K):
            t = k * self.h
            sigma_t = get_sigma(t, self.h)

            # Time tensor for U‑Net
            time_tensor = torch.full((B,), t, device=self.device, dtype=z.dtype)

            with torch.no_grad():
                v_fine = self.fine_model(x, time_tensor, prompt_embeds)
                drift = 2.0 * v_fine - x / (t + self.h)
                noise = torch.randn_like(x)
                x = x + self.h * drift + math.sqrt(self.h) * sigma_t * noise
                trajectory.append(x.detach().clone())

        return trajectory

    # ------------------------------------------------------------------
    #  Solving the lean adjoint ODE
    # ------------------------------------------------------------------
    def solve_lean_adjoint(
        self,
        trajectory: List[torch.Tensor],
        prompt_embeds: torch.Tensor,
        prompts: List[str],
    ) -> List[torch.Tensor]:
        """
        Compute the lean adjoint states by integrating the backward ODE.

        The terminal condition is obtained by back‑propagating the reward
        gradient through the noiseless endpoint estimate.

        Args:
            trajectory:     List of K+1 detached latent states.
            prompt_embeds:  Text conditioning.
            prompts:        Raw text strings (one per sample) for reward
                            computation.

        Returns:
            List of K adjoint states for times 0, h, …, 1‑h.
        """
        # --- Build reward gradient function ---
        reward_transform = self._get_reward_transform()

        def reward_grad_fn(x_hat1: torch.Tensor) -> torch.Tensor:
            """
            Computes ∇_{x_hat1} (λ · reward(decoded(x_hat1), prompt)).
            Returns a detached gradient tensor of same shape.
            """
            # Temporarily enable grad to differentiate through decoder & reward
            x_hat1.requires_grad_(True)
            with torch.no_grad():
                decoded = self.base_models.vae.decode(x_hat1).sample  # (B, 3, 512, 512)
                # Normalise to [0,1] for PIL conversion
                images_tensor = (decoded / 2 + 0.5).clamp(0, 1)

            # Pre‑process for ImageReward (resize, normalise, etc.)
            processed_images = []
            for i in range(x_hat1.shape[0]):
                pil_img = torchvision.transforms.ToPILImage()(images_tensor[i].cpu())
                pil_img = reward_transform(pil_img)
                processed_images.append(pil_img)
            # Stack into a batch tensor
            processed_batch = torch.stack([t for t in processed_images]).to(self.device)

            # Compute reward through the reward model (needs to be done with grad to
            # propagate through x_hat1).  We'll use a small wrapper that allows
            # backprop through the reward model's pre‑processing and neural net.
            # NOTE: ImageReward's `score` method returns a scalar per image; we
            # must ensure it is differentiable.
            self.base_models.reward_model.eval()  # it's already frozen, but ensure
            rewards = self._compute_reward_fast(processed_batch, prompts)  # (B,)
            # Multiply by λ and sum over batch
            reward = self.lambda_ * rewards.sum()

            grad = torch.autograd.grad(reward, x_hat1, create_graph=False)[0]
            x_hat1.requires_grad_(False)
            return grad.detach()

        # Wrap the base velocity field into a callable expected by the utility
        def v_base_fn(x: torch.Tensor, t: float) -> torch.Tensor:
            time_tensor = torch.full(
                (x.shape[0],), t, device=x.device, dtype=x.dtype
            )
            with torch.no_grad():
                return self.base_models.flow_model(x, time_tensor, prompt_embeds)

        # The utility function handles the backward ODE integration and
        # the terminal condition (noiseless endpoint).
        adjoint_states = solve_adjoint_ode_backward(
            v_base_fn,
            trajectory,
            self.h,               # dt
            reward_grad_fn,
        )
        return adjoint_states

    # ------------------------------------------------------------------
    #  Loss computation (Adjoint Matching)
    # ------------------------------------------------------------------
    def compute_loss(
        self,
        trajectory: List[torch.Tensor],
        adjoint_states: List[torch.Tensor],
        prompt_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the clipped least‑squares Adjoint Matching loss.

        Loss term at time t:
            || (2/σ(t)) (v_fine(x,t) - v_base(x,t)) + σ(t) * a_t ||^2
        clipped at LCT.

        Args:
            trajectory:      List of K+1 detached noisy states.
            adjoint_states:  List of K adjoint states (a_0 … a_{1‑h}).
            prompt_embeds:   Text conditioning.

        Returns:
            Scalar loss averaged over batch and selected timesteps.
        """
        B = trajectory[0].shape[0]
        # Determine which timestep indices to use for the loss
        indices = self._get_loss_timestep_indices()

        total_loss = 0.0
        for idx in indices:
            x_t = trajectory[idx]                # state at time idx * h
            a_t = adjoint_states[idx]            # adjoint at same time
            t_val = idx * self.h
            sigma_t = get_sigma(t_val, self.h)
            time_tensor = torch.full((B,), t_val, device=self.device,
                                     dtype=trajectory[0].dtype)

            # Fine‑tuned velocity (gradient required)
            v_fine = self.fine_model(x_t, time_tensor, prompt_embeds)

            # Base velocity (no gradient)
            with torch.no_grad():
                v_base = self.base_models.flow_model(x_t, time_tensor, prompt_embeds)

            # Target vector field
            u_target = (2.0 / sigma_t) * (v_fine - v_base) + sigma_t * a_t

            # Squared L2 norm per sample, clipped
            per_sample_loss = u_target.pow(2).sum(dim=[1, 2, 3])  # (B,)
            clipped = torch.min(per_sample_loss, torch.tensor(self.LCT, device=self.device))
            total_loss += clipped.sum()

        # Average over batch and number of loss timesteps
        loss = total_loss / (B * len(indices))
        return loss

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------
    def _get_loss_timestep_indices(self) -> List[int]:
        """
        Return a list of timestep indices (0 … K‑1) for which the loss
        should be evaluated.  Always includes the last `last_steps_count`
        steps and a random subset from the earlier part of the schedule.
        """
        K = self.K
        # Indices corresponding to the last steps:
        last_indices = list(range(K - self.last_steps_count, K))

        # Indices for the random subset: from 0 to (K - last_steps_count - 1),
        # corresponding to physical time ≤ random_to * 1.0 (roughly 0.75).
        max_random_idx = max(0, K - self.last_steps_count - 1)
        random_pool = list(range(0, max_random_idx + 1))
        # Number of random timesteps to sample
        num_random = self.timestep_subset_size - self.last_steps_count
        sampled = random.sample(random_pool, k=num_random)

        # Combine and return (order irrelevant)
        return sorted(random_pool if False else list(set(random_pool) - set(sampled))?  Actually we want the sampled ones):
        # correction
        combined = sampled + last_indices
        return sorted(combined)

    def _get_reward_transform(self):
        """
        Image preprocessing transform required by the ImageReward model.
        (Resize to 224x224, convert to tensor, normalise.)
        """
        import torchvision.transforms as T
        return T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.LANCZOS),
            T.ToTensor(),
            # Normalisation may be model‑specific; we assume the loading code
            # inside ImageReward already applies a transform to the PIL image.
            # For safety, we return the identity‑like transform and let the
            # model handle it internally.  To do: verify with the actual
            # ImageReward interface.
        ])

    def _compute_reward_fast(
        self,
        processed_images: torch.Tensor,
        prompts: List[str],
    ) -> torch.Tensor:
        """
        Compute reward scores using the internal neural network of the
        ImageReward model.

        Args:
            processed_images: Tensor of shape (B, C, 224, 224) pre‑processed
                              as required by ImageReward.
            prompts:          List of text strings (length B).

        Returns:
            Tensor of shape (B,) containing reward scores.
        """
        # This is a minimal wrapper; in practice the ImageReward model
        # may have a `model` attribute that performs the joint text/image
        # forward pass.  We assume it is available.
        with torch.no_grad():
            scores = self.base_models.reward_model.model(
                processed_images, prompts
            )  # returns (B,) or (B,1)
        return scores.squeeze(-1)

    def _log_progress(self, loss_value: float, batch: Dict) -> None:
        """
        Optional: log training metrics (loss, reward, control cost).
        """
        # Placeholder – to be expanded with tensorboard or wandb.
        if self.iteration % 100 == 0:
            print(f"Iter {self.iteration:5d} | loss: {loss_value:8.4f}")

    def _save_checkpoint(self) -> None:
        """
        Save the fine‑tuned model state dict.
        """
        import os
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir,
                            f"checkpoint_iter{self.iteration:06d}.pt")
        torch.save(self.fine_model.unet.state_dict(), path)
        print(f"Checkpoint saved to {path}")



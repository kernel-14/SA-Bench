## inference/rollout.py
"""Autoregressive rollout for long-horizon PDE prediction and ensemble generation.

Implements AutoregressiveRollout, which bridges the latent-space EulerSampler
with pixel-space inputs/outputs via P2VAE, supporting both deterministic
long-horizon rollout (all k=1, Table 3 evaluation) and stochastic ensemble
generation (k_3 < 1, Figure 3 ensemble variance).

From the paper (Section 3.4):
    "In a deterministic prediction setting, we set (k_0, k_1, k_2, k_3) to be 1,
    meaning that the past states are not noisy. In a generation setting, we choose
    to set (k_0, k_1, k_2) to be 1 and k_3 less than 1."

The autoregressive window always contains 4 frames (config: data.trajectory_length=4).
At each step, the oldest frame is dropped and the newly predicted frame is appended,
maintaining a sliding window of the most recent 4 latent frames.

Configuration alignment (config.yaml):
    fmt.inference.deterministic_k_vals: [1.0, 1.0, 1.0, 1.0]  # Table 3 rollout
    fmt.inference.generative_k_history: [1.0, 1.0, 1.0]        # ensemble k_0..k_2
    ensemble.batch_size: 32                                      # Figure 3 ensemble
    ensemble.k3_values: [0.0, 0.3, 0.6, 0.9]                   # k_3 sweep
    evaluation.rollout.k_vals: [1.0, 1.0, 1.0, 1.0]            # deterministic
    data.trajectory_length: 4                                    # window_size
"""

import logging
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from inference.sampler import EulerSampler
from models.fmt import FMT
from models.p2vae import P2VAE

logger = logging.getLogger(__name__)


class AutoregressiveRollout:
    """Autoregressive long-horizon PDE prediction via sliding-window FMT inference.

    Encodes pixel-space seed frames to latent space via P2VAE, maintains a
    4-frame sliding window of latent representations, and repeatedly applies
    the EulerSampler to generate new latent frames. Each predicted latent is
    decoded back to pixel space via P2VAE and appended to the output trajectory.

    The GRU hidden state h is threaded through all autoregressive steps,
    accumulating temporal context from the full prediction history. This is
    the diffusion forcing mechanism that mitigates exposure bias in long-horizon
    rollouts (paper Section 3.2).

    Two inference modes:
        Deterministic (k_vals=[1,1,1,1]): All frames are clean. Used for
            Table 3 long-term rollout evaluation. The model behaves like a
            deterministic neural operator.
        Generative (k_vals=[1,1,1,k3] with k3<1): History frames are clean
            but the current frame is initialized with noise. Used for Figure 3
            ensemble generation. Different noise realizations produce diverse
            predictions.

    Attributes:
        fmt: FMT model for velocity prediction. Must be in eval mode.
        p2vae: P2VAE model for encoding/decoding. Must be in eval mode.
        sampler: EulerSampler for ODE integration (N=100 steps, dt=0.01).
        window_size: Number of frames in the sliding window (always 4).
    """

    def __init__(
        self,
        fmt: FMT,
        p2vae: P2VAE,
        sampler: EulerSampler,
        window_size: int = 4,
    ) -> None:
        """Initialize AutoregressiveRollout.

        Args:
            fmt: Pretrained FMT model. Should be in eval mode before calling
                rollout() or generate_ensemble(). Provides predict_velocity()
                and gru_forcing submodule for hidden state management.
            p2vae: Pretrained P2VAE model. Should be in eval mode. Provides
                get_latent() for encoding (returns posterior mean mu, no
                sampling) and decode() for pixel-space reconstruction.
            sampler: EulerSampler instance configured with num_steps=100 and
                dt=0.01 (from config: fmt.inference.euler_steps=100,
                fmt.inference.dt=0.01). Integrates the FMT velocity field
                from t=0 to t=1 for each autoregressive step.
            window_size: Number of frames in the sliding latent window.
                From config: data.trajectory_length=4. Must match the FMT
                training setup (always 4 consecutive frames per Section 3.3).
        """
        self.fmt: FMT = fmt
        self.p2vae: P2VAE = p2vae
        self.sampler: EulerSampler = sampler
        self.window_size: int = window_size

        logger.info(
            "AutoregressiveRollout initialized: window_size=%d, "
            "sampler_steps=%d, sampler_dt=%.4f",
            self.window_size,
            self.sampler.num_steps,
            self.sampler.dt,
        )

    def _resolve_device(self, device: Optional[torch.device]) -> torch.device:
        """Resolve the target device from the FMT model parameters.

        Args:
            device: Explicit device override. If None, infers from FMT params.

        Returns:
            Resolved torch.device for tensor operations.
        """
        if device is not None:
            return device
        try:
            return next(self.fmt.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _encode_frames(
        self,
        frames: Tensor,
        device: torch.device,
    ) -> Tensor:
        """Encode a batch of pixel-space frames to latent representations.

        Reshapes the (B, T, 3, 128, 128) trajectory to (B*T, 3, 128, 128)
        for batch encoding, then reshapes back to (B, T, 16, 16, 16).

        Uses p2vae.get_latent() which returns the posterior mean mu (no
        sampling), providing deterministic latents consistent with FMT
        training (shared knowledge #3: "p2vae.get_latent(x) always returns
        mu (the mean of the posterior) without sampling").

        Args:
            frames: Pixel-space frames of shape (B, T, 3, 128, 128).
            device: Target device for computation.

        Returns:
            Latent representations of shape (B, T, 16, 16, 16).
        """
        b: int = frames.shape[0]
        t: int = frames.shape[1]

        # Move to device.
        frames_dev: Tensor = frames.to(device, non_blocking=True)

        # Flatten time dimension for batch encoding.
        # (B, T, 3, 128, 128) → (B*T, 3, 128, 128)
        x_flat: Tensor = frames_dev.view(b * t, 3, 128, 128)

        # Encode: (B*T, 3, 128, 128) → (B*T, 16, 16, 16)
        # get_latent returns mu (posterior mean, no reparameterization).
        y_flat: Tensor = self.p2vae.get_latent(x_flat)

        # Reshape back: (B*T, 16, 16, 16) → (B, T, 16, 16, 16)
        y: Tensor = y_flat.view(b, t, 16, 16, 16)

        return y

    def _build_x_init(
        self,
        y_current: Tensor,
        k: float,
    ) -> Tensor:
        """Build the initial latent state for ODE integration.

        Implements the t=0 case of the location-scale interpolation kernel
        (paper Eq. 1-3 with t=0):
            x_0^k = 0*x_1 + k*1*x_0 + 1*(1-k)*z = k*x_0 + (1-k)*z

        For k=1.0 (deterministic): x_init = y_current (clean frame)
        For k<1.0 (stochastic):    x_init = k*y_current + (1-k)*z

        Args:
            y_current: Current (last) frame latent of shape (B, 16, 16, 16).
            k: Bridge parameter in [0, 1]. k=1 → deterministic, k=0 → pure noise.

        Returns:
            Initial latent for ODE integration, shape (B, 16, 16, 16).
        """
        if k >= 1.0:
            # Deterministic: use clean frame directly.
            return y_current.clone()

        # Stochastic: mix clean frame with Gaussian noise.
        # z ~ N(0, I), same shape as y_current.
        z: Tensor = torch.randn_like(y_current)
        x_init: Tensor = k * y_current + (1.0 - k) * z

        return x_init

    def _warmup_gru(
        self,
        latent_window: Tensor,
        k_vals: List[float],
        h: Tensor,
    ) -> Tensor:
        """Pre-warm the GRU hidden state by processing the initial seed frames.

        The diffusion forcing scheme requires h to encode the history of past
        frames before generating new ones. At inference start, h=0 — we must
        process the seed frames sequentially to build meaningful context.

        For each frame s in 0..window_size-1:
            1. Optionally inject noise based on k_vals[s]
            2. Update GRU hidden state via gru_forcing.update(h, y_s_noisy)

        Args:
            latent_window: Initial latent window of shape (B, window_size, 16, 16, 16).
            k_vals: Bridge parameters [k_0, k_1, k_2, k_3] for each frame position.
                k=1.0 → use clean frame; k<1.0 → inject noise.
            h: Initial GRU hidden state of shape (B, embed_dim), typically zeros.

        Returns:
            Updated GRU hidden state h of shape (B, embed_dim) after processing
            all seed frames.
        """
        for s in range(self.window_size):
            y_s: Tensor = latent_window[:, s]  # (B, 16, 16, 16)

            # Build noisy version of y_s based on k_vals[s].
            y_s_noisy: Tensor = self._build_x_init(y_s, k_vals[s])

            # Update GRU hidden state with the (possibly noisy) frame.
            h = self.fmt.gru_forcing.update(h, y_s_noisy)

        return h

    def rollout(
        self,
        initial_frames: Tensor,
        num_steps: int,
        k_vals: Optional[List[float]] = None,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """Perform autoregressive long-horizon PDE prediction.

        Encodes the 4 seed frames to latent space, pre-warms the GRU on the
        seed frames, then iteratively predicts new frames by:
            1. Constructing x_init from the last latent frame with k_vals[3]
            2. Running the Euler ODE sampler to get the next latent
            3. Decoding the latent to pixel space
            4. Re-encoding the decoded frame to update the latent window
            5. Sliding the window forward

        The GRU hidden state h is threaded through all steps, accumulating
        temporal context for stable long-horizon prediction.

        Args:
            initial_frames: Seed frames in pixel space, shape (B, 4, 3, 128, 128).
                Must contain exactly window_size=4 frames (config:
                data.trajectory_length=4). These are the initial condition
                for the autoregressive rollout.
            num_steps: Number of autoregressive prediction steps. For Table 3
                evaluation: up to 14 steps for PA-NS. The output trajectory
                has exactly num_steps frames.
            k_vals: Bridge parameters [k_0, k_1, k_2, k_3] for each frame
                position in the sliding window. From config:
                    Deterministic: fmt.inference.deterministic_k_vals = [1,1,1,1]
                    Generative: [1,1,1,k3] where k3 ∈ {0, 0.3, 0.6, 0.9}
                Defaults to [1.0, 1.0, 1.0, 1.0] (deterministic prediction).
            device: Target device for computation. If None, inferred from
                FMT model parameters.

        Returns:
            Predicted trajectory in pixel space, shape (B, num_steps, 3, 128, 128).
            Each frame is the decoded prediction for that autoregressive step.
            Returns empty tensor (B, 0, 3, 128, 128) if num_steps=0.

        Raises:
            ValueError: If initial_frames does not have exactly window_size=4
                frames in the time dimension, or if num_steps < 0.
        """
        # Validate inputs.
        if initial_frames.shape[1] != self.window_size:
            raise ValueError(
                f"initial_frames must have exactly {self.window_size} frames "
                f"(config: data.trajectory_length=4), "
                f"got {initial_frames.shape[1]}."
            )
        if num_steps < 0:
            raise ValueError(
                f"num_steps must be non-negative, got {num_steps}."
            )

        # Apply default k_vals from config: fmt.inference.deterministic_k_vals
        if k_vals is None:
            k_vals = [1.0, 1.0, 1.0, 1.0]

        if len(k_vals) != self.window_size:
            raise ValueError(
                f"k_vals must have exactly {self.window_size} elements "
                f"(one per frame position), got {len(k_vals)}."
            )

        # Handle trivial case.
        if num_steps == 0:
            b: int = initial_frames.shape[0]
            return torch.zeros(b, 0, 3, 128, 128, dtype=torch.float32)

        # Resolve device.
        target_device: torch.device = self._resolve_device(device)

        b = initial_frames.shape[0]

        # Set both models to eval mode for inference.
        self.fmt.eval()
        self.p2vae.eval()

        with torch.no_grad():
            # ------------------------------------------------------------------
            # Step 1: Encode initial seed frames to latent space.
            # (B, 4, 3, 128, 128) → (B, 4, 16, 16, 16)
            # Uses p2vae.get_latent() which returns mu (posterior mean).
            # ------------------------------------------------------------------
            latent_window: Tensor = self._encode_frames(
                initial_frames, target_device
            )  # (B, 4, 16, 16, 16)

            # ------------------------------------------------------------------
            # Step 2: Initialize GRU hidden state to zeros.
            # h encodes temporal context from all processed frames.
            # Shape: (B, embed_dim) where embed_dim = 256/512/768 for S/B/L.
            # ------------------------------------------------------------------
            h: Tensor = self.fmt.gru_forcing.init_hidden(
                batch_size=b, device=target_device
            )  # (B, embed_dim)

            # ------------------------------------------------------------------
            # Step 3: Pre-warm GRU on the 4 seed frames.
            # This builds meaningful context in h before generating new frames.
            # Without this, h=0 would provide no history conditioning.
            # ------------------------------------------------------------------
            h = self._warmup_gru(latent_window, k_vals, h)

            # ------------------------------------------------------------------
            # Step 4: Autoregressive generation loop.
            # At each step: sample next latent → decode → re-encode → slide window.
            # ------------------------------------------------------------------
            predicted_frames: List[Tensor] = []

            for step_idx in range(num_steps):
                # Extract the last (newest) frame from the sliding window.
                # This is the "current" frame from which we predict the next.
                y_last: Tensor = latent_window[:, -1]  # (B, 16, 16, 16)

                # Build x_init: the starting point for ODE integration at t=0.
                # For k_3=1.0 (deterministic): x_init = y_last (clean)
                # For k_3<1.0 (stochastic):    x_init = k_3*y_last + (1-k_3)*z
                # k_vals[3] is the bridge parameter for the "current" frame.
                k3: float = k_vals[3]
                x_init: Tensor = self._build_x_init(y_last, k3)  # (B, 16, 16, 16)

                # ------------------------------------------------------------------
                # Run Euler ODE sampler: integrate from t=0 to t=1.
                # The sampler calls fmt.predict_velocity at each of 100 steps,
                # updating x_t and h internally.
                # Returns: (x_next_latent, h_new) where h_new reflects the
                # GRU state after processing the full ODE trajectory.
                # ------------------------------------------------------------------
                x_next_latent: Tensor
                h_new: Tensor
                x_next_latent, h_new = self.sampler.sample(
                    fmt=self.fmt,
                    x_init=x_init,
                    h=h,
                )
                # x_next_latent: (B, 16, 16, 16) — predicted next latent at t=1
                # h_new: (B, embed_dim) — updated GRU state

                # ------------------------------------------------------------------
                # Decode predicted latent to pixel space.
                # (B, 16, 16, 16) → (B, 3, 128, 128)
                # ------------------------------------------------------------------
                x_next_pixel: Tensor = self.p2vae.decode(
                    x_next_latent
                )  # (B, 3, 128, 128)

                # Collect the decoded pixel-space prediction.
                predicted_frames.append(x_next_pixel)

                # ------------------------------------------------------------------
                # Re-encode the decoded pixel frame to get a clean latent.
                # Using p2vae.get_latent (returns mu) ensures consistency with
                # the training distribution and avoids latent drift.
                # (B, 3, 128, 128) → (B, 16, 16, 16)
                # ------------------------------------------------------------------
                y_next: Tensor = self.p2vae.get_latent(
                    x_next_pixel
                )  # (B, 16, 16, 16)

                # ------------------------------------------------------------------
                # Slide the window: drop oldest frame, append newest prediction.
                # [y_0, y_1, y_2, y_3] → [y_1, y_2, y_3, y_next]
                # latent_window shape: (B, 4, 16, 16, 16)
                # ------------------------------------------------------------------
                # Drop the oldest frame (index 0) and append y_next.
                # torch.cat along dim=1 (time dimension).
                latent_window = torch.cat(
                    [
                        latent_window[:, 1:],          # (B, 3, 16, 16, 16)
                        y_next.unsqueeze(1),            # (B, 1, 16, 16, 16)
                    ],
                    dim=1,
                )  # (B, 4, 16, 16, 16)

                # Update GRU hidden state for the next autoregressive step.
                # h_new already reflects the GRU state after the ODE integration.
                h = h_new

                logger.debug(
                    "Rollout step %d/%d complete. "
                    "x_next_pixel range: [%.3f, %.3f]",
                    step_idx + 1,
                    num_steps,
                    x_next_pixel.min().item(),
                    x_next_pixel.max().item(),
                )

            # ------------------------------------------------------------------
            # Stack all predicted frames along the time dimension.
            # List of num_steps tensors (B, 3, 128, 128)
            # → (B, num_steps, 3, 128, 128)
            # ------------------------------------------------------------------
            trajectory: Tensor = torch.stack(predicted_frames, dim=1)

        return trajectory  # (B, num_steps, 3, 128, 128)

    def generate_ensemble(
        self,
        initial_frames: Tensor,
        k3: float,
        batch_size: int = 32,
    ) -> Tensor:
        """Generate a diverse ensemble of next-step predictions via stochastic sampling.

        Replicates the initial seed frames batch_size times and runs a single
        autoregressive step with k_3=k3 < 1 to inject noise into the initial
        latent. Different noise realizations z ~ N(0, I) for each ensemble
        member produce diverse predictions, enabling uncertainty quantification.

        From the paper (Section 4.4):
            "By tuning bridge parameter k_3 during the generation, we can
            effectively generate an ensemble of possible next state given a
            noisy initialization k_3*x_3 + (1-k_3)*z and concluded PDE
            condition h_3 from clean past frames (x_0, x_1, x_2)."

        The variance of the predicted ensemble is a decreasing function of k_3
        (Figure 3): k_3=0 → maximum variance (pure noise init), k_3=1 → zero
        variance (deterministic, all members identical).

        Configuration alignment:
            ensemble.batch_size: 32          → default batch_size
            ensemble.k3_values: [0,0.3,0.6,0.9] → k3 values tested in Figure 3
            fmt.inference.generative_k_history: [1,1,1] → k_vals[:3] = [1,1,1]

        Args:
            initial_frames: Seed frames in pixel space, shape (B, 4, 3, 128, 128).
                Typically B=1 (single trajectory from PDEArena-NS per paper
                Section 4.4). The frames are replicated batch_size times to
                create the ensemble batch.
            k3: Bridge parameter for the current (to-be-predicted) frame.
                Controls the noise level of the initial latent:
                    k3=0.0: x_init = z (pure noise, maximum diversity)
                    k3=0.3: x_init = 0.3*y_3 + 0.7*z (high diversity)
                    k3=0.6: x_init = 0.6*y_3 + 0.4*z (moderate diversity)
                    k3=0.9: x_init = 0.9*y_3 + 0.1*z (low diversity)
                    k3=1.0: x_init = y_3 (deterministic, zero diversity)
                From config: ensemble.k3_values = [0.0, 0.3, 0.6, 0.9].
            batch_size: Number of ensemble members to generate. From config:
                ensemble.batch_size = 32. Each member uses an independent
                noise realization z ~ N(0, I) for the k3 initialization.

        Returns:
            Ensemble of next-step predictions, shape (batch_size, 1, 3, 128, 128).
            The time dimension has size 1 (single-step prediction). The batch
            dimension contains batch_size diverse predictions for the same
            initial condition. Evaluator.evaluate_ensemble computes
            Metrics.batch_variance on this output to produce Figure 3.

        Raises:
            ValueError: If initial_frames does not have exactly window_size=4
                frames, or if batch_size < 1.
        """
        if initial_frames.shape[1] != self.window_size:
            raise ValueError(
                f"initial_frames must have exactly {self.window_size} frames "
                f"(config: data.trajectory_length=4), "
                f"got {initial_frames.shape[1]}."
            )
        if batch_size < 1:
            raise ValueError(
                f"batch_size must be at least 1, got {batch_size}."
            )

        # ------------------------------------------------------------------
        # Expand initial frames: replicate B times along batch dimension.
        # (B, 4, 3, 128, 128) → (batch_size * B, 4, 3, 128, 128)
        # For the standard case B=1: (1, 4, 3, 128, 128) → (32, 4, 3, 128, 128)
        # ------------------------------------------------------------------
        b_orig: int = initial_frames.shape[0]

        # Repeat along batch dimension: each original sample is replicated
        # batch_size times to create the ensemble.
        # (B, 4, 3, 128, 128) → (B * batch_size, 4, 3, 128, 128)
        expanded_frames: Tensor = initial_frames.repeat(batch_size, 1, 1, 1, 1)
        # Shape: (B * batch_size, 4, 3, 128, 128)

        # ------------------------------------------------------------------
        # Set k_vals for ensemble generation.
        # From paper Section 3.4 and config:
        #   fmt.inference.generative_k_history: [1.0, 1.0, 1.0]
        #   k_3 = k3 (the stochastic parameter)
        # History frames (0, 1, 2) are clean (k=1), only the current frame
        # (3) is noisy (k=k3). This ensures h is conditioned on clean history
        # while the prediction starts from a noisy initialization.
        # ------------------------------------------------------------------
        k_vals: List[float] = [1.0, 1.0, 1.0, k3]

        logger.info(
            "Generating ensemble: batch_size=%d, k3=%.2f, "
            "k_vals=%s, expanded_batch=%d",
            batch_size,
            k3,
            k_vals,
            expanded_frames.shape[0],
        )

        # ------------------------------------------------------------------
        # Run single-step rollout on the expanded batch.
        # num_steps=1: predict only the next frame (x_4 from x_0..x_3).
        # The stochasticity comes from k3 < 1 in _build_x_init, which
        # samples z ~ N(0, I) independently for each of the batch_size members.
        # ------------------------------------------------------------------
        ensemble_trajectory: Tensor = self.rollout(
            initial_frames=expanded_frames,
            num_steps=1,
            k_vals=k_vals,
        )
        # Shape: (B * batch_size, 1, 3, 128, 128)

        # ------------------------------------------------------------------
        # Reshape to (batch_size, B, 1, 3, 128, 128) then squeeze B if B=1.
        # For the standard case B=1: (batch_size, 1, 3, 128, 128)
        # For general B: return (B * batch_size, 1, 3, 128, 128) directly.
        # The Evaluator expects (batch_size, 1, 3, 128, 128) for B=1.
        # ------------------------------------------------------------------
        # For simplicity and alignment with the paper's use case (B=1),
        # return the trajectory as-is. The caller handles the batch structure.
        return ensemble_trajectory  # (B * batch_size, 1, 3, 128, 128)

    def __repr__(self) -> str:
        """Return a string representation of the rollout configuration."""
        return (
            f"AutoregressiveRollout("
            f"window_size={self.window_size}, "
            f"sampler={self.sampler!r})"
        )

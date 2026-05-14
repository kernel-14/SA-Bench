## diffusion/ddpm.py
"""DDPM noise schedule and forward/reverse process utilities for PGR.

Implements the DDPMScheduler class that encapsulates the DDPM noise schedule
mathematics used by ConditionalDiffusion for both training (add_noise) and
sampling (step). All schedule tensors are precomputed in __init__ and stored
as plain float32 tensors on the target device.

This class holds no learnable parameters and is not an nn.Module. It is a
stateless utility consumed exclusively by ConditionalDiffusion.

Config references (config.yaml):
    diffusion.num_timesteps: 100    # DDPM denoising steps
    diffusion.beta_start:    1e-4   # linear beta schedule start
    diffusion.beta_end:      0.02   # linear beta schedule end
    hardware.device:         "cuda" # target device for schedule tensors
"""

import torch
import torch.nn.functional as F


class DDPMScheduler:
    """DDPM noise schedule and forward/reverse process utilities.

    Precomputes all schedule tensors (betas, alphas, cumulative products,
    posterior variance, etc.) in __init__ and stores them as float32 tensors
    on the target device. Provides three methods consumed by ConditionalDiffusion:

        - add_noise:        Forward process q(x_t | x_0) for training.
        - step:             One reverse denoising step p(x_{t-1} | x_t).
        - sample_timesteps: Uniform timestep sampling for training batches.

    All schedule tensors use 0-indexed timesteps [0, num_timesteps-1] for
    PyTorch compatibility, consistent with the paper's 1-indexed notation
    (n ~ Unif(1, N)) up to an index offset that has no effect on learning.

    Attributes:
        num_timesteps: Total number of diffusion steps T. Corresponds to
            config.diffusion.num_timesteps (default 100).
        device: PyTorch device string for all schedule tensors. Corresponds
            to config.hardware.device (default "cuda").
        betas: Linear beta schedule, shape (T,). Values in [beta_start, beta_end].
        alphas: 1 - betas, shape (T,).
        alphas_cumprod: Cumulative product of alphas (ᾱ_t), shape (T,).
        alphas_cumprod_prev: ᾱ_{t-1} with ᾱ_0 = 1.0, shape (T,).
        sqrt_alphas_cumprod: sqrt(ᾱ_t), shape (T,). Used in add_noise.
        sqrt_one_minus_alphas_cumprod: sqrt(1 - ᾱ_t), shape (T,). Used in
            add_noise and step.
        posterior_variance: β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t), shape (T,).
            Zero at t=0 (no noise at the final denoising step).
    """

    def __init__(
        self,
        num_timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str = "cuda",
    ) -> None:
        """Precomputes and stores all DDPM schedule tensors.

        Implements the linear beta schedule from Ho et al. (2020) with the
        standard DDPM closed-form forward process and posterior variance
        computation. All tensors are stored as float32 on the target device
        without gradient tracking.

        Args:
            num_timesteps: Total number of diffusion steps T. Corresponds to
                config.diffusion.num_timesteps (default 100). The reverse
                process iterates from t=T-1 down to t=0 in ConditionalDiffusion.
            beta_start: Starting value of the linear beta schedule. Corresponds
                to config.diffusion.beta_start (default 1e-4). Small value
                ensures minimal noise is added at the first step.
            beta_end: Ending value of the linear beta schedule. Corresponds to
                config.diffusion.beta_end (default 0.02). Controls the maximum
                noise level at the final forward step.
            device: PyTorch device string for all precomputed tensors.
                Corresponds to config.hardware.device (default "cuda").
                Must match the device used by ConditionalDiffusion for input
                tensors to avoid device mismatch errors in add_noise and step.
        """
        self.num_timesteps: int = num_timesteps
        self.device: str = device

        # ── Linear beta schedule ──────────────────────────────────────────────
        # β_t = linspace(beta_start, beta_end, T), shape: (T,)
        # Values increase linearly from beta_start to beta_end.
        # Corresponds to config.diffusion.beta_start=1e-4, beta_end=0.02.
        betas: torch.Tensor = torch.linspace(
            beta_start,
            beta_end,
            num_timesteps,
            dtype=torch.float32,
            device=device,
        )

        # ── Alpha schedule ────────────────────────────────────────────────────
        # α_t = 1 - β_t, shape: (T,)
        # Represents the fraction of signal retained at each step.
        alphas: torch.Tensor = 1.0 - betas

        # ── Cumulative product of alphas ──────────────────────────────────────
        # ᾱ_t = ∏_{i=1}^{t} α_i = cumprod(alphas), shape: (T,)
        # ᾱ_t controls the signal-to-noise ratio at timestep t:
        #   x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε
        alphas_cumprod: torch.Tensor = torch.cumprod(alphas, dim=0)

        # ── Previous cumulative product (ᾱ_{t-1}) ────────────────────────────
        # ᾱ_{t-1} is needed for the posterior variance computation.
        # Construction: drop the last element of alphas_cumprod and prepend 1.0.
        # This gives ᾱ_0 = 1.0 (no noise at step 0), ᾱ_1 = alphas_cumprod[0], etc.
        # F.pad(tensor, (left_pad, right_pad), value=pad_value) pads a 1D tensor.
        alphas_cumprod_prev: torch.Tensor = F.pad(
            alphas_cumprod[:-1],  # Drop last element: shape (T-1,)
            (1, 0),               # Pad 1 element on the left, 0 on the right
            value=1.0,            # ᾱ_0 = 1.0 (no noise before the process starts)
        )  # Shape: (T,)

        # ── Derived quantities for add_noise ─────────────────────────────────
        # sqrt(ᾱ_t): scales the clean signal x_0 in the forward process.
        sqrt_alphas_cumprod: torch.Tensor = torch.sqrt(alphas_cumprod)

        # sqrt(1 - ᾱ_t): scales the noise ε in the forward process.
        # Also used in the reverse step to compute the posterior mean.
        sqrt_one_minus_alphas_cumprod: torch.Tensor = torch.sqrt(
            1.0 - alphas_cumprod
        )

        # ── Posterior variance ────────────────────────────────────────────────
        # σ_t² = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        # This is the variance of the reverse process posterior q(x_{t-1} | x_t, x_0).
        # At t=0: alphas_cumprod_prev[0] = 1.0, so (1 - 1.0) = 0 → σ_0² = 0.
        # This correctly means no noise is added at the final denoising step.
        # The division is safe because the numerator is 0 when the denominator
        # approaches 0 (at t=0, 1 - alphas_cumprod[0] is small but nonzero,
        # and the numerator 1 - alphas_cumprod_prev[0] = 0 cancels it).
        posterior_variance: torch.Tensor = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # ── Store all schedule tensors as instance attributes ─────────────────
        # Stored as plain tensors (not nn.Parameters, not registered buffers)
        # since DDPMScheduler is not an nn.Module. All tensors are float32 on
        # self.device and do not require gradients.
        self.betas: torch.Tensor = betas
        self.alphas: torch.Tensor = alphas
        self.alphas_cumprod: torch.Tensor = alphas_cumprod
        self.alphas_cumprod_prev: torch.Tensor = alphas_cumprod_prev
        self.sqrt_alphas_cumprod: torch.Tensor = sqrt_alphas_cumprod
        self.sqrt_one_minus_alphas_cumprod: torch.Tensor = sqrt_one_minus_alphas_cumprod
        self.posterior_variance: torch.Tensor = posterior_variance

    # ── Public API ────────────────────────────────────────────────────────────

    def add_noise(
        self,
        x0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Implements the forward process q(x_t | x_0) for DDPM training.

        Applies the closed-form forward process reparameterization:
            x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε

        This is called inside ConditionalDiffusion.train_step() to corrupt
        clean normalized transitions x_0 before passing them to the denoising
        network for noise prediction.

        The batched indexing pattern uses per-sample timestep values from t
        to gather the appropriate schedule coefficients, then reshapes them
        to (B, 1) for broadcasting against (B, input_dim) tensors.

        Args:
            x0: Clean (normalized) transition tuples. Float32 tensor of shape
                (B, input_dim) where input_dim = 2*obs_dim + action_dim + 1.
                Must be on self.device. Produced by ConditionalDiffusion after
                normalizing the concatenated (s, a, s', r) transition.
            noise: Sampled Gaussian noise ε ~ N(0, I). Float32 tensor of shape
                (B, input_dim), same shape as x0. Must be on self.device.
                Sampled by ConditionalDiffusion.train_step() via torch.randn_like.
            t: Diffusion timestep indices. Long tensor of shape (B,) with values
                in [0, num_timesteps-1]. Sampled by sample_timesteps() in
                ConditionalDiffusion.train_step(). Must be on self.device.

        Returns:
            Float32 tensor of shape (B, input_dim) — the noisy transition x_t
            at the given timesteps. Passed to the denoising network as input,
            along with t and the condition c, to predict the noise ε.
        """
        # Gather per-sample schedule coefficients using t as indices.
        # self.sqrt_alphas_cumprod shape: (T,)
        # t shape: (B,) with values in [0, T-1]
        # Result after indexing: (B,)
        sqrt_ac: torch.Tensor = self.sqrt_alphas_cumprod[t]           # (B,)
        sqrt_oneminus_ac: torch.Tensor = self.sqrt_one_minus_alphas_cumprod[t]  # (B,)

        # Reshape to (B, 1) for broadcasting against (B, input_dim).
        # This allows element-wise multiplication with x0 and noise without
        # explicit expansion — PyTorch broadcasts (B, 1) × (B, D) → (B, D).
        sqrt_ac = sqrt_ac.view(-1, 1)           # (B, 1)
        sqrt_oneminus_ac = sqrt_oneminus_ac.view(-1, 1)  # (B, 1)

        # Apply the forward process reparameterization:
        # x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε
        x_t: torch.Tensor = sqrt_ac * x0 + sqrt_oneminus_ac * noise  # (B, input_dim)

        return x_t

    def step(
        self,
        x_t: torch.Tensor,
        predicted_noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Implements one reverse denoising step p(x_{t-1} | x_t).

        Computes the posterior mean μ_t from the predicted noise ε_θ and
        adds scaled Gaussian noise (zero at t=0 for a deterministic final step):

            μ_t = (1/sqrt(α_t)) * (x_t - β_t / sqrt(1 - ᾱ_t) * ε_θ)
            x_{t-1} = μ_t + mask(t > 0) * sqrt(σ_t²) * z,   z ~ N(0, I)

        Called inside ConditionalDiffusion._ddpm_reverse() in a loop from
        t = num_timesteps-1 down to t = 0. At each step, the model predicts
        the noise ε_θ via CFG sampling, and this method computes x_{t-1}.

        The t=0 mask ensures no noise is added at the final denoising step,
        producing a deterministic output. The clamp on posterior_variance
        before sqrt prevents NaN from sqrt(0) at t=0.

        Args:
            x_t: Current noisy sample at timestep t. Float32 tensor of shape
                (B, input_dim). Must be on self.device. Initialized as pure
                Gaussian noise x_T ~ N(0, I) at the start of generation, then
                iteratively denoised by this method.
            predicted_noise: Noise prediction ε_θ(x_t, t, c) from the
                denoising network after CFG combination. Float32 tensor of
                shape (B, input_dim), same shape as x_t. Must be on self.device.
                Produced by ConditionalDiffusion._cfg_sample().
            t: Diffusion timestep indices. Long tensor of shape (B,) with values
                in [0, num_timesteps-1]. In ConditionalDiffusion._ddpm_reverse(),
                this is a constant tensor (same timestep for all samples in the
                batch) created via t.expand(B) or torch.full((B,), t_val).
                Must be on self.device.

        Returns:
            Float32 tensor of shape (B, input_dim) — the denoised sample
            x_{t-1} at the previous timestep. After T reverse steps, this
            converges to a sample from the learned data distribution p_θ(x_0).
        """
        # ── Gather per-sample schedule values ────────────────────────────────
        # All schedule tensors have shape (T,); index with t to get (B,) values.
        betas_t: torch.Tensor = self.betas[t]                                    # (B,)
        sqrt_oneminus_ac_t: torch.Tensor = self.sqrt_one_minus_alphas_cumprod[t] # (B,)
        posterior_var_t: torch.Tensor = self.posterior_variance[t]               # (B,)

        # Compute sqrt(1/α_t) = 1 / sqrt(α_t) for the posterior mean formula.
        # self.alphas[t] shape: (B,)
        sqrt_recip_alphas_t: torch.Tensor = torch.rsqrt(self.alphas[t])          # (B,)

        # ── Reshape to (B, 1) for broadcasting against (B, input_dim) ────────
        betas_t = betas_t.view(-1, 1)                    # (B, 1)
        sqrt_oneminus_ac_t = sqrt_oneminus_ac_t.view(-1, 1)  # (B, 1)
        sqrt_recip_alphas_t = sqrt_recip_alphas_t.view(-1, 1)  # (B, 1)
        posterior_var_t = posterior_var_t.view(-1, 1)    # (B, 1)

        # ── Compute posterior mean μ_t ────────────────────────────────────────
        # μ_t = (1/sqrt(α_t)) * (x_t - β_t / sqrt(1 - ᾱ_t) * ε_θ)
        # This is the DDPM reverse step formula from Ho et al. (2020), Eq. 11.
        mean: torch.Tensor = sqrt_recip_alphas_t * (
            x_t - (betas_t / sqrt_oneminus_ac_t) * predicted_noise
        )  # (B, input_dim)

        # ── Add noise (zero at t=0) ───────────────────────────────────────────
        # Sample z ~ N(0, I) for the stochastic component.
        noise: torch.Tensor = torch.randn_like(x_t)  # (B, input_dim)

        # Create a binary mask: 1.0 where t > 0, 0.0 where t == 0.
        # This ensures no noise is added at the final denoising step (t=0),
        # producing a deterministic output x_0 from the learned distribution.
        # t shape: (B,) → mask shape: (B, 1) after view for broadcasting.
        mask: torch.Tensor = (t > 0).float().view(-1, 1)  # (B, 1)

        # Compute standard deviation: sqrt(σ_t²) = sqrt(posterior_variance_t)
        # Clamp to min=1e-20 before sqrt to prevent NaN at t=0 where
        # posterior_variance[0] = 0. The mask ensures the noise term is
        # multiplied by 0 at t=0 regardless, but the clamp provides an
        # additional numerical safeguard.
        std: torch.Tensor = torch.sqrt(
            posterior_var_t.clamp(min=1e-20)
        )  # (B, 1)

        # x_{t-1} = μ_t + mask(t > 0) * std * z
        x_prev: torch.Tensor = mean + mask * std * noise  # (B, input_dim)

        return x_prev

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """Samples random diffusion timesteps for a training batch.

        Implements uniform timestep sampling t ~ Unif(0, num_timesteps-1)
        for DDPM training. Called inside ConditionalDiffusion.train_step()
        to determine which noise level to apply to each sample in the batch.

        The paper states n ~ Unif(1, N) (1-indexed). We use 0-indexed
        [0, num_timesteps-1] for PyTorch compatibility — this is equivalent
        since the schedule tensors are indexed consistently.

        Args:
            batch_size: Number of timestep samples to draw. Corresponds to
                the training batch size (config.sampling.batch_size, default 256).
                Each sample in the batch receives an independently sampled
                timestep, allowing the model to learn denoising at all noise
                levels simultaneously.

        Returns:
            Long tensor of shape (batch_size,) with values uniformly sampled
            from [0, num_timesteps-1] (inclusive). On self.device. Dtype is
            torch.long for use as indices into the schedule tensors in
            add_noise() and step().
        """
        return torch.randint(
            low=0,
            high=self.num_timesteps,
            size=(batch_size,),
            device=self.device,
            dtype=torch.long,
        )

    def __repr__(self) -> str:
        """Returns a concise string representation of the DDPM scheduler."""
        beta_start: float = float(self.betas[0].item())
        beta_end: float = float(self.betas[-1].item())
        return (
            f"DDPMScheduler("
            f"num_timesteps={self.num_timesteps}, "
            f"beta_start={beta_start:.1e}, "
            f"beta_end={beta_end:.2f}, "
            f"device='{self.device}')"
        )

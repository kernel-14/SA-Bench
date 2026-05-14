## inference/sampler.py
"""Euler ODE sampler for Flow Marching Transformer inference.

Implements the Euler discretization of the learned velocity field from FMT,
transporting a latent state from t=0 to t=1 to produce the predicted next
latent frame.

From the paper (Section 3.4):
    "We use the Euler ODE sampler (discretization on t) to propagate an
    intermediate state x_t^k to x_1. The discretization is taken to be
    N=100 throughout the evaluation phase, with dt=0.01."

The Euler update rule:
    x_{t+dt} = x_t + dt * g_θ(x_t, t, h)

where g_θ is the FMT velocity field and h is the GRU hidden state encoding
PDE history from previous frames.

The sampler operates entirely in latent space (c16p16, shape (B, 16, 16, 16))
and is agnostic to:
  - The temporal pyramid (handled inside FMT)
  - GRU updates between autoregressive steps (handled in rollout.py)
  - Pixel-space decoding (handled in rollout.py via p2vae.decode)
  - The k initialization mode (deterministic k=1 vs generative k<1)

Configuration alignment (config.yaml):
    fmt.inference.euler_steps: 100   → num_steps default
    fmt.inference.dt: 0.01           → derived as 1/num_steps
"""

import logging
from typing import List, Optional, Tuple

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class EulerSampler:
    """Euler ODE sampler for integrating the FMT velocity field.

    Integrates the learned velocity field g_θ(x_t, t, h) from t=0 to t=1
    using the explicit Euler method with fixed step size dt=1/num_steps.

    This is a pure numerical integrator with no learnable parameters.
    It wraps calls to fmt.predict_velocity and accumulates the Euler steps.

    The GRU hidden state h is held fixed throughout the 100 Euler steps —
    it encodes the PDE condition from history frames and is updated between
    autoregressive steps by rollout.py, not inside the ODE integration.

    Two inference modes (both handled by how x_init is constructed before
    calling sample, not inside the sampler):

    Deterministic prediction (k=1, Table 3 evaluation):
        x_init = x_0  (clean previous latent, no noise)

    Generative sampling (k_3 < 1, Figure 3 ensemble):
        z ~ N(0, I)
        x_init = k_3 * x_3 + (1 - k_3) * z
        k_3 values: {0, 0.3, 0.6, 0.9} from config.yaml: ensemble.k3_values

    Attributes:
        num_steps: Number of Euler integration steps. From config.yaml:
            fmt.inference.euler_steps = 100.
        dt: Step size for each Euler step. Computed as 1.0 / num_steps.
            From config.yaml: fmt.inference.dt = 0.01 (= 1/100).
    """

    def __init__(self, num_steps: int = 100) -> None:
        """Initialize the Euler sampler.

        Args:
            num_steps: Number of Euler integration steps from t=0 to t=1.
                From config.yaml: fmt.inference.euler_steps = 100.
                The step size dt = 1.0 / num_steps is derived automatically.
                With num_steps=100: dt=0.01, matching the paper (Section 3.4).

        Raises:
            ValueError: If num_steps is not a positive integer.
        """
        if num_steps <= 0:
            raise ValueError(
                f"num_steps must be a positive integer, got {num_steps}."
            )

        self.num_steps: int = num_steps
        # dt = 1/N = 0.01 for N=100 (paper Section 3.4: "dt=0.01")
        self.dt: float = 1.0 / float(num_steps)

        logger.info(
            "EulerSampler initialized: num_steps=%d, dt=%.4f",
            self.num_steps,
            self.dt,
        )

    def sample_step(
        self,
        x_t: Tensor,
        t: float,
        velocity: Tensor,
    ) -> Tensor:
        """Execute a single Euler integration step.

        Implements the explicit Euler update:
            x_{t+dt} = x_t + dt * velocity

        This is a stateless operation — no side effects, no gradient
        computation. The velocity was already computed by FMT at time t.

        Args:
            x_t: Current latent state at time t, shape (B, 16, 16, 16).
                dtype should be float32 for numerical stability.
            t: Current time scalar (Python float) in [0, 1). Accepted for
                API completeness and logging; the velocity was already
                evaluated at this t by the caller.
            velocity: Velocity prediction g_θ(x_t, t, h) from FMT,
                shape (B, 16, 16, 16). Same dtype and device as x_t.

        Returns:
            Updated latent state x_{t+dt} of shape (B, 16, 16, 16).
            Same dtype and device as x_t.

        Note on numerical behavior near t→1:
            At the last step (t≈0.99), the velocity magnitude grows as
            g_θ ≈ (x_1 - x_t)/(1-t). The Euler step dt * velocity has
            magnitude dt/(1-t) = 0.01/0.01 = 1.0, which is the correct
            final correction. This is well-behaved for N=100 steps.
        """
        return x_t + self.dt * velocity

    def sample(
        self,
        fmt: "FMT",  # type: ignore[name-defined]  # forward reference
        x_init: Tensor,
        h: Tensor,
        condition_latents: Optional[List[Tensor]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Integrate the FMT velocity field from t=0 to t=1 via Euler steps.

        Runs num_steps=100 Euler integration steps, calling
        fmt.predict_velocity at each step to obtain the velocity, then
        advancing the latent state by dt=0.01.

        The GRU hidden state h is passed to every predict_velocity call
        but is NOT updated inside this method. The h returned is identical
        to the h passed in. GRU updates between autoregressive steps are
        the responsibility of rollout.py, keeping this sampler stateless
        and reusable across different rollout strategies.

        The time grid is constructed as t_step = step * dt to avoid
        floating-point accumulation errors from repeated addition.

        Args:
            fmt: FMT model instance. Must implement predict_velocity(
                x_tk: Tensor, t: Tensor, h: Tensor) -> Tensor.
                The model should already be in eval mode and on the
                correct device before calling this method.
            x_init: Initial latent state at t=0, shape (B, 16, 16, 16).
                For deterministic prediction (k=1): x_init = x_0 (clean).
                For generative sampling (k_3<1): x_init = k_3*x_3 + (1-k_3)*z.
                dtype: float32. Device: same as fmt parameters.
            h: GRU hidden state encoding PDE history from previous frames,
                shape (B, embed_dim) where embed_dim is 256/512/768 depending
                on the FMT variant (FMT-S/B/L). Initialized to zeros at the
                start of each trajectory by rollout.py.
                dtype: float32. Device: same as x_init.
            condition_latents: Optional list of history latent frames for
                additional context. Currently unused in the base implementation
                (h already encodes history via the GRU). Accepted for API
                extensibility (e.g., classifier-free guidance). Default: None.

        Returns:
            Tuple of:
                x_1_latent: Predicted latent at t=1, shape (B, 16, 16, 16).
                    This is the output of the ODE integration — the predicted
                    next frame in latent space. Decode via p2vae.decode() in
                    rollout.py to obtain the pixel-space prediction.
                h_unchanged: The GRU hidden state h, returned unchanged.
                    Shape (B, embed_dim). The caller (rollout.py) is
                    responsible for updating h via gru_forcing.update()
                    after decoding x_1_latent.

        Note:
            This method runs under torch.inference_mode() for efficiency.
            If called during training (e.g., for validation), the caller
            should ensure no gradient computation is needed.
        """
        b: int = x_init.shape[0]
        device: torch.device = x_init.device

        # Validate input shapes.
        if x_init.ndim != 4:
            raise ValueError(
                f"x_init must have shape (B, C, H, W), got {x_init.shape}."
            )
        if h.ndim != 2:
            raise ValueError(
                f"h must have shape (B, embed_dim), got {h.shape}."
            )
        if h.shape[0] != b:
            raise ValueError(
                f"Batch size mismatch: x_init has B={b} but h has B={h.shape[0]}."
            )

        # Run integration under inference mode for efficiency.
        # torch.inference_mode() is stronger than no_grad: disables version
        # tracking and is safe for pure inference (no in-place ops on inputs).
        with torch.inference_mode():
            # Initialize the running latent state.
            # Clone to avoid modifying the caller's x_init tensor.
            x_t: Tensor = x_init.clone()

            # Euler integration loop: num_steps=100 steps from t=0 to t≈1.
            # Time grid: t_step = step * dt avoids floating-point accumulation
            # from repeated addition (e.g., 0.01 + 0.01 + ... ≠ exactly 1.0).
            for step in range(self.num_steps):
                # Compute current time: t_step ∈ {0.00, 0.01, ..., 0.99}
                # Using multiplication instead of accumulation for precision.
                t_current: float = step * self.dt

                # Construct batch timestep tensor: shape (B,), dtype float32.
                # FMT's TimestepEmbedder expects a 1D tensor of shape (B,).
                t_tensor: Tensor = torch.full(
                    (b,),
                    fill_value=t_current,
                    dtype=torch.float32,
                    device=device,
                )

                # Query FMT for the velocity at current (x_t, t, h).
                # predict_velocity returns g_θ(x_t, t, h) of shape (B, 16, 16, 16).
                # h is held fixed throughout the 100 steps — it encodes the
                # PDE condition from history frames (frames 0, 1, 2) and is
                # updated between autoregressive steps by rollout.py.
                velocity: Tensor = fmt.predict_velocity(
                    x_tk=x_t,
                    t=t_tensor,
                    h=h,
                )

                # Euler step: x_{t+dt} = x_t + dt * velocity
                x_t = self.sample_step(x_t=x_t, t=t_current, velocity=velocity)

            # x_t is now the predicted latent at t=1 (approximately).
            # The final time after num_steps steps is:
            #   t_final = num_steps * dt = 100 * 0.01 = 1.0
            x_1_latent: Tensor = x_t

        # Return the predicted latent and the unchanged GRU hidden state.
        # The caller (rollout.py) updates h via gru_forcing.update() after
        # decoding x_1_latent to pixel space.
        return x_1_latent, h

    def __repr__(self) -> str:
        """Return a string representation of the sampler configuration."""
        return (
            f"EulerSampler(num_steps={self.num_steps}, dt={self.dt:.4f})"
        )

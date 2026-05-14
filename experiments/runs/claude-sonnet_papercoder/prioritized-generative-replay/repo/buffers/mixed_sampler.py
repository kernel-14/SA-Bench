## buffers/mixed_sampler.py
"""Mixed sampler for Prioritized Generative Replay (PGR).

Combines real transitions from D_real and synthetic transitions from D_syn
at a configurable ratio r, implementing the core data mixing mechanism
described in Section 4.3 of the paper:

    "we randomly sample synthetic and real data mixed according to some
     ratio r to train our policy π"

Consumed exclusively by PGRTrainer._update_policy(), which calls sample()
utd_ratio (20) times per environment step. The ratio r maps directly to
config.sampling.synthetic_ratio = 0.5 in the baseline configuration,
yielding 128 real + 128 synthetic transitions per batch of 256.

The sampler handles the early-training regime (D_syn empty before the first
inner loop at step 10000) by falling back to sampling entirely from D_real,
ensuring the policy receives utd_ratio gradient updates from the very first
environment step.
"""

from typing import Dict

import torch

from buffers.replay_buffer import ReplayBuffer


class MixedSampler:
    """Samples mixed batches from D_real and D_syn at a configurable ratio.

    Implements the data mixing strategy from PGR Section 4.3. The synthetic
    ratio r determines what fraction of each batch is drawn from D_syn:

        n_syn  = int(batch_size * synthetic_ratio)   # e.g. 128 for r=0.5, B=256
        n_real = batch_size - n_syn                  # e.g. 128 for r=0.5, B=256

    Falls back to sampling entirely from D_real when D_syn is empty (before
    the first inner loop) or when D_syn has fewer transitions than n_syn.

    The ratio is mutable via set_ratio() to support the scaling experiments
    in Section 5.3 (r=0.75 with batch=512, r=0.875 with batch=1024).

    Attributes:
        real_buffer: Reference to D_real (live environment transitions).
        syn_buffer: Reference to D_syn (conditionally generated transitions).
        synthetic_ratio: Fraction of each batch drawn from D_syn. Mutable
            via set_ratio(). Corresponds to config.sampling.synthetic_ratio.
        device: PyTorch device string for output tensors. Both buffers must
            use the same device. Corresponds to config.hardware.device.
    """

    def __init__(
        self,
        real_buffer: ReplayBuffer,
        syn_buffer: ReplayBuffer,
        synthetic_ratio: float = 0.5,
        device: str = "cuda",
    ) -> None:
        """Initialises the mixed sampler with references to both replay buffers.

        No data is copied at construction time — the sampler always reads
        live from the buffers at sample() call time.

        Args:
            real_buffer: The real experience replay buffer D_real. Must be
                the same ReplayBuffer instance used by PGRTrainer for
                environment transition storage. Always populated from the
                first environment step.
            syn_buffer: The synthetic replay buffer D_syn. Empty until the
                first inner loop fires at step inner_loop_freq=10000. The
                sampler handles this empty state gracefully via fallback.
            synthetic_ratio: Fraction of each batch to draw from D_syn.
                Must satisfy 0.0 <= synthetic_ratio < 1.0. Corresponds to
                config.sampling.synthetic_ratio (default 0.5). Updated via
                set_ratio() for scaling experiments:
                    - 0.5  (baseline, batch=256): 128 real + 128 synthetic
                    - 0.75 (scaling, batch=512):  128 real + 384 synthetic
                    - 0.875 (catastrophic case):  128 real + 896 synthetic
            device: PyTorch device string. Stored for consistency checks.
                Output tensors are already on this device because
                ReplayBuffer.sample() handles device placement internally.
                Corresponds to config.hardware.device (default "cuda").
        """
        if not (0.0 <= synthetic_ratio < 1.0):
            raise ValueError(
                f"synthetic_ratio must satisfy 0.0 <= ratio < 1.0, "
                f"got {synthetic_ratio}. A ratio of 1.0 would yield zero "
                f"real transitions per batch, which is degenerate."
            )

        self.real_buffer: ReplayBuffer = real_buffer
        self.syn_buffer: ReplayBuffer = syn_buffer
        self.synthetic_ratio: float = float(synthetic_ratio)
        self.device: str = device

    # ── Public API ────────────────────────────────────────────────────────────

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Samples a mixed batch from D_real and D_syn at the configured ratio.

        Implements two regimes:

        **Regime 1 — Early training fallback (D_syn empty or too small):**
        Before the first inner loop call (step < inner_loop_freq=10000),
        D_syn contains zero transitions. The sampler falls back to drawing
        the full batch from D_real. This ensures the policy receives
        utd_ratio=20 gradient updates per step from the very first
        environment step, without any crashes or special-casing in the
        caller.

        The same fallback applies if D_syn has been partially filled but
        contains fewer transitions than n_syn — this can occur if generation
        was interrupted or if D_syn was just cleared at the start of a new
        inner loop before regeneration completes.

        **Regime 2 — Normal operation (D_syn sufficiently populated):**
        Draws n_real transitions from D_real and n_syn transitions from
        D_syn independently, then concatenates along the batch dimension.
        The concatenation order (real first, then synthetic) is arbitrary
        and has no effect on learning since REDQ samples i.i.d.

        The returned dict follows the universal transition format shared
        across all PGR modules (from the design's "Shared Knowledge"):

            {
                'observations':      Tensor (batch_size, obs_dim),
                'actions':           Tensor (batch_size, action_dim),
                'next_observations': Tensor (batch_size, obs_dim),
                'rewards':           Tensor (batch_size, 1),
                'dones':             Tensor (batch_size, 1),
            }

        All tensors are float32 on self.device (device placement is handled
        by ReplayBuffer.sample() internally; torch.cat preserves device).

        Args:
            batch_size: Total number of transitions to return. Corresponds
                to config.sampling.batch_size (default 256). The caller
                (PGRTrainer._update_policy) is responsible for passing the
                correct batch size — this method does not read from config.

        Returns:
            Transition dict with float32 tensors of shape (batch_size, dim)
            on self.device. Keys: 'observations', 'actions',
            'next_observations', 'rewards', 'dones'.

        Raises:
            RuntimeError: If D_real is also empty (should never happen in
                normal training since D_real is populated before any policy
                updates begin).
        """
        # Compute the desired split sizes.
        n_syn: int = int(batch_size * self.synthetic_ratio)
        n_real: int = batch_size - n_syn

        # ── Regime 1: Fallback when D_syn is empty or too small ───────────────
        # D_syn is empty before the first inner loop (step < 10000) and
        # immediately after clear() is called at the start of each inner loop
        # before regeneration completes. Fall back to all-real sampling.
        syn_available: int = len(self.syn_buffer)
        if syn_available == 0 or syn_available < n_syn:
            # Sample the full batch from D_real.
            return self.real_buffer.sample(batch_size)

        # ── Regime 2: Normal mixed sampling ──────────────────────────────────
        # Sample independently from each buffer.
        real_batch: Dict[str, torch.Tensor] = self.real_buffer.sample(n_real)
        syn_batch: Dict[str, torch.Tensor] = self.syn_buffer.sample(n_syn)

        # Concatenate along the batch dimension (dim=0) for each key.
        # Both dicts have identical key sets since both come from
        # ReplayBuffer.sample(), which always returns the same five keys.
        # torch.cat preserves device and dtype — no explicit .to() needed.
        mixed_batch: Dict[str, torch.Tensor] = {
            key: torch.cat([real_batch[key], syn_batch[key]], dim=0)
            for key in real_batch
        }

        return mixed_batch

    def set_ratio(self, ratio: float) -> None:
        """Updates the synthetic data ratio for scaling experiments.

        Called by PGRTrainer during the scaling experiments described in
        Section 5.3 of the paper to switch between:
            - r=0.5  (baseline, batch=256):  128 real + 128 synthetic
            - r=0.75 (scaling, batch=512):   128 real + 384 synthetic
            - r=0.875 (catastrophic case):   128 real + 896 synthetic

        The batch_size change is handled at the PGRTrainer level — this
        method only updates the fraction. The new ratio takes effect on
        the next call to sample().

        Args:
            ratio: New synthetic data fraction. Must satisfy
                0.0 <= ratio < 1.0. A ratio of 1.0 is rejected because it
                would yield zero real transitions per batch, which is
                degenerate and not supported by the paper's methodology.

        Raises:
            ValueError: If ratio is outside [0.0, 1.0).
        """
        if not (0.0 <= ratio < 1.0):
            raise ValueError(
                f"synthetic_ratio must satisfy 0.0 <= ratio < 1.0, "
                f"got {ratio}. A ratio of 1.0 would yield zero real "
                f"transitions per batch, which is degenerate."
            )
        self.synthetic_ratio = float(ratio)

    def __repr__(self) -> str:
        """Returns a concise string representation of the sampler state."""
        return (
            f"MixedSampler("
            f"synthetic_ratio={self.synthetic_ratio:.3f}, "
            f"real_buffer_size={len(self.real_buffer)}, "
            f"syn_buffer_size={len(self.syn_buffer)}, "
            f"device='{self.device}')"
        )

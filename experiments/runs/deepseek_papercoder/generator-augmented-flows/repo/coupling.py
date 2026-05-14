"""
coupling.py – Trajectory construction for independent (IC), optimal transport (OT),
and generator‑augmented (GC) couplings.

This module implements the Coupling class described in the design. At each
training step it turns a batch of real images and random noise into pairs of
perturbed points (x_ti, x_tip1) according to the chosen coupling strategy.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor

from model import ConsistencyModel  # from our model.py


class Coupling:
    """
    Constructs intermediate points for the consistency loss using one of three
    coupling schemes.

    Args:
        type:   coupling type, one of ``"ic"``, ``"ot"``, ``"gc"``.
        model:  the (EMA) consistency model; required for ``"gc"``.
        mu:     probability (0 <= mu <= 1) of using the GC‑predicted endpoint
                instead of the original data point. Only relevant for ``"gc"``.
    """

    def __init__(
        self,
        type: str,
        model: Optional[ConsistencyModel] = None,
        mu: float = 0.0,
    ) -> None:
        if type not in {"ic", "ot", "gc"}:
            raise ValueError(f"Unknown coupling type: '{type}'. Use 'ic', 'ot', or 'gc'.")
        if type == "gc" and model is None:
            raise ValueError("GC coupling requires a ConsistencyModel instance.")
        self.type = type
        self.model = model
        self.mu = mu

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def construct_pair(
        self,
        x_star: Tensor,
        noise: Tensor,
        sigma_i: Tensor,
        sigma_ip1: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Return the pair of perturbed points (x_ti, x_tip1) that will be used
        in the consistency loss.

        Args:
            x_star:    clean images, shape (B, C, H, W).
            noise:     Gaussian noise of same shape as x_star.
            sigma_i:   noise level for the earlier timestep (0‑dim tensor or (B,)).
            sigma_ip1: noise level for the later timestep (0‑dim tensor or (B,)).

        Returns:
            x_ti:   point at time t_i, shape (B, C, H, W).
            x_tip1: point at time t_{i+1}, shape (B, C, H, W).
        """
        B = x_star.shape[0]
        # Broadcast noise levels to (B,1,1,1) for easy multiplication.
        sigma_i_bc = self._broadcast_sigma(sigma_i, B)
        sigma_ip1_bc = self._broadcast_sigma(sigma_ip1, B)

        # ---------- Independent coupling ----------
        if self.type == "ic":
            x_ti = x_star + sigma_i_bc * noise
            x_tip1 = x_star + sigma_ip1_bc * noise
            return x_ti, x_tip1

        # ---------- Minibatch optimal transport coupling ----------
        elif self.type == "ot":
            # 1. Flatten images and compute pairwise squared Euclidean distances.
            x_flat = x_star.reshape(B, -1)        # (B, C*H*W)
            n_flat = noise.reshape(B, -1)
            with torch.no_grad():
                # squared L2 distance: ||x - n||^2 = sum( (x-n)^2 )
                # Equivalent to torch.cdist(p=2).square()
                cost = torch.cdist(x_flat, n_flat, p=2).square()  # (B, B)

            # 2. Solve the linear assignment problem (Hungarian algorithm).
            cost_np = cost.cpu().numpy()
            try:
                from scipy.optimize import linear_sum_assignment
            except ImportError:
                raise ImportError(
                    "scipy is required for OT coupling. "
                    "Install with `pip install scipy`."
                )
            row_ind, col_ind = linear_sum_assignment(cost_np)
            # col_ind gives the permutation: data i is matched to noise col_ind[i].
            perm = torch.tensor(col_ind, device=x_star.device, dtype=torch.long)

            # 3. Reorder noise according to the optimal assignment.
            noise_ot = noise[perm]

            # 4. Construct the pair using the matched noise.
            x_ti = x_star + sigma_i_bc * noise_ot
            x_tip1 = x_star + sigma_ip1_bc * noise_ot
            return x_ti, x_tip1

        # ---------- Generator-augmented coupling ----------
        elif self.type == "gc":
            # Shortcut: if mu == 0, fall back to IC (no model call needed).
            if self.mu == 0.0:
                return self.construct_pair(x_star, noise, sigma_i, sigma_ip1)

            # Step 1: IC intermediate point (used to predict the clean endpoint).
            x_ti_ic = x_star + sigma_i_bc * noise

            # Prepare sigma_i as (B,) for the model (the model expects 1D input).
            sigma_model = self._to_1d(sigma_i, B)

            # Step 2: Predict clean data with the (EMA) model – no gradient.
            with torch.no_grad():
                x_hat = self.model(x_ti_ic, sigma_model)

            # Step 3: Stochastic mixing mask: each sample independently uses the
            # predicted endpoint (x_hat) with probability mu, else the original
            # clean data point (x_star).
            # Shape (B, 1, 1, 1) allows broadcasting over spatial dimensions.
            mask = torch.bernoulli(
                torch.full((B, 1, 1, 1), self.mu, device=x_star.device)
            )

            # Step 4: Blend endpoints.
            x_choice = mask * x_hat + (1.0 - mask) * x_star

            # Step 5: Construct the GC intermediate points.
            x_ti = x_choice + sigma_i_bc * noise
            x_tip1 = x_choice + sigma_ip1_bc * noise
            return x_ti, x_tip1

        # ------------------------------------------------------------------
        else:
            raise NotImplementedError(
                f"Coupling type '{self.type}' is not implemented."
            )

    def gc_predict(self, x_t: Tensor, sigma_t: Tensor) -> Tensor:
        """
        Convenience wrapper: predict the clean endpoint from a noisy point.

        This is used for auxiliary tasks such as computing the proxy term or
        transport cost. No gradient flows through the model.

        Args:
            x_t:     noisy images, shape (B, C, H, W).
            sigma_t: noise level(s). Can be a scalar (0‑dim) or (B,).

        Returns:
            pred: predicted clean images, shape (B, C, H, W).
        """
        B = x_t.shape[0]
        sigma_1d = self._to_1d(sigma_t, B)
        with torch.no_grad():
            return self.model(x_t, sigma_1d)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    @staticmethod
    def _broadcast_sigma(sigma: Tensor, batch_size: int) -> Tensor:
        """
        Convert a noise level tensor to shape (batch_size, 1, 1, 1) so that
        it can broadcast with image tensors.

        sigma may be a 0‑dim scalar or a 1D tensor.
        """
        if sigma.dim() == 0:                # scalar
            return sigma.view(1, 1, 1, 1).expand(batch_size, -1, -1, -1)
        # 1D tensor
        return sigma.reshape(-1, 1, 1, 1)

    @staticmethod
    def _to_1d(sigma: Tensor, batch_size: int) -> Tensor:
        """
        Convert a noise level tensor to shape (batch_size,) for model input.

        sigma may be a 0‑dim scalar or a 1D tensor (possibly of length 1).
        """
        if sigma.dim() == 0:
            return sigma.unsqueeze(0).expand(batch_size)
        if sigma.numel() == 1:
            return sigma.expand(batch_size)
        return sigma   # already (B,)

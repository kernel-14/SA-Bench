## models/wdno.py
"""
Wavelet Diffusion Neural Operator (WDNO) orchestrator.

Coordinates the Base‑Resolution Model (BRM), Super‑Resolution Model (SRM),
wavelet transforms, and an optional control surrogate to perform simulation,
control, and zero‑shot super‑resolution of PDE systems.

All generation is performed in the wavelet domain; the outputs are transformed
back to physical space via the inverse wavelet transform.

The class relies on:
  - `diffusion.DDPM`          for BRM / SRM DDPM instances
  - `wavelet_utils.WaveletTransform` for forward/inverse transforms
  - an optional control surrogate (not used directly, but passed to the
    objective function during control).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from diffusion import DDPM
from wavelet_utils import WaveletTransform


# ------------------------------------------------------------------------------
# Helper used internally by DDIM steps
# ------------------------------------------------------------------------------

def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """Extract appropriate values from a 1‑D schedule tensor and reshape.

    Args:
        a: 1‑D tensor of pre‑computed values (length T).
        t: integer indices, shape (B,).
        x_shape: shape of the denoising target, e.g., (B, C, H, W).

    Returns:
        Tensor of shape (B, 1, 1, ...) ready for broadcasting.
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


# ------------------------------------------------------------------------------
# Helper for 1D wavelet repetition to target spatial shape
# ------------------------------------------------------------------------------

def _repeat_1d_coeffs_to_target(
    coeffs_1d: List[torch.Tensor],
    target_shape: Tuple[int, ...],
) -> torch.Tensor:
    """Repeat 1D wavelet coefficients to fill a higher‑dimensional grid.

    Args:
        coeffs_1d: list of two tensors [cA, cD], each of shape (L,).
        target_shape: spatial shape of the target wavelet domain,
            e.g., (H', W') for 2D wavelet or (T', H', W') for 3D.

    Returns:
        Tensor of shape (2, *target_shape) obtained by repeating each
        coefficient along the missing spatial dimensions.
    """
    if len(coeffs_1d) != 2:
        raise ValueError("1D wavelet must provide exactly 2 coefficients (cA, cD).")

    L = coeffs_1d[0].shape[0]
    if L != target_shape[-1]:
        raise ValueError(
            f"1D coefficient length {L} does not match last dimension "
            f"of target shape {target_shape[-1]}."
        )

    repeated = []
    for coeff in coeffs_1d:
        # coeff shape (L,)
        # Build view: (1, 1, ..., 1, L) with as many leading 1's as needed
        view_shape = (1,) + (1,) * (len(target_shape) - 1) + (L,)
        t = coeff.view(*view_shape)
        # Expand to target_shape and add a leading batch dim of 1, then squeeze batch
        t = t.expand(1, *target_shape)  # shape (1, *target_shape)
        repeated.append(t.squeeze(0))   # shape (*target_shape)

    # Stack along a new leading dimension to get (2, *target_shape)
    return torch.stack(repeated, dim=0)


# ------------------------------------------------------------------------------
# WDNO class
# ------------------------------------------------------------------------------

class WDNO:
    """Orchestrates wavelet‑domain diffusion for simulation, control and super‑resolution.

    Args:
        brm: Base‑Resolution Diffusion Model (DDPM instance).
        srm: Super‑Resolution Diffusion Model (DDPM instance), may be ``None``
            if multi‑resolution training is not used.
        wavelet_transform: Configured wavelet transformation object
            (must be a ``WaveletTransform``).
        control_surrogate: Optional surrogate model for control objectives.
            Not used directly by WDNO; the caller's ``objective_fn`` may rely on it.
        ddim_steps: Number of DDIM sampling steps (default 50).
        ddim_eta: Stochasticity parameter for DDIM (0.0 = deterministic).
        device: PyTorch device (default ``'cuda'`` or ``'cpu'``).
    """

    def __init__(
        self,
        brm: DDPM,
        srm: Optional[DDPM],
        wavelet_transform: WaveletTransform,
        control_surrogate: Optional[Any] = None,
        ddim_steps: int = 50,
        ddim_eta: float = 0.0,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.brm = brm
        self.srm = srm
        self.wavelet_transform = wavelet_transform
        self.control_surrogate = control_surrogate

        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Compute number of wavelet subbands on‑the‑fly
        if wavelet_transform.ndim == 2:
            self.num_subbands = 4
        elif wavelet_transform.ndim == 3:
            self.num_subbands = 8
        else:
            raise ValueError(f"Unsupported wavelet ndim: {wavelet_transform.ndim}")

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------

    @torch.no_grad()
    def simulate(
        self,
        condition: Dict[str, torch.Tensor],
        guidance_w: float = 0.0,
    ) -> torch.Tensor:
        """Generate a full PDE trajectory given equation parameters.

        Args:
            condition: dictionary mapping parameter names (e.g., ``"u0"``, ``"f"``)
                to physical tensors.  1D signals are treated specially; 2D/3D fields
                are transformed with the same wavelet as the state.
            guidance_w: classifier‑free guidance weight (default 0 = unconditional).

        Returns:
            Predicted physical trajectory tensor.  Shape depends on the experiment
            (e.g., ``(B, T, X)`` for 1D, ``(B, C, T, X, Y)`` for 2D).
        """
        # Prepare wavelet‑domain conditioning
        cond_wavelet = self._prepare_condition(condition)
        batch_size = cond_wavelet.shape[0]
        spatial_w = cond_wavelet.shape[2:]
        out_channels = self.brm.out_channels

        # DDIM sampling with optional classifier‑free guidance
        W_state = self.brm.sample_ddim(
            cond=cond_wavelet,
            guidance_w=guidance_w,
            ddim_steps=self.ddim_steps,
            ddim_eta=self.ddim_eta,
        )

        # Convert generated wavelet to physical trajectory
        trajectory = self._wavelet_to_physical(W_state)
        return trajectory

    def control(
        self,
        condition: Dict[str, torch.Tensor],
        objective_fn: Callable[..., torch.Tensor],
        lambda_val: float,
        guidance_w: float = 1.0,
    ) -> torch.Tensor:
        """Generate an optimal control force sequence via guided diffusion.

        Args:
            condition: dictionary containing PDE parameters (initial condition,
                target state, etc.).  The force field must **not** be present
                because it is generated.
            objective_fn: differentiable callable ``J = objective_fn(f, **condition)``
                where ``f`` is the physical force trajectory (with batch dim) and
                ``condition`` is the same dictionary (extended with ``'f'`` if needed).
                The callable must return a scalar objective to minimise.
            lambda_val: weight of the energy‑based guidance term.
            guidance_w: classifier‑free guidance weight.

        Returns:
            Optimal control force tensor in physical space.
        """
        # ---- 1. Prepare condition wavelet --------------------------------------
        cond_wavelet = self._prepare_condition(condition)   # (B, C_cond, *spatial_w)
        batch_size = cond_wavelet.shape[0]
        spatial_w = cond_wavelet.shape[2:]
        out_channels = self.brm.out_channels

        # ---- 2. Initialise noisy force wavelet ---------------------------------
        device = cond_wavelet.device
        W_k = torch.randn(batch_size, out_channels, *spatial_w, device=device)
        W_k.requires_grad_(True)

        # ---- 3. DDIM schedule setup -------------------------------------------
        ddim_steps = self.ddim_steps
        times = torch.linspace(
            self.brm.n_timesteps - 1, 0, ddim_steps, device=device
        ).round().long()

        # Unconditional conditioning (all zeros)
        zero_cond = torch.zeros_like(cond_wavelet)

        # ---- 4. Iterative denoising with guidance -----------------------------
        for i in range(len(times) - 1):
            t_cur = times[i]
            t_next = times[i + 1]

            # Obtain classifier‑free guided noise (no grad)
            with torch.no_grad():
                eps_cond = self.brm.denoiser(
                    W_k, t_cur.expand(batch_size), cond_wavelet
                )
                eps_uncond = self.brm.denoiser(
                    W_k, t_cur.expand(batch_size), zero_cond
                )
                eps_guided = eps_uncond + guidance_w * (eps_cond - eps_uncond)

            # Clean estimate (Eq. 5 in paper)
            alpha_bar_t = _extract(self.brm.alphas_cumprod, t_cur, W_k.shape)
            W0_hat = (W_k - (1 - alpha_bar_t).sqrt() * eps_guided) / alpha_bar_t.sqrt()

            # Physical force from clean estimate
            f_hat = self._wavelet_to_physical(W0_hat)

            # Objective and gradient
            # We pass the condition dict as keyword arguments; the objective_fn
            # may use the keys it needs (e.g., "initial", "target").
            J = objective_fn(f_hat, **condition)
            grad_J = torch.autograd.grad(J, W_k, retain_graph=False, create_graph=False)[0]
            # In case grad_J is None (e.g., J constant), treat as zero
            if grad_J is None:
                grad_J = torch.zeros_like(W_k)

            # Final noise prediction used for the DDIM step
            eps_final = eps_guided + lambda_val * grad_J

            # Perform DDIM step
            W_k = self._ddim_step(
                W_k, eps_final, t_cur, t_next, alpha_bar_t, self.ddim_eta
            )

            # Detach and prepare for next iteration
            W_k = W_k.detach().requires_grad_(True)

        # ---- 5. Final physical control force ----------------------------------
        final_force = self._wavelet_to_physical(W_k)
        return final_force

    @torch.no_grad()
    def super_resolve(
        self,
        low_res_wavelet: torch.Tensor,
        high_res_cond: torch.Tensor,
        levels: int = 1,
    ) -> torch.Tensor:
        """Iteratively upscale wavelet coefficients using the Super‑Resolution Model.

        Args:
            low_res_wavelet: flat wavelet coefficients at the current low resolution,
                shape ``(B, C, H_l, W_l)`` (or 3D equivalent).
            high_res_cond: wavelet coefficients of the high‑resolution equation
                parameters, shape ``(B, C_cond, H_h, W_h)``.
            levels: number of successive super‑resolution steps.

        Returns:
            High‑resolution wavelet coefficients of the generated trajectory,
            shape ``(B, C, H_h, W_h)``.

        Raises:
            RuntimeError: if the SRM is ``None``.
        """
        if self.srm is None:
            raise RuntimeError("Super‑Resolution Model (SRM) is not available.")

        current_low = low_res_wavelet
        # For simplicity, we assume the high_res_cond is already at the finest
        # resolution and the SRM is conditioned on it at every level.
        for _ in range(levels):
            # Upsample low‑res to match high‑res spatial shape
            upsampled_low = F.interpolate(
                current_low,
                size=high_res_cond.shape[2:],
                mode="nearest",
            )
            # Concatenate upsampled low‑res and high‑res condition
            srm_cond = torch.cat([upsampled_low, high_res_cond], dim=1)

            # Generate high‑res wavelet with SRM
            current_low = self.srm.sample_ddim(
                cond=srm_cond,
                guidance_w=0.0,
                ddim_steps=self.ddim_steps,
                ddim_eta=self.ddim_eta,
            )
        return current_low

    # --------------------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------------------

    def _prepare_condition(
        self,
        condition: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Transform all physical condition tensors to wavelet coefficients and
        concatenate them into a single conditioning tensor of shape
        ``(B, C_total, *spatial_w)``.

        The spatial shape ``spatial_w`` is inferred from the first 2D/3D tensor
        in the dictionary.
        """
        if not condition:
            raise ValueError("The condition dictionary must not be empty.")

        # ---- Infer reference wavelet spatial shape ---------------------------
        ref_tensor = None
        for val in condition.values():
            if val.dim() >= 2:  # at least 2 spatial dims
                ref_tensor = val
                break
        if ref_tensor is None:
            raise ValueError(
                "No condition tensor with spatial dimensions ≥ 2 found; "
                "cannot determine wavelet spatial shape."
            )

        # Compute the wavelet shape by transforming a single‑channel dummy
        # of the same spatial extent as the reference tensor.
        with torch.no_grad():
            # Build dummy with spatial dims only, add channel dim of 1
            if self.wavelet_transform.ndim == 2:
                dummy = torch.zeros(1, *ref_tensor.shape[-2:], device=self.device)
            else:   # ndim == 3
                dummy = torch.zeros(1, *ref_tensor.shape[-3:], device=self.device)
            coeffs = self.wavelet_transform.forward(dummy[0])  # drop batch
            ref_spatial = coeffs[0].shape   # e.g., (H', W') or (T', H', W')

        # ---- Process each condition component --------------------------------
        wavelets = []
        for key, val in condition.items():
            wav = self._condition_to_wavelet(val, ref_spatial)
            wavelets.append(wav)

        return torch.cat(wavelets, dim=1)  # channel‑wise concatenation

    def _condition_to_wavelet(
        self,
        tensor: torch.Tensor,
        target_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """Convert a single physical condition tensor to flat wavelet coefficients.

        Args:
            tensor: input tensor (may contain a batch dimension).
            target_shape: desired spatial wavelet shape for the output subbands.

        Returns:
            Tensor of shape ``(B, C_out, *target_shape)``.
        """
        # Ensure there is a batch dimension
        no_batch = False
        if tensor.dim() > 0 and tensor.dim() != self.wavelet_transform.ndim:
            # If dim == 1 (scalar sequence) or dim == self.wavelet_transform.ndim+1 (batch+spatial)
            pass
        else:
            # Add batch dim if missing
            if tensor.dim() != self.wavelet_transform.ndim + 1:
                tensor = tensor.unsqueeze(0)
            no_batch = tensor.shape[0] == 1

        B = tensor.shape[0]

        # ---- 1D signals (e.g., initial condition in Burgers') ---------------
        if tensor.dim() == 2 and tensor.shape[-1] == target_shape[-1]:
            # shape (B, L) where L == target_shape[-1]
            # Apply 1D wavelet per sample
            coeff_list_1d = []
            for b in range(B):
                cA, cD = _wavedec1d(
                    tensor[b],
                    wavelet=self.wavelet_transform.wavelet,
                    mode=self.wavelet_transform.mode,
                )
                coeff_list_1d.append([cA, cD])
            # Stack into (B, 2, *target_shape) by repeating each coefficient
            repeated = []
            for b in range(B):
                repeated.append(
                    _repeat_1d_coeffs_to_target(coeff_list_1d[b], target_shape).unsqueeze(0)
                )
            return torch.cat(repeated, dim=0)   # (B, 2, *target_shape)

        # ---- 2D fields (for 2D wavelet transforms) ---------------------------
        if self.wavelet_transform.ndim == 2:
            # Expected input: (B, H, W) or (B, C, H, W)
            if tensor.dim() == 3:   # (B, H, W)
                tensor = tensor.unsqueeze(1)   # (B, 1, H, W)
            # Now (B, C, H, W)
            C_in = tensor.shape[1]
            coeffs_per_ch = []
            for b in range(B):
                for c in range(C_in):
                    coeffs = self.wavelet_transform.forward(tensor[b, c])  # list of 4 tensors
                    coeffs_per_ch.append(coeffs)
            # Reorganise: for each subband index, stack over batch and channel
            K = 4
            stacked = []
            for k in range(K):
                # For each sample and channel, take k-th coefficient
                parts = []
                for b in range(B):
                    for c in range(C_in):
                        idx = b * C_in + c
                        parts.append(coeffs_per_ch[idx][k])  # shape (*target_shape)
                # Stack to (B*C_in, *target_shape), then reshape to (B, C_in, *target_shape)
                parts = torch.stack(parts, dim=0)
                parts = parts.view(B, C_in, *target_shape)
                stacked.append(parts)
            # Stack subbands along channel dim -> (B, C_in*K, *target_shape)
            return torch.cat(stacked, dim=1)

        # ---- 3D fields (for 3D wavelet transforms) ---------------------------
        if self.wavelet_transform.ndim == 3:
            # Expected input: (B, T, H, W) or (B, C, T, H, W)
            if tensor.dim() == 4:   # (B, T, H, W)
                tensor = tensor.unsqueeze(1)   # (B, 1, T, H, W)
            # Now (B, C, T, H, W)
            C_in = tensor.shape[1]
            coeffs_per_ch = []
            for b in range(B):
                for c in range(C_in):
                    coeffs = self.wavelet_transform.forward(tensor[b, c])  # list of 8 tensors
                    coeffs_per_ch.append(coeffs)
            K = 8
            stacked = []
            for k in range(K):
                parts = []
                for b in range(B):
                    for c in range(C_in):
                        idx = b * C_in + c
                        parts.append(coeffs_per_ch[idx][k])  # shape (*target_shape)
                parts = torch.stack(parts, dim=0)
                parts = parts.view(B, C_in, *target_shape)
                stacked.append(parts)
            return torch.cat(stacked, dim=1)

        raise RuntimeError(
            f"Unsupported tensor shape {tensor.shape} for wavelet ndim={self.wavelet_transform.ndim}."
        )

    @torch.no_grad()
    def _ddim_step(
        self,
        x: torch.Tensor,
        noise_pred: torch.Tensor,
        t_cur: torch.Tensor,
        t_next: torch.Tensor,
        alpha_bar_t: torch.Tensor,
        eta: float,
    ) -> torch.Tensor:
        """Single DDIM reverse step.

        Args:
            x: current noisy sample ``x_t``.
            noise_pred: predicted noise ``ε`` (guided or modified).
            t_cur, t_next: scalar timestep indices (broadcastable to batch).
            alpha_bar_t: precomputed ``\bar{α}_t`` for current step.
            eta: stochasticity coefficient (0 = deterministic).

        Returns:
            Next sample ``x_{t_next}``.
        """
        batch_size = x.shape[0]
        # Expand t scalars to batch size if needed
        if t_cur.dim() == 0:
            t_cur = t_cur.expand(batch_size)
        if t_next.dim() == 0:
            t_next = t_next.expand(batch_size)

        alpha_bar_next = _extract(self.brm.alphas_cumprod, t_next, x.shape)

        # When next step is -1 (final step), set ᾱ = 1 and σ = 0
        final_step = (t_next < 0)
        if final_step.any():
            alpha_bar_next = torch.where(
                final_step.reshape_as(alpha_bar_next),
                torch.ones_like(alpha_bar_next),
                alpha_bar_next,
            )
            sigma = torch.zeros_like(alpha_bar_next)
        else:
            sigma = eta * (
                (1 - alpha_bar_next) / (1 - alpha_bar_t)
            ).sqrt() * (1 - alpha_bar_t / alpha_bar_next).sqrt()

        # Predicted x0
        x0_hat = (x - (1 - alpha_bar_t).sqrt() * noise_pred) / alpha_bar_t.sqrt()

        # Direction pointing back to x_t
        dir_xt = (1 - alpha_bar_next - sigma**2).sqrt() * noise_pred

        # Random noise (zero if eta == 0)
        z = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)

        # Next sample
        x_next = alpha_bar_next.sqrt() * x0_hat + dir_xt + sigma * z
        return x_next

    def _wavelet_to_physical(self, wavelet_flat: torch.Tensor) -> torch.Tensor:
        """Convert flat wavelet coefficients to physical space.

        Args:
            wavelet_flat: tensor of shape ``(B, C, *spatial_w)`` where
                ``C = num_physical_channels * num_subbands``.

        Returns:
            Physical tensor of shape ``(B, num_physical_channels, *original_spatial)``.
        """
        B, C, *spatial_w = wavelet_flat.shape
        K = self.num_subbands
        phy_channels = C // K

        phys = []
        for b in range(B):
            # Split into subband slices: each list element (phy_channels, *spatial_w)
            coeffs = [
                wavelet_flat[b, i * phy_channels : (i + 1) * phy_channels]
                for i in range(K)
            ]
            # Reconstruct per physical channel
            chs = []
            for c in range(phy_channels):
                sub = [coeffs[k][c] for k in range(K)]  # list of K tensors of shape spatial_w
                rec = self.wavelet_transform.inverse(sub)
                # rec may need cropping to match original physical shape (often equal)
                # We'll trust that the wavelet transform preserves size.
                chs.append(rec)
            phys.append(torch.stack(chs, dim=0))   # (phy_channels, *original_spatial)

        return torch.stack(phys, dim=0)   # (B, phy_channels, *original_spatial)


# ------------------------------------------------------------------------------
# 1D wavelet helper (replicates dataset.py logic)
# ------------------------------------------------------------------------------

def _wavedec1d(
    x: torch.Tensor,
    wavelet: str,
    mode: str = "periodization",
) -> List[torch.Tensor]:
    """1D discrete wavelet transform returning [cA, cD].

    Uses the ``pywt`` interface via ``ptwt`` (PyTorch Wavelet Toolbox).
    The wavelet object must be compatible with ``ptwt.wavedec``.

    Args:
        x: input 1D tensor, shape ``(L,)``.
        wavelet: wavelet name, e.g., ``'bior2.4'``.
        mode: signal extension mode.

    Returns:
        [cA, cD] each of half the length of ``x``.
    """
    import ptwt
    coeffs = ptwt.wavedec(x, wavelet, level=1, mode=mode)
    # wavedec returns [cA_n, cD_n, cD_{n-1}, ...] for level=1 => [cA, cD]
    return [coeffs[0], coeffs[1]]


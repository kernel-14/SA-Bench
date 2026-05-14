## evaluation/metrics.py
"""Evaluation metrics for WDNO (Wavelet Diffusion Neural Operator).

This module implements all evaluation metrics used in the WDNO paper.
It is a pure computation module with no dependencies on model or training
code — only torch is needed. Every method operates on tensors and returns
scalar floats or tensors.

Paper sources:
    - MSE (simulation): Section 4, Table 1 — "MSE measured on entire state
      sequences excluding initial conditions"
    - MAE, L∞: Appendix C.1, Table 5 — additional metrics for compressible NS
    - Relative L2: Section 4.5 (ERA5), Appendix A (wavelet reconstruction)
    - Control objective (1D): Section 4.1, Eq. 6
    - Smoke leakage (2D): Section 4.4, Table 2b
    - Per-timestep MSE: Figure 5a, Figure 9 (long-term dependency analysis)

Config references:
    - evaluation.exclude_initial_condition: true
    - evaluation.simulation_metrics: [mse, mae, l_inf, relative_l2]
    - evaluation.control_metrics: [mean_J, std_J]
    - evaluation.compute_per_timestep_error: true
    - data.burgers.control_alpha: 0.1
    - data.burgers.nx_coarse: 120  (dx = 1/(nx_coarse-1) = 1/119)
    - data.burgers.nt_coarse: 80   (dt = T/nt_coarse = 8.0/80 = 0.1)
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Small epsilon values for numerical stability
_EPS_RELATIVE_L2: float = 1e-10
_EPS_SMOKE_LEAKAGE: float = 1e-8


class Metrics:
    """Stateless utility class implementing all WDNO evaluation metrics.

    All methods are instance methods (no state) to match the design spec
    and allow consistent usage by the Evaluator class. The caller is
    responsible for passing correctly shaped tensors and appropriate
    scalar parameters (alpha, dx, dt) derived from the config.

    Shape conventions:
        1D PDE experiments (Burgers', advection, compressible NS):
            State u:  [N, T+1, X]  e.g. [N, 81, 120] including t=0
            Force f:  [N, T, X]    e.g. [N, 80, 120]
            pred/target passed to metrics: [N, T+1, X] or [N, T, X]

        2D PDE experiments (fluid_2d, ERA5):
            State:    [N, T, H, W]  e.g. [N, 32, 64, 64]
            Smoke:    [N, T]        percentage scalar per timestep
                   or [N, T, H, W] density field

    Paper convention (config: evaluation.exclude_initial_condition=true):
        MSE is computed on state sequences EXCLUDING the initial condition
        (t=0 slice). When exclude_ic=True, the time dimension is sliced
        from index 1 onward before computing the metric.
    """

    def __init__(self) -> None:
        """Initialize the Metrics instance.

        No state to initialize. This class is purely functional.
        """
        pass

    # -----------------------------------------------------------------------
    # Core simulation metrics
    # -----------------------------------------------------------------------

    def mse(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        exclude_ic: bool = True,
    ) -> float:
        """Compute Mean Squared Error between predicted and target state sequences.

        Paper Section 4: "We report the Mean Squared Error (MSE) measured on
        entire state sequences excluding initial conditions."

        Config: evaluation.exclude_initial_condition=true.

        Args:
            pred: Predicted state tensor. Shape [N, T+1, X] for 1D PDEs
                (e.g., [N, 81, 120] for Burgers') or [N, T, H, W] for 2D
                PDEs (e.g., [N, 32, 64, 64] for fluid_2d). dtype=float32.
                The time dimension is dim=1 for all experiments.
            target: Ground-truth state tensor. Must have the same shape as
                pred. dtype=float32.
            exclude_ic: If True, exclude the initial condition (t=0 slice
                at dim=1) before computing MSE. Default True per paper
                convention. Config: evaluation.exclude_initial_condition=true.
                Set to False for super-resolution evaluation where the IC
                has already been excluded by the pipeline.

        Returns:
            Scalar MSE value as a Python float. Mean over all elements
            (N, T, X or N, T, H, W) after optional IC exclusion.

        Raises:
            ValueError: If pred and target have different shapes.

        Example:
            >>> metrics = Metrics()
            >>> pred = torch.randn(10, 81, 120)   # Burgers' predictions
            >>> target = torch.randn(10, 81, 120)  # Ground truth
            >>> mse_val = metrics.mse(pred, target, exclude_ic=True)
            >>> # Computed on pred[:, 1:, :] vs target[:, 1:, :] (80 steps)
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must have the same shape. "
                f"Got pred.shape={tuple(pred.shape)} and "
                f"target.shape={tuple(target.shape)}."
            )

        # Exclude initial condition (t=0) if requested
        # Time dimension is always dim=1 for all WDNO experiments
        if exclude_ic and pred.shape[1] > 1:
            pred = pred[:, 1:, ...]
            target = target[:, 1:, ...]

        # Compute MSE over all elements
        mse_val: torch.Tensor = torch.mean((pred - target) ** 2)
        return float(mse_val.item())

    def mae(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> float:
        """Compute Mean Absolute Error between predicted and target tensors.

        Paper Appendix C.1, Table 5: MAE column for 1D compressible NS
        comparisons. Also used in Figure 9 for per-timestep MAE visualization
        of WDNO vs DDPM on 1D Burgers'.

        The caller (Evaluator) is responsible for any IC exclusion before
        calling this method. This method computes MAE on the full input.

        Args:
            pred: Predicted tensor of arbitrary shape. dtype=float32.
            target: Ground-truth tensor. Must have the same shape as pred.
                dtype=float32.

        Returns:
            Scalar MAE value as a Python float. Mean of absolute differences
            over all elements.

        Raises:
            ValueError: If pred and target have different shapes.
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must have the same shape. "
                f"Got pred.shape={tuple(pred.shape)} and "
                f"target.shape={tuple(target.shape)}."
            )

        mae_val: torch.Tensor = torch.mean(torch.abs(pred - target))
        return float(mae_val.item())

    def l_inf(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> float:
        """Compute L∞ (maximum absolute) error between predicted and target tensors.

        Paper Appendix C.1, Table 5: "L∞ Error" column. The paper notes:
        "the L∞ error values across different methods are relatively similar
        because this metric only considers the maximum value across the entire
        spatiotemporal domain, thus capturing less information."

        Computes the global maximum absolute error over ALL elements (all N
        samples, all T timesteps, all spatial points). This is a single
        worst-case value, not a per-sample maximum.

        Args:
            pred: Predicted tensor of arbitrary shape. dtype=float32.
            target: Ground-truth tensor. Must have the same shape as pred.
                dtype=float32.

        Returns:
            Scalar L∞ error as a Python float. Global maximum of
            |pred - target| over all elements.

        Raises:
            ValueError: If pred and target have different shapes.
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must have the same shape. "
                f"Got pred.shape={tuple(pred.shape)} and "
                f"target.shape={tuple(target.shape)}."
            )

        l_inf_val: torch.Tensor = torch.max(torch.abs(pred - target))
        return float(l_inf_val.item())

    def relative_l2(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> float:
        """Compute relative L2 error: ||pred - target||_2 / ||target||_2.

        Paper Section 4.5 (ERA5): "relative L2 error as low as 0.0161".
        Paper Appendix A (wavelet reconstruction): "relative l2 errors of
        such reconstructions are on the order of 1e-7".

        Computes the ratio of the L2 norm of the error to the L2 norm of
        the target, over all elements in the batch. This gives a single
        relative error value for the entire batch.

        Args:
            pred: Predicted tensor of arbitrary shape. dtype=float32.
            target: Ground-truth tensor. Must have the same shape as pred.
                dtype=float32.

        Returns:
            Scalar relative L2 error as a Python float.
            Returns 0.0 if ||target||_2 < eps (near-zero target).

        Raises:
            ValueError: If pred and target have different shapes.
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must have the same shape. "
                f"Got pred.shape={tuple(pred.shape)} and "
                f"target.shape={tuple(target.shape)}."
            )

        numerator: torch.Tensor = torch.norm(pred - target, p=2)
        denominator: torch.Tensor = torch.norm(target, p=2)

        if denominator.item() < _EPS_RELATIVE_L2:
            logger.warning(
                "relative_l2: target norm is near zero (%.2e < %.2e). "
                "Returning 0.0 to avoid division by zero.",
                float(denominator.item()),
                _EPS_RELATIVE_L2,
            )
            return 0.0

        return float((numerator / denominator).item())

    # -----------------------------------------------------------------------
    # Control objective metrics
    # -----------------------------------------------------------------------

    def control_objective_1d(
        self,
        u_T: torch.Tensor,
        u_star: torch.Tensor,
        f: torch.Tensor,
        alpha: float,
        dx: float,
        dt: float,
    ) -> float:
        """Compute the 1D Burgers' control objective I.

        Paper Section 4.1, Equation 6:
            I = integral_D |u(T,x) - u*(x)|^2 dx
              + alpha * integral_{[0,T]×D} |f(t,x)|^2 dt dx

        Critical (paper Appendix F.1): u_T must be the ground-truth solver
        output given the generated control force f, NOT the model's predicted
        state. The Controller class is responsible for running the solver
        before calling this method.

        Numerical integration uses the trapezoidal rule via torch.trapezoid.

        Config alignment:
            alpha = 0.1 (config: data.burgers.control_alpha)
            dx = 1/(nx_coarse-1) = 1/119 ≈ 0.0084 (config: data.burgers.nx_coarse=120)
            dt = T/nt_coarse = 8.0/80 = 0.1 (config: data.burgers.T=8.0,
                                               data.burgers.nt_coarse=80)

        Args:
            u_T: Final state from ground-truth solver at t=T. Shape [N, X]
                e.g. [N, 120] for Burgers'. dtype=float32.
                This is u_{g.t.}(T, x) from the solver, not the model output.
            u_star: Target state. Shape [N, X] or [X] (broadcast to batch).
                e.g. [N, 120] or [120]. dtype=float32.
            f: Control force sequence. Shape [N, T, X] e.g. [N, 80, 120].
                dtype=float32.
            alpha: Weight of control energy regularization. From config:
                data.burgers.control_alpha=0.1.
            dx: Spatial grid spacing for trapezoidal integration.
                = 1.0 / (X - 1) = 1/119 for X=120.
            dt: Temporal grid spacing for trapezoidal integration.
                = T / nt_coarse = 8.0 / 80 = 0.1.

        Returns:
            Mean control objective I over the N test samples as a Python float.
            Paper Table 2a: WDNO achieves I=0.0205 on 1D Burgers'.

        Raises:
            ValueError: If u_T and u_star have incompatible shapes.
            ValueError: If f has wrong number of dimensions.
        """
        if u_T.dim() != 2:
            raise ValueError(
                f"u_T must be 2D [N, X], got shape {tuple(u_T.shape)}."
            )
        if f.dim() != 3:
            raise ValueError(
                f"f must be 3D [N, T, X], got shape {tuple(f.shape)}."
            )

        N: int = u_T.shape[0]
        X: int = u_T.shape[1]

        # Broadcast u_star to match u_T shape if needed
        if u_star.dim() == 1:
            # [X] → [N, X]
            u_star_expanded: torch.Tensor = u_star.unsqueeze(0).expand(N, -1)
        elif u_star.dim() == 2:
            if u_star.shape[0] == 1 and N > 1:
                # [1, X] → [N, X]
                u_star_expanded = u_star.expand(N, -1)
            elif u_star.shape[0] == N:
                u_star_expanded = u_star
            else:
                raise ValueError(
                    f"u_star shape {tuple(u_star.shape)} is incompatible with "
                    f"u_T shape {tuple(u_T.shape)}. Expected [N, X] or [X]."
                )
        else:
            raise ValueError(
                f"u_star must be 1D [X] or 2D [N, X], got shape {tuple(u_star.shape)}."
            )

        # Ensure all tensors are on the same device and dtype
        device: torch.device = u_T.device
        u_star_expanded = u_star_expanded.to(device=device, dtype=u_T.dtype)
        f = f.to(device=device, dtype=u_T.dtype)

        # -----------------------------------------------------------------------
        # Term 1: integral_D |u(T,x) - u*(x)|^2 dx
        # Trapezoidal integration over spatial dimension X
        # integrand_1: [N, X]
        # -----------------------------------------------------------------------
        integrand_1: torch.Tensor = (u_T - u_star_expanded) ** 2  # [N, X]

        # torch.trapezoid integrates along the last dimension by default
        # dx is the uniform spacing between grid points
        integral_1: torch.Tensor = torch.trapezoid(
            integrand_1, dx=dx, dim=-1
        )  # [N]

        # -----------------------------------------------------------------------
        # Term 2: alpha * integral_{[0,T]×D} |f(t,x)|^2 dt dx
        # First integrate over space (dim=-1), then over time (dim=-1 again)
        # integrand_2: [N, T, X]
        # -----------------------------------------------------------------------
        integrand_2: torch.Tensor = f ** 2  # [N, T, X]

        # Integrate over spatial dimension X: [N, T, X] → [N, T]
        integral_2_space: torch.Tensor = torch.trapezoid(
            integrand_2, dx=dx, dim=-1
        )  # [N, T]

        # Integrate over temporal dimension T: [N, T] → [N]
        integral_2: torch.Tensor = torch.trapezoid(
            integral_2_space, dx=dt, dim=-1
        )  # [N]

        # -----------------------------------------------------------------------
        # Total objective: I = term1 + alpha * term2, averaged over N samples
        # -----------------------------------------------------------------------
        I_per_sample: torch.Tensor = integral_1 + alpha * integral_2  # [N]
        I_mean: float = float(I_per_sample.mean().item())

        logger.debug(
            "control_objective_1d: N=%d, mean_I=%.6f, "
            "mean_state_dev=%.6f, mean_control_energy=%.6f",
            N,
            I_mean,
            float(integral_1.mean().item()),
            float((alpha * integral_2).mean().item()),
        )

        return I_mean

    def smoke_leakage_2d(
        self,
        smoke_trajectory: torch.Tensor,
        target_bucket_mask: torch.Tensor,
    ) -> float:
        """Compute the 2D fluid control objective I (smoke leakage percentage).

        Paper Section 4.4: "I is defined as the percentage of smoke not
        passing through the target bucket." The control goal is to minimize I
        (maximize smoke reaching the target bucket).

        Paper Table 2b: WDNO achieves I=0.0679 (≈6.79% leakage, meaning
        ≈93.21% of smoke reaches the target bucket).

        Supports two input formats for smoke_trajectory:
            1. Density field: [N, T, H, W] — smoke density at each timestep
            2. Percentage scalars: [N, T] — pre-computed percentage through
               bucket at each timestep (target_bucket_mask is ignored)

        For the density field format, the leakage is computed at the final
        timestep (t=T-1) as the fraction of total smoke NOT in the bucket.

        Args:
            smoke_trajectory: Smoke data. Either:
                - Density field: shape [N, T, H, W] e.g. [N, 32, 64, 64].
                  Smoke density values at each grid cell and timestep.
                - Percentage scalars: shape [N, T] e.g. [N, 32].
                  Pre-computed percentage of smoke through bucket per step.
                dtype=float32.
            target_bucket_mask: Boolean or float mask indicating the target
                bucket region. Shape [H, W] e.g. [64, 64]. True/1.0 where
                the target bucket is located. Used only when smoke_trajectory
                is a density field (4D input). Ignored for 2D input.
                dtype=bool or float32.

        Returns:
            Scalar smoke leakage I as a Python float. Mean over N samples.
            I = 1 - (smoke in bucket / total smoke) at final timestep.
            Range [0, 1]. Lower is better.
            Paper Table 2b: WDNO achieves I=0.0679.

        Raises:
            ValueError: If smoke_trajectory has unexpected number of dimensions.
        """
        device: torch.device = smoke_trajectory.device

        if smoke_trajectory.dim() == 2:
            # Pre-computed percentage scalars: [N, T]
            # smoke_trajectory[:, t] = fraction of smoke through bucket at step t
            # Use final timestep value
            smoke_pct_final: torch.Tensor = smoke_trajectory[:, -1]  # [N]
            leakage_per_sample: torch.Tensor = 1.0 - smoke_pct_final.clamp(0.0, 1.0)
            leakage_mean: float = float(leakage_per_sample.mean().item())

            logger.debug(
                "smoke_leakage_2d (percentage input): N=%d, mean_leakage=%.6f",
                smoke_trajectory.shape[0],
                leakage_mean,
            )
            return leakage_mean

        elif smoke_trajectory.dim() == 4:
            # Density field: [N, T, H, W]
            N: int = smoke_trajectory.shape[0]
            T: int = smoke_trajectory.shape[1]
            H: int = smoke_trajectory.shape[2]
            W: int = smoke_trajectory.shape[3]

            # Extract smoke density at final timestep: [N, H, W]
            smoke_final: torch.Tensor = smoke_trajectory[:, -1, :, :]  # [N, H, W]

            # Move mask to same device and convert to float
            bucket_mask: torch.Tensor = target_bucket_mask.to(
                device=device, dtype=smoke_final.dtype
            )  # [H, W]

            # Validate mask shape
            if bucket_mask.shape != (H, W):
                raise ValueError(
                    f"target_bucket_mask shape {tuple(bucket_mask.shape)} does not "
                    f"match smoke field spatial dimensions ({H}, {W})."
                )

            # Smoke in target bucket at final timestep: [N]
            # Sum over spatial dims (H, W) after masking
            smoke_in_bucket: torch.Tensor = (
                smoke_final * bucket_mask.unsqueeze(0)
            ).sum(dim=(-2, -1))  # [N]

            # Total smoke at final timestep: [N]
            total_smoke: torch.Tensor = smoke_final.sum(dim=(-2, -1))  # [N]

            # Percentage of smoke NOT in bucket (leakage)
            # Guard against near-zero total smoke
            leakage_per_sample = 1.0 - smoke_in_bucket / (
                total_smoke + _EPS_SMOKE_LEAKAGE
            )  # [N]

            # Clamp to [0, 1] for numerical stability
            leakage_per_sample = leakage_per_sample.clamp(0.0, 1.0)

            leakage_mean = float(leakage_per_sample.mean().item())

            logger.debug(
                "smoke_leakage_2d (density input): N=%d, T=%d, H=%d, W=%d, "
                "mean_leakage=%.6f, mean_smoke_in_bucket=%.6f, "
                "mean_total_smoke=%.6f",
                N, T, H, W,
                leakage_mean,
                float(smoke_in_bucket.mean().item()),
                float(total_smoke.mean().item()),
            )
            return leakage_mean

        else:
            raise ValueError(
                f"smoke_trajectory must be 2D [N, T] (percentage scalars) or "
                f"4D [N, T, H, W] (density field), got {smoke_trajectory.dim()}D "
                f"tensor with shape {tuple(smoke_trajectory.shape)}."
            )

    # -----------------------------------------------------------------------
    # Per-timestep analysis
    # -----------------------------------------------------------------------

    def per_timestep_mse(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE at each timestep for long-term dependency analysis.

        Paper Figure 5a: "errors of baselines and WDNO at different time steps
        in the 2D simulation experiment." Shows that WDNO exhibits the slowest
        error growth, confirming its ability to capture long-term dependencies.

        Paper Figure 9: "WDNO's and DDPM's MAE of different time steps" on
        1D Burgers'. Shows WDNO achieves lower error at moments of abrupt changes.

        Config: evaluation.compute_per_timestep_error=true.

        The caller decides whether to pass the full sequence (including t=0)
        or exclude it. For visualization purposes (Figure 5a, Figure 9),
        the full sequence is typically passed to show error starting from 0.

        Args:
            pred: Predicted state tensor. Shape [N, T, X] for 1D PDEs or
                [N, T, H, W] for 2D PDEs. The time dimension is dim=1.
                dtype=float32.
            target: Ground-truth state tensor. Must have the same shape as
                pred. dtype=float32.

        Returns:
            Per-timestep MSE tensor of shape [T]. Element t contains the
            MSE at timestep t, averaged over the batch N and all spatial
            dimensions (X or H×W). Kept on the same device as input.
            The Evaluator converts this to numpy for matplotlib plotting.

        Raises:
            ValueError: If pred and target have different shapes.
            ValueError: If pred has fewer than 2 dimensions.
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must have the same shape. "
                f"Got pred.shape={tuple(pred.shape)} and "
                f"target.shape={tuple(target.shape)}."
            )
        if pred.dim() < 2:
            raise ValueError(
                f"pred must have at least 2 dimensions [N, T, ...], "
                f"got {pred.dim()}D tensor with shape {tuple(pred.shape)}."
            )

        # Squared error: same shape as pred
        sq_err: torch.Tensor = (pred - target) ** 2

        # Mean over batch dimension (dim=0) and all spatial dimensions (dim >= 2)
        # Result shape: [T]
        # For 1D: sq_err [N, T, X] → mean over dims 0 and 2 → [T]
        # For 2D: sq_err [N, T, H, W] → mean over dims 0, 2, 3 → [T]
        T: int = sq_err.shape[1]

        # Collect all non-time dimensions to average over
        # dim=0 is batch, dim=1 is time, dims 2+ are spatial
        dims_to_average = [0] + list(range(2, sq_err.dim()))

        per_t_mse: torch.Tensor = sq_err.mean(dim=dims_to_average)  # [T]

        logger.debug(
            "per_timestep_mse: T=%d, min_mse=%.6f at t=%d, max_mse=%.6f at t=%d",
            T,
            float(per_t_mse.min().item()),
            int(per_t_mse.argmin().item()),
            float(per_t_mse.max().item()),
            int(per_t_mse.argmax().item()),
        )

        return per_t_mse

## Code: inference/control.py

```python
## inference/control.py
"""Control inference module for WDNO (Wavelet Diffusion Neural Operator).

This module implements the Controller class that handles the control inference
task for WDNO. It combines classifier-free guidance (CFG) from the diffusion
model with classifier-based guidance from the control objective I to generate
optimal control sequences.

Key design (paper Section 3.1, Eq. 4):
    W_f_hat^(k) = (W_f^(k) - sqrt(1-alpha_bar_k)*eps_theta) / sqrt(alpha_bar_k)
    W_f^(k-1) = W_f^(k) - eta*(eps_theta + lambda*nabla_I(W_f_hat^(k))) + xi

Critical: the gradient nabla_I is computed on the denoised estimate W_f_hat,
not the noisy W_f^(k), to avoid noise contaminating the gradient signal.

Paper sources:
    - Control inference: Section 3.1, Eq. 4, Algorithm 1
    - Control objective (1D Burgers'): Section 4.1, Eq. 6
    - Control objective (2D fluid): Section 4.4
    - Evaluation protocol: Appendix F.1 (always use ground-truth solver)
    - Guidance schedule: Table 18 ("Scheduler of guidance: cosine")
    - Interpolation for solver: Appendix F.1

Config references:
    - inference.burgers.guidance_lambda: 120000
    - inference.burgers.guidance_schedule: cosine
    - inference.burgers.ddim_steps: 50
    - inference.burgers.ddim_eta: 1.0
    - inference.fluid_2d.guidance_lambda: 100
    - inference.fluid_2d.ddim_steps: 100
    - data.burgers.control_alpha: 0.1
    - data.burgers.num_test_control: 50
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F

from config import Config
from models.wdno_pipeline import WDNOPipeline
from wavelet.wavelet_transform import WaveletTransform

logger = logging.getLogger(__name__)


class Controller:
    """Control inference wrapper for WDNO.

    Generates optimal control sequences by combining classifier-free guidance
    (learned from training data distribution p(f|a)) with classifier-based
    guidance (from the control objective I). The two guidance mechanisms
    complement each other: CFG ensures the generated control is physically
    plausible, while classifier guidance steers it toward lower objective I.

    The final control objective is always evaluated using the ground-truth
    PDE solver output, never the model's predicted state (paper Appendix F.1).

    Attributes:
        pipeline: Fully initialized WDNOPipeline with loaded BRM checkpoint.
            Provides the control() method that runs the guided DDIM loop.
        wavelet_transform: WaveletTransform instance configured for the
            current experiment. Used to transform conditioning inputs and
            to reconstruct physical-space control from wavelet coefficients.
        solver_fn: Ground-truth PDE solver callable. Signature:
            solver_fn(f: Tensor) -> Tensor
            For 1D Burgers': takes f [B, T, X] and returns u_T [B, X].
            For 2D fluid: takes f [B, T, N_ctrl] and returns smoke trajectory
            [B, T, H, W] or final smoke percentage [B].
        config: Experiment configuration. Drives all inference hyperparameters.
        experiment: Experiment name string from config.experiment.
        device: Compute device string from config.device.
        spatial_dim: Spatial dimensionality of the PDE (1 or 2).
        guidance_lambda: Maximum guidance weight lambda_max.
            Config: inference.<experiment>.guidance_lambda.
        guidance_schedule: Guidance schedule type ('cosine').
            Config: inference.<experiment>.guidance_schedule.
        ddim_steps: DDIM sampling steps.
            Config: inference.<experiment>.ddim_steps.
        ddim_eta: DDIM stochasticity parameter.
            Config: inference.<experiment>.ddim_eta.
        cfg_weight: Classifier-free guidance weight.
            Config: diffusion.cfg_weight.
        control_alpha: Weight alpha in control objective I.
            Config: data.burgers.control_alpha (0.1).
        dx: Spatial step size for integration (1/(nx_coarse-1)).
        dt: Temporal step size for integration (T/nt_coarse).
        solver_nx: Solver internal spatial grid size for interpolation.
            Config: data.burgers.solver_nx (1920).
        solver_nt: Solver internal temporal grid size for interpolation.
            Config: data.burgers.solver_nt (76800).
        nx_coarse: Coarse spatial grid size (120).
        nt_coarse: Coarse temporal steps (80).
    """

    def __init__(
        self,
        pipeline: WDNOPipeline,
        wavelet_transform: WaveletTransform,
        solver_fn: Callable[[torch.Tensor], torch.Tensor],
        config: Config,
    ) -> None:
        """Initialize the Controller.

        Extracts all inference hyperparameters from config and stores the
        injected dependencies. No model initialization occurs here.

        Args:
            pipeline: Fully initialized WDNOPipeline with loaded BRM
                checkpoint. The BRM model must be trained for the control
                task (conditioned on u_0 and u_star for 1D Burgers', or
                on initial density for 2D fluid).
            wavelet_transform: WaveletTransform instance configured for the
                current experiment. Must match the instance used during
                training (same wavelet type, mode, level, spatial_dim).
                Config: wavelet.<experiment>.wavelet_type,
                wavelet.<experiment>.padding_mode.
            solver_fn: Ground-truth PDE solver callable. For 1D Burgers':
                wraps the finite difference solver from BurgersGenerator.
                For 2D fluid: wraps the incompressible NS simulator.
                Signature: solver_fn(f: Tensor) -> Tensor.
                The solver is used for final objective evaluation only
                (not during the DDIM guidance loop, since it is not
                differentiable through PyTorch autograd).
            config: Experiment configuration. Reads inference hyperparameters
                from config.inference[config.experiment] and data parameters
                from config.data[config.experiment].
        """
        self.pipeline: WDNOPipeline = pipeline
        self.wavelet_transform: WaveletTransform = wavelet_transform
        self.solver_fn: Callable[[torch.Tensor], torch.Tensor] = solver_fn
        self.config: Config = config

        # Experiment identification
        self.experiment: str = config.experiment
        self.device: str = config.device
        self.spatial_dim: int = config.spatial_dim

        # Inference hyperparameters from config
        self.guidance_lambda: float = config.guidance_lambda
        self.guidance_schedule: str = config.guidance_schedule
        self.ddim_steps: int = config.ddim_steps
        self.ddim_eta: float = config.ddim_eta
        self.cfg_weight: float = config.cfg_weight

        # Data parameters for objective computation
        self.control_alpha: float = config.control_alpha
        self.nx_coarse: int = config.nx_coarse
        self.nt_coarse: int = config.nt_coarse

        # Spatial and temporal step sizes for numerical integration
        # dx = 1 / (nx_coarse - 1) for domain [0, 1]
        # dt = T / nt_coarse for time horizon T
        self.dx: float = 1.0 / max(self.nx_coarse - 1, 1)
        self.dt: float = config.T / max(self.nt_coarse, 1)

        # Solver internal grid sizes for interpolation (Appendix F.1)
        self.solver_nx: int = config.solver_nx   # 1920 = 120 * 16
        self.solver_nt: int = config.solver_nt   # 76800 = 4800 * 16

        # Move pipeline models to device
        target_device = torch.device(self.device)
        self.pipeline.brm = self.pipeline.brm.to(target_device)
        if self.pipeline.srm is not None:
            self.pipeline.srm = self.pipeline.srm.to(target_device)

        logger.info(
            "Controller initialized: experiment=%s, spatial_dim=%d, device=%s, "
            "guidance_lambda=%.1f, guidance_schedule=%s, ddim_steps=%d, "
            "ddim_eta=%.2f, cfg_weight=%.2f, control_alpha=%.4f, "
            "dx=%.4f, dt=%.4f",
            self.experiment,
            self.spatial_dim,
            self.device,
            self.guidance_lambda,
            self.guidance_schedule,
            self.ddim_steps,
            self.ddim_eta,
            self.cfg_weight,
            self.control_alpha,
            self.dx,
            self.dt,
        )

    # -----------------------------------------------------------------------
    # Public inference method
    # -----------------------------------------------------------------------

    def run(
        self,
        cond_raw: torch.Tensor,
        target: torch.Tensor,
        alpha: float,
    ) -> Tuple[torch.Tensor, float]:
        """Run control inference to generate an optimal control sequence.

        Implements the full WDNO control pipeline:
            1. Transform conditioning data to wavelet space
            2. Build differentiable control objective function
            3. Delegate to WDNOPipeline.control() for guided DDIM sampling
            4. Reconstruct physical-space control via inverse wavelet transform
            5. Evaluate final objective using ground-truth solver

        The final control objective is always evaluated using the ground-truth
        PDE solver output, never the model's predicted state (paper Appendix F.1:
        "the state deviation in our reported evaluation metric I is always based
        on the output u(T,x) of the ground-truth solver given the control force f").

        Args:
            cond_raw: Raw conditioning tensor in physical space. Format:
                - 1D Burgers': Tensor [B, X] containing initial condition u_0.
                  The target state u* is passed separately via `target`.
                - 2D fluid: Tensor [B, H, W] containing initial smoke density.
                  Can also be a tuple (initial_density, ...) for multi-field
                  conditioning.
                Must be float32 on any device (moved to self.device internally).
            target: Target state tensor. Format:
                - 1D Burgers': u* [B, X] — desired final state at t=T.
                  Config: data.burgers.num_test_control=50 test samples.
                - 2D fluid: target_bucket_mask [H, W] — boolean mask of
                  target bucket region, or None to use stored mask.
                Must be float32.
            alpha: Weight of control energy regularization in objective I.
                Config: data.burgers.control_alpha=0.1.
                I = ||u(T)-u*||^2 + alpha * ||f||^2 (1D Burgers', Eq. 6).

        Returns:
            Tuple (f_pred, control_objective_value) where:
                f_pred: Generated control sequence in physical space.
                    Shape [B, T, X] = [B, 80, 120] for 1D Burgers'.
                    Shape [B, T, N_ctrl] = [B, 32, 3584] for 2D fluid.
                    dtype=float32.
                control_objective_value: Scalar float I evaluated using
                    the ground-truth solver. Lower is better.
                    Paper Table 2a: WDNO achieves 0.0205 on 1D Burgers'.
                    Paper Table 2b: WDNO achieves 0.0679 on 2D fluid.

        Raises:
            ValueError: If cond_raw has unexpected shape for the experiment.
        """
        device = torch.device(self.device)

        # --- Step 1: Move inputs to device ---
        if isinstance(cond_raw, torch.Tensor):
            cond_raw = cond_raw.to(device=device, dtype=torch.float32)
        target = target.to(device=device, dtype=torch.float32)

        # --- Step 2: Prepare conditioning wavelet coefficients ---
        W_cond: torch.Tensor = self._prepare_conditioning(
            cond_raw=cond_raw,
            target=target,
        )
        # W_cond: [B, C_cond, T_c, X_c] (1D) or [B, C_cond, T_c, H_c, W_c] (2D)

        logger.debug(
            "run: W_cond.shape=%s, target.shape=%s",
            tuple(W_cond.shape),
            tuple(target.shape),
        )

        # --- Step 3: Build differentiable objective function ---
        objective_fn: Callable[[torch.Tensor], torch.Tensor] = (
            self._build_objective_fn(
                target=target,
                alpha=alpha,
                dx=self.dx,
                dt=self.dt,
            )
        )

        # --- Step 4: Run guided DDIM sampling via pipeline ---
        # The pipeline handles the full DDIM loop with combined CFG +
        # classifier guidance internally. Returns wavelet coefficients
        # of the generated control sequence.
        W_f_final: torch.Tensor = self.pipeline.control(
            W_cond=W_cond,
            objective_fn=objective_fn,
            num_sr_levels=0,  # Control inference always at base resolution
        )
        # W_f_final: [B, C_f, T_c, X_c] (1D) or [B, C_f, T_c, H_c, W_c] (2D)

        logger.debug(
            "run: W_f_final.shape=%s",
            tuple(W_f_final.shape),
        )

        # --- Step 5: Inverse wavelet transform to get physical-space control ---
        # Determine original physical shape for inverse DWT
        original_physical_shape: Tuple[int, ...] = self._get_control_physical_shape(
            batch_size=W_f_final.shape[0],
        )

        with torch.no_grad():
            f_pred: torch.Tensor = self.wavelet_transform.inverse(
                W_f_final,
                original_shape=original_physical_shape,
            )
        # f_pred: [B, T, X] = [B, 80, 120] for 1D Burgers'

        logger.debug(
            "run: f_pred.shape=%s (physical space)",
            tuple(f_pred.shape),
        )

        # --- Step 6: Evaluate final control objective using ground-truth solver ---
        # Critical: always use solver output, never model's predicted state
        control_objective_value: float = self.evaluate_control_objective(
            f_pred=f_pred,
            target=target,
            solver_fn=self.solver_fn,
            alpha=alpha,
        )

        logger.info(
            "run: control_objective=%.6f (experiment=%s)",
            control_objective_value,
            self.experiment,
        )

        return f_pred.float(), control_objective_value

    # -----------------------------------------------------------------------
    # Objective function construction
    # -----------------------------------------------------------------------

    def _build_objective_fn(
        self,
        target: torch.Tensor,
        alpha: float,
        dx: float,
        dt: float,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Build a differentiable control objective function.

        Returns a callable that takes physical-space force estimate f_hat
        and returns a scalar tensor representing the control objective I.
        The returned callable is used inside WDNOPipeline._control_guidance_step
        to compute the guidance gradient.

        For 1D Burgers' (spatial_dim == 1):
            I = integral_D |u(T,x) - u*(x)|^2 dx
              + alpha * integral_{[0,T]×D} |f(t,x)|^2 dt dx

        Since the finite difference solver is not differentiable through
        PyTorch autograd, the gradient of the state deviation term w.r.t.
        f_hat is zero (solver called with no_grad). The guidance gradient
        flows primarily through the regularization term alpha*||f||^2,
        which is fully differentiable. The CFG mechanism (learned p(f|u_0,u*))
        handles the state deviation constraint implicitly.

        For 2D fluid (spatial_dim == 2):
            I = percentage of smoke NOT passing through target bucket
              + alpha * ||f||^2 (regularization)

        Args:
            target: Target state tensor. For 1D: u* [B, X]. For 2D: target
                bucket mask [H, W] or smoke percentage target.
            alpha: Control energy regularization weight.
                Config: data.burgers.control_alpha=0.1.
            dx: Spatial step size for numerical integration.
                Computed as 1/(nx_coarse-1).
            dt: Temporal step size for numerical integration.
                Computed as T/nt_coarse.

        Returns:
            Callable objective_fn(f_hat: Tensor) -> Tensor (scalar).
            The callable closes over target, alpha, dx, dt, and solver_fn.
            It must preserve the autograd computation graph through f_hat
            for the regularization term (the differentiable part).
        """
        if self.spatial_dim == 1:
            return self._build_objective_fn_1d(target, alpha, dx, dt)
        else:
            return self._build_objective_fn_2d(target, alpha, dx, dt)

    def _build_objective_fn_1d(
        self,
        target: torch.Tensor,
        alpha: float,
        dx: float,
        dt: float,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Build control objective for 1D Burgers' equation.

        Implements Eq. 6 from paper Section 4.1:
            I = integral_D |u(T,x) - u*(x)|^2 dx
              + alpha * integral_{[0,T]×D} |f(t,x)|^2 dt dx

        The state deviation term uses the solver with no_grad (non-differentiable).
        The regularization term is fully differentiable w.r.t. f_hat.

        Args:
            target: Target state u* of shape [B, X] = [B, 120]. float32.
            alpha: Regularization weight. Config: data.burgers.control_alpha=0.1.
            dx: Spatial step size (1/(nx_coarse-1) = 1/119 ≈ 0.0084).
            dt: Temporal step size (T/nt_coarse = 8.0/80 = 0.1).

        Returns:
            Callable objective_fn(f_hat: Tensor) -> Tensor (scalar).
            f_hat shape: [B, T, X] = [B, 80, 120].
        """
        # Capture variables in closure
        _target: torch.Tensor = target.detach()
        _alpha: float = alpha
        _dx: float = dx
        _dt: float = dt
        _solver_fn: Callable = self.solver_fn

        def objective_fn_1d(f_hat: torch.Tensor) -> torch.Tensor:
            """1D Burgers' control objective.

            Args:
                f_hat: Physical-space force estimate. Shape [B, T, X].
                    May have requires_grad=True (gradient flows through
                    the regularization term).

            Returns:
                Scalar tensor I = state_deviation + alpha * control_energy.
                Gradient w.r.t. f_hat is non-zero only through the
                regularization term (solver is not differentiable).
            """
            # --- Regularization term: alpha * integral |f|^2 dt dx ---
            # Fully differentiable w.r.t. f_hat
            # Numerical integration using trapezoidal rule approximation:
            # sum over all (t, x) grid points * dt * dx
            control_energy: torch.Tensor = (
                _alpha * torch.sum(f_hat ** 2) * _dt * _dx
            )

            # --- State deviation term: integral |u(T,x) - u*(x)|^2 dx ---
            # Run ground-truth solver with no_grad (non-differentiable path)
            # The gradient of this term w.r.t. f_hat is zero.
            # The CFG mechanism handles the state constraint implicitly.
            try:
                with torch.no_grad():
                    # Interpolate f_hat to solver resolution if needed
                    f_for_solver: torch.Tensor = f_hat.detach()

                    # Run solver: f [B, T, X] → u_T [B, X]
                    u_T: torch.Tensor = _solver_fn(f_for_solver)
                    u_T = u_T.to(f_hat.device)

                    # Compute state deviation (no gradient)
                    target_expanded: torch.Tensor = _target
                    if target_expanded.shape != u_T.shape:
                        # Handle batch dimension mismatch
                        if target_expanded.dim() == 1:
                            target_expanded = target_expanded.unsqueeze(0).expand_as(u_T)
                        elif target_expanded.shape[0] == 1 and u_T.shape[0] > 1:
                            target_expanded = target_expanded.expand_as(u_T)

                    state_deviation_val: float = float(
                        torch.sum((u_T - target_expanded) ** 2).item() * _dx
                    )

                # Add state deviation as a constant (no gradient through solver)
                # This provides the correct objective value for logging/monitoring
                # while the gradient signal comes from the regularization term
                state_deviation_const: torch.Tensor = torch.tensor(
                    state_deviation_val,
                    dtype=f_hat.dtype,
                    device=f_hat.device,
                    requires_grad=False,
                )
                total_objective: torch.Tensor = state_deviation_const + control_energy

            except Exception as exc:
                logger.warning(
                    "Solver call failed in objective_fn_1d: %s. "
                    "Using regularization term only for guidance.",
                    exc,
                )
                # Fall back to regularization-only objective
                total_objective = control_energy

            return total_objective

        return objective_fn_1d

    def _build_objective_fn_2d(
        self,
        target: torch.Tensor,
        alpha: float,
        dx: float,
        dt: float,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Build control objective for 2D incompressible fluid.

        Implements the smoke leakage objective from paper Section 4.4:
            I = percentage of smoke NOT passing through target bucket

        The control objective I is defined as the percentage of smoke not
        reaching the target bucket (minimize I = maximize smoke in bucket).
        Paper Table 2b: WDNO achieves I=0.0679 (6.79% smoke leakage).

        Args:
            target: Target bucket mask [H, W] boolean tensor, or None.
                If None, uses a default top-center bucket mask.
            alpha: Regularization weight for control energy.
            dx: Spatial step size (not used for 2D fluid objective).
            dt: Temporal step size (not used for 2D fluid objective).

        Returns:
            Callable objective_fn(f_hat: Tensor) -> Tensor (scalar).
            f_hat shape: [B, T, N_ctrl] = [B, 32, 3584] or [B, T, H, W].
        """
        # Capture variables in closure
        _target: torch.Tensor = target.detach() if target is not None else None
        _alpha: float = alpha
        _solver_fn: Callable = self.solver_fn

        def objective_fn_2d(f_hat: torch.Tensor) -> torch.Tensor:
            """2D fluid control objective (smoke leakage).

            Args:
                f_hat: Physical-space control estimate. Shape [B, T, N_ctrl]
                    or [B, T, H_ctrl, W_ctrl].

            Returns:
                Scalar tensor I = smoke_leakage + alpha * control_energy.
            """
            # --- Control energy regularization (differentiable) ---
            control_energy: torch.Tensor = _alpha * torch.sum(f_hat ** 2)

            # --- Smoke leakage (non-differentiable, uses solver) ---
            try:
                with torch.no_grad():
                    # Run fluid simulator: f → smoke trajectory
                    smoke_result = _solver_fn(f_hat.detach())

                    if isinstance(smoke_result, torch.Tensor):
                        if smoke_result.dim() == 1:
                            # Solver returns final smoke percentage directly
                            # smoke_result: [B] — percentage reaching bucket
                            smoke_pct_final: torch.Tensor = smoke_result.to(f_hat.device)
                            leakage_val: float = float(
                                (1.0 - smoke_pct_final.mean()).item()
                            )
                        elif smoke_result.dim() >= 3:
                            # Solver returns smoke trajectory [B, T, H, W]
                            smoke_trajectory: torch.Tensor = smoke_result.to(f_hat.device)

                            if _target is not None and _target.dtype == torch.bool:
                                # target is a bucket mask [H, W]
                                bucket_mask = _target.to(f_hat.device)
                                # Smoke in bucket at final timestep
                                smoke_final = smoke_trajectory[:, -1]  # [B, H, W]
                                smoke_in_bucket = torch.sum(
                                    smoke_final * bucket_mask.float(), dim=(-2, -1)
                                )  # [B]
                                total_smoke = torch.sum(
                                    smoke_trajectory[:, 0], dim=(-2, -1)
                                ).clamp(min=1e-8)  # [B]
                                smoke_pct = smoke_in_bucket / total_smoke  # [B]
                                leakage_val = float((1.0 - smoke_pct.mean()).item())
                            else:
                                # No mask: use last channel as smoke percentage
                                leakage_val = float(
                                    (1.0 - smoke_trajectory[:, -1].mean()).item()
                                )
                        else:
                            leakage_val = 0.0
                    else:
                        leakage_val = 0.0

                leakage_const: torch.Tensor = torch.tensor(
                    leakage_val,
                    dtype=f_hat.dtype,
                    device=f_hat.device,
                    requires_grad=False,
                )
                total_objective: torch.Tensor = leakage_const + control_energy

            except Exception as exc:
                logger.warning(
                    "Solver call failed in objective_fn_2d: %s. "
                    "Using regularization term only for guidance.",
                    exc,
                )
                total_objective = control_energy

            return total_objective

        return objective_fn_2d

    # -----------------------------------------------------------------------
    # Final objective evaluation
    # -----------------------------------------------------------------------

    def evaluate_control_objective(
        self,
        f_pred: torch.Tensor,
        target: torch.Tensor,
        solver_fn: Callable[[torch.Tensor], torch.Tensor],
        alpha: float,
    ) -> float:
        """Evaluate the control objective using the ground-truth solver.

        This is the definitive evaluation used for reporting results in
        Table 2a and Table 2b. Always uses the ground-truth solver output
        u(T,x) from the generated f, never the model's predicted state.

        Paper Appendix F.1: "the state deviation in our reported evaluation
        metric I is always based on the output u(T,x) of the ground-truth
        solver given the control force f(t,x)."

        For 1D Burgers' (Eq. 6):
            I = integral_D |u(T,x) - u*(x)|^2 dx
              + alpha * integral_{[0,T]×D} |f(t,x)|^2 dt dx

        For 2D fluid:
            I = percentage of smoke NOT passing through target bucket

        Args:
            f_pred: Generated control sequence in physical space.
                Shape [B, T, X] = [B, 80, 120] for 1D Burgers'.
                Shape [B, T, N_ctrl] = [B, 32, 3584] for 2D fluid.
                dtype=float32.
            target: Target state tensor.
                For 1D: u* [B, X] or [X] (broadcast to batch).
                For 2D: target bucket mask [H, W] or smoke percentage.
            solver_fn: Ground-truth PDE solver callable. Same as
                self.solver_fn but passed explicitly for flexibility.
            alpha: Control energy regularization weight.
                Config: data.burgers.control_alpha=0.1.

        Returns:
            Scalar float I (mean over batch). Lower is better.
            Paper Table 2a: WDNO achieves 0.0205 on 1D Burgers'.
            Paper Table 2b: WDNO achieves 0.0679 on 2D fluid.
        """
        device = torch.device(self.device)
        f_pred = f_pred.to(device=device, dtype=torch.float32)
        target = target.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            if self.spatial_dim == 1:
                return self._evaluate_objective_1d(
                    f_pred=f_pred,
                    target=target,
                    solver_fn=solver_fn,
                    alpha=alpha,
                )
            else:
                return self._evaluate_objective_2d(
                    f_pred=f_pred,
                    target=target,
                    solver_fn=solver_fn,
                    alpha=alpha,
                )

    def _evaluate_objective_1d(
        self,
        f_pred: torch.Tensor,
        target: torch.Tensor,
        solver_fn: Callable[[torch.Tensor], torch.Tensor],
        alpha: float,
    ) -> float:
        """Evaluate 1D Burgers' control objective using ground-truth solver.

        Args:
            f_pred: Control sequence [B, T, X] = [B, 80, 120]. float32.
            target: Target state u* [B, X] or [X]. float32.
            solver_fn: Finite difference solver. Takes f [B, T, X] and
                returns u_T [B, X] (final state at t=T).
            alpha: Regularization weight (0.1).

        Returns:
            Mean control objective I over the batch. Scalar float.
        """
        B: int = f_pred.shape[0]

        # Interpolate f_pred to solver resolution if needed
        # Appendix F.1: solver runs at [80*16, 120*16] internally
        f_for_solver: torch.Tensor = self._interpolate_for_solver(
            f_pred=f_pred,
            target_shape=(self.nt_coarse, self.nx_coarse),
        )

        # Run ground-truth solver: f [B, T, X] → u_T [B, X]
        try:
            u_T: torch.Tensor = solver_fn(f_for_solver)
            u_T = u_T.to(f_pred.device, dtype=torch.float32)
        except Exception as exc:
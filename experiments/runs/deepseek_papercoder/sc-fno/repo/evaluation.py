# evaluation.py
# ============================================================================
# Purpose: Provide the Evaluator class that computes quantitative metrics
#          (relative L² error, R²) for the state u and for the parameter
#          sensitivities ∂u/∂p, and generates sample prediction plots
#          comparable to those shown in the SC‑FNO paper.  The module is
#          designed to work with a trained FNO model and a PDEDataset of
#          test samples.
# ============================================================================

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.autograd as autograd
import torch.utils.data
import matplotlib.pyplot as plt

from config import Config
from dataset import PDEDataset
from models.fno import FNO
from utils import relative_l2_error, r2_score


class Evaluator:
    """Evaluates a trained FNO‑based surrogate model on a test dataset.

    This class implements the exact evaluation protocol described in the paper:
    - metrics are reported for u(t) and for ∂u/∂p (per parameter and averaged).
    - all computations respect the predicted part of the solution (time steps
      from M to N).
    - sample plots are generated with the same visual style as Figures 3, 6, etc.

    Attributes:
        model:  The trained FNO (moved to `device` and set to eval mode).
        dataset: The test split of the PDEDataset.
        config: Global configuration object.
        device: torch device (cuda/cpu).
        grid:   Coordinate grid tensor (used as input to the FNO).
                Its shape depends on the equation:
                  ODE : (1, time, 1)
                  PDE1,2,4 : (1, S_x, N_time, 2)
                  PDE3 : (1, S_x, S_y, 2)
        output_slice: slice object that, when applied to a model output of the
                      same shape as the grid (excluding batch), restricts the
                      time dimension to [M, N_time) (i.e., the predicted horizon).
                      For PDE3 (no time), it is equivalent to :, i.e. noop.
    """

    def __init__(
        self,
        model: FNO,
        dataset: PDEDataset,
        config: Config,
    ) -> None:
        """Instantiate the evaluator.

        Args:
            model:   A trained FNO (or SC‑FNO variant).  The model will be put
                     into evaluation mode and moved to the device specified in
                     `config.global.device`.
            dataset: Test dataset (instance of PDEDataset).  It is expected to
                     provide samples of the form (p, u_input, u_true, J_true),
                     where u_true and J_true cover the full spatio‑temporal
                     domain (including the initial conditioning time steps).
            config:  The global configuration dataclass.
        """
        self.model = model.eval()
        self.dataset = dataset
        self.config = config

        # ------- Device --------------------------------------------------------
        self.device = torch.device(config.global_params["device"])
        self.model.to(self.device)

        # ------- Equation specifics --------------------------------------------
        eq_name = config.equation
        eq_cfg = config.sol_params  # per‑equation dict
        self.eq_name = eq_name
        self._eq_cfg = eq_cfg

        # ------- Build the coordinate grid -------------------------------------
        self.grid = self._build_grid()
        self.grid = self.grid.to(self.device)

        # ------- Determine output slicing --------------------------------------
        # The model outputs the solution for the entire temporal domain, but we
        # only evaluate (and supervised during training) the part from M to N.
        # We store a slice object that picks those time indices.
        M = eq_cfg["M"]
        N_tot = eq_cfg["N_time"]
        spatial_dims = eq_cfg.get("spatial_dims", 0)

        if spatial_dims == 0:   # ODE
            self.output_slice = (slice(None), slice(M, N_tot))   # (batch, time) -> slice time
        elif spatial_dims == 1: # 1D + time
            # stored grid is (1, S_x, N_time, 2) -> output shape (B, 1, S_x, N_time)
            # We need to slice along time axis (dim=3). Store tuple of slices.
            self.output_slice = (slice(None), slice(None), slice(None), slice(M, N_tot))
        elif spatial_dims == 2: # PDE3: output is only final spatial map, no time
            self.output_slice = (slice(None),)  # no slicing needed; just keep all
        else:
            raise ValueError(f"Unsupported spatial_dims={spatial_dims}")

        # ------- Evaluation settings from config -------------------------------
        self.save_plots = config.eval_params.get("save_plots", False)
        self.plot_format = config.eval_params.get("plot_format", "pdf")
        self._output_dir = Path(config.global_params["output_dir"])
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Public interface
    # =========================================================================

    @torch.no_grad()
    def compute_metrics(self) -> Dict[str, float]:
        """Compute all evaluation metrics on the entire test set.

        The method iterates over the dataset (batch size = 1 for safety with
        Jacobian computation), performs model inference with gradient tracking
        for the Jacobian, and computes:

        * state 'u': relative L² error and R² (over the predicted time interval)
        * sensitivity '∂u/∂p': per parameter relative L² and R², and their
          average (the paper’s “Mean Jacobian Metrics”).

        Returns:
            A dictionary with keys:
                'u_relative_l2', 'u_r2',
                'dp_<param>_relative_l2', 'dp_<param>_r2' for each parameter,
                'avg_dp_relative_l2', 'avg_dp_r2'.
        """
        model = self.model
        device = self.device
        dataset = self.dataset
        param_names = self._eq_cfg["param_names"]
        n_params = len(param_names)

        # Accumulators for state values
        u_pred = []
        u_true = []
        # Accumulators for Jacobians – one list per parameter
        jac_pred = {name: [] for name in param_names}
        jac_true = {name: [] for name in param_names}

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=0
        )

        for batch in loader:
            p, u_input, u_t, J_t = [b.to(device) for b in batch]
            # p: (1, P); u_input: (1, *input_shape); u_t: (1, *full_output_shape_if_unsliced);
            # J_t: (1, P, *full_output_shape)

            # ----- State prediction (no grad) ---------------------------------
            grid_batch = self._expand_grid(p.size(0))
            u_hat = model(u_input, p, grid_batch)   # (1, 1, *output_shape_unsliced)
            # Slice to predicted part
            u_hat_sliced = self._slice(u_hat)
            u_true_sliced = self._slice(u_t)
            u_pred.append(u_hat_sliced.cpu())
            u_true.append(u_true_sliced.cpu())

            # ----- Jacobian prediction (gradient computation) ------------------
            # Need p with grad; we re‑run forward with gradient tracking.
            p_grad = p.clone().detach().requires_grad_(True)
            u_hat_grad = model(u_input, p_grad, grid_batch)
            u_hat_grad_sliced = self._slice(u_hat_grad)   # (1, 1, *pred_shape)

            # Compute full Jacobian matrix ∂u/∂p at every output point.
            # We flatten the output and use autograd.functional.jacobian.
            # (batch_size=1, so we can directly use the single sample.)
            def _output_flat(pp: torch.Tensor) -> torch.Tensor:
                u_out = model(u_input, pp, grid_batch)
                return self._slice(u_out).reshape(-1)

            J_flat = autograd.functional.jacobian(_output_flat, p_grad)
            # J_flat shape: (total_output_points, P)
            # Reshape to (1, P, *pred_shape)
            pred_shape = u_hat_sliced.shape[2:]  # spatial + time (if any)
            J_hat = J_flat.T.reshape(1, n_params, *pred_shape)  # (1, P, *pred_shape)

            # True Jacobian sliced to predicted part
            J_true_sliced = self._slice(J_t)

            # Store per‑parameter flattened values
            for i, name in enumerate(param_names):
                jac_pred[name].append(J_hat[0, i].cpu().flatten())
                jac_true[name].append(J_true_sliced[0, i].cpu().flatten())

        # ----- Stack tensors and compute metrics -------------------------------
        u_pred = torch.cat(u_pred).reshape(-1)
        u_true = torch.cat(u_true).reshape(-1)

        metrics = {}
        metrics["u_relative_l2"] = relative_l2_error(u_pred, u_true).item()
        metrics["u_r2"] = r2_score(u_pred, u_true).item()

        # Per parameter
        param_metrics = {}
        for name in param_names:
            pred = torch.cat(jac_pred[name]).reshape(-1)
            true = torch.cat(jac_true[name]).reshape(-1)
            if true.numel() == 0:
                # No data for this parameter (should not happen)
                rel_l2 = float("nan")
                r2 = float("nan")
            else:
                rel_l2 = relative_l2_error(pred, true).item()
                r2 = r2_score(pred, true).item()
            metrics[f"dp_{name}_relative_l2"] = rel_l2
            metrics[f"dp_{name}_r2"] = r2
            param_metrics[name] = (rel_l2, r2)

        # Average Jacobian metrics
        if param_metrics:
            avg_rel_l2 = np.mean([v[0] for v in param_metrics.values()])
            avg_r2 = np.mean([v[1] for v in param_metrics.values()])
            metrics["avg_dp_relative_l2"] = avg_rel_l2
            metrics["avg_dp_r2"] = avg_r2
        else:
            metrics["avg_dp_relative_l2"] = float("nan")
            metrics["avg_dp_r2"] = float("nan")

        return metrics

    def plot_sample(
        self,
        sample_idx: int = 0,
        save_path: Optional[str] = None,
    ) -> None:
        """Generate publication‑style plots for a single test sample.

        The figure layout depends on the equation type:
          - ODE: line plots of u(t) and ∂u/∂p vs time.
          - 1D+time: false‑colour image of u(x,t) and line profiles at
            selected x positions; optionally sensitivity images.
          - 2D spatial: colormesh of ω(x,y) and its sensitivities.

        Args:
            sample_idx: Index of the sample in the test dataset.
            save_path:  If ``None``, the figure is shown on screen.  Otherwise
                        it is saved to the path (overridden by config if
                        save_plots is True).
        """
        model = self.model
        dataset = self.dataset
        device = self.device

        # 1. Retrieve sample
        p, u_input, u_t, J_t = dataset[sample_idx]
        p = p.unsqueeze(0).to(device)
        u_input = u_input.unsqueeze(0).to(device)
        u_t_sliced = self._slice(u_t.unsqueeze(0)).squeeze(0).cpu().numpy()
        J_t_sliced = self._slice(J_t.unsqueeze(0)).squeeze(0).cpu().numpy()  # (P, *pred_shape)

        # 2. Model prediction (state)
        grid_batch = self._expand_grid(1)
        with torch.no_grad():
            u_hat = model(u_input, p, grid_batch)
        u_hat_sliced = self._slice(u_hat).squeeze(0).cpu().numpy()

        # 3. Jacobian prediction via AD
        p_grad = p.clone().detach().requires_grad_(True)
        u_hat_grad = model(u_input, p_grad, grid_batch)
        u_hat_grad_sliced = self._slice(u_hat_grad)   # (1, 1, *pred_shape)

        def _out_flat(pp):
            u_out = model(u_input, pp, grid_batch)
            return self._slice(u_out).reshape(-1)

        J_flat = autograd.functional.jacobian(_out_flat, p_grad)
        pred_shape = u_hat_sliced.shape[2:]
        n_params = p.size(1)
        J_hat = J_flat.T.reshape(1, n_params, *pred_shape).squeeze(0).cpu().numpy()  # (P, *pred_shape)

        # 4. Dispatch to equation‑specific plotter
        eq_name = self.eq_name
        spatial_dims = self._eq_cfg.get("spatial_dims", 0)
        if spatial_dims == 0:
            self._plot_ode(
                u_hat_sliced, u_t_sliced,
                J_hat, J_t_sliced,
                self._eq_cfg,
                save_path,
            )
        elif spatial_dims == 1:
            self._plot_pde1d(
                u_hat_sliced, u_t_sliced,
                J_hat, J_t_sliced,
                self._eq_cfg,
                save_path,
            )
        elif spatial_dims == 2:
            self._plot_pde2d(
                u_hat_sliced, u_t_sliced,
                J_hat, J_t_sliced,
                self._eq_cfg,
                save_path,
            )

    # =========================================================================
    # Private helpers – grid, slicing, Jacobian
    # =========================================================================

    def _build_grid(self) -> torch.Tensor:
        """Create a fixed coordinate grid for the current equation.

        Returns:
            Tensor of shape (1, *spatial_dims, n_coords) float32.
        """
        eq_cfg = self._eq_cfg
        spatial_dims = eq_cfg.get("spatial_dims", 0)
        N_tot = eq_cfg["N_time"]
        M = eq_cfg["M"]

        if spatial_dims == 0:   # ODE
            t = torch.linspace(
                eq_cfg["temporal_domain"][0],
                eq_cfg["temporal_domain"][1],
                N_tot,
                dtype=torch.float32,
            )
            # The FNO expects grid to have channel 1 for time.
            # We'll create grid of shape (1, N_tot, 1) so that after cat
            # with input and param embedding, the channels include grid.
            grid = t.view(1, N_tot, 1)  # (1, time, coord)
        elif spatial_dims == 1: # 1D space + time
            x = torch.linspace(
                eq_cfg["spatial_domain"][0],
                eq_cfg["spatial_domain"][1],
                eq_cfg["S_x"],
                dtype=torch.float32,
            )
            t = torch.linspace(
                eq_cfg["temporal_domain"][0],
                eq_cfg["temporal_domain"][1],
                N_tot,
                dtype=torch.float32,
            )
            X, T = torch.meshgrid(x, t, indexing="ij")   # (S_x, N_tot)
            grid = torch.stack([X, T], dim=-1)           # (S_x, N_tot, 2)
            grid = grid.unsqueeze(0)                     # (1, S_x, N_tot, 2)
        elif spatial_dims == 2: # 2D spatial (PDE3)
            dom = eq_cfg["spatial_domain"]  # [x0, x1, y0, y1]
            x0, x1, y0, y1 = dom[0], dom[1], dom[2], dom[3]
            x = torch.linspace(x0, x1, eq_cfg["S_x"], dtype=torch.float32)
            y = torch.linspace(y0, y1, eq_cfg["S_y"], dtype=torch.float32)
            X, Y = torch.meshgrid(x, y, indexing="ij")   # (S_x, S_y)
            grid = torch.stack([X, Y], dim=-1)           # (S_x, S_y, 2)
            grid = grid.unsqueeze(0)                     # (1, S_x, S_y, 2)
        else:
            raise ValueError(f"Unsupported spatial_dims {spatial_dims}")

        return grid

    def _expand_grid(self, batch_size: int) -> torch.Tensor:
        """Repeat the stored grid for a batch of size `batch_size`."""
        return self.grid.expand(batch_size, *self.grid.shape[1:])

    def _slice(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply the stored output_slice to the tensor.

        The tensor is expected to have a batch dimension (first dim) and
        may have additional spatial/time dimensions.  The slicing always
        includes the batch dimension (slice(None)).
        """
        return tensor[self.output_slice]

    # =========================================================================
    # Plotting routines
    # =========================================================================

    def _plot_ode(
        self,
        u_pred: np.ndarray,  # (time,)  (predicted interval)
        u_true: np.ndarray,
        J_pred: np.ndarray,  # (P, time)
        J_true: np.ndarray,
        eq_cfg: dict,
        save_path: Optional[str],
    ) -> None:
        """Line plots for ODEs: u(t) and each ∂u/∂p vs t."""
        P = J_pred.shape[0]
        param_names = eq_cfg["param_names"]
        t_pred = np.linspace(
            eq_cfg["temporal_domain"][0] + (eq_cfg["temporal_domain"][1] - eq_cfg["temporal_domain"][0])
            / eq_cfg["N_time"] * eq_cfg["M"],
            eq_cfg["temporal_domain"][1],
            u_pred.shape[-1],
        )

        fig, axes = plt.subplots(1 + P, 1, figsize=(8, 2 * (1 + P)))
        # State plot
        axes[0].plot(t_pred, u_true, "k-", label="True u")
        axes[0].plot(t_pred, u_pred, "r--", label="Pred u")
        axes[0].legend()
        axes[0].set_ylabel("u(t)")

        for i in range(P):
            ax = axes[i + 1]
            ax.plot(t_pred, J_true[i], "k-")
            ax.plot(t_pred, J_pred[i], "r--")
            ax.set_ylabel(f"∂u/∂{param_names[i]}")

        axes[-1].set_xlabel("t")

        self._finalize_plot(save_path or "ode_sample")

    def _plot_pde1d(
        self,
        u_pred: np.ndarray,  # (S_x, time_pred)
        u_true: np.ndarray,
        J_pred: np.ndarray,  # (P, S_x, time_pred)
        J_true: np.ndarray,
        eq_cfg: dict,
        save_path: Optional[str],
    ) -> None:
        """Plots for 1D+time equations: false‑colour and line profiles."""
        S_x = u_pred.shape[0]
        N_pred = u_pred.shape[1]
        x = np.linspace(eq_cfg["spatial_domain"][0], eq_cfg["spatial_domain"][1], S_x)
        t_pred = np.linspace(
            eq_cfg["temporal_domain"][0] + (eq_cfg["temporal_domain"][1] - eq_cfg["temporal_domain"][0])
            / eq_cfg["N_time"] * eq_cfg["M"],
            eq_cfg["temporal_domain"][1],
            N_pred,
        )

        n_params = J_pred.shape[0]
        param_names = eq_cfg["param_names"]

        # Create figure with 2 + n_params rows
        fig, axes = plt.subplots(2 + n_params, 2, figsize=(10, 3 * (2 + n_params)))

        # --- Row 0: false colour u true vs pred ---
        im0 = axes[0, 0].pcolormesh(t_pred, x, u_true, shading="auto")
        axes[0, 0].set_title("True u(x,t)")
        plt.colorbar(im0, ax=axes[0, 0])
        im1 = axes[0, 1].pcolormesh(t_pred, x, u_pred, shading="auto")
        axes[0, 1].set_title("Predicted u(x,t)")
        plt.colorbar(im1, ax=axes[0, 1])

        # --- Row 1: line profiles at selected x ---
        x_sel = [0.0, 0.5, 1.0]
        ax = axes[1, 0]
        for xv in x_sel:
            idx = np.abs(x - xv).argmin()
            ax.plot(t_pred, u_true[idx, :], label=f"x={xv:.2f} true")
        ax.set_title("Selected spatial profiles (true)")
        ax.legend()
        ax = axes[1, 1]
        for xv in x_sel:
            idx = np.abs(x - xv).argmin()
            ax.plot(t_pred, u_pred[idx, :], label=f"x={xv:.2f} pred")
        ax.set_title("Selected spatial profiles (pred)")
        ax.legend()

        # --- Remaining rows: sensitivity for each parameter ---
        for i in range(n_params):
            ax_true = axes[2 + i, 0]
            ax_pred = axes[2 + i, 1]
            sen_true = J_true[i]
            sen_pred = J_pred[i]
            # Option 1: false colour (since it's 2D)
            imt = ax_true.pcolormesh(t_pred, x, sen_true, shading="auto")
            ax_true.set_title(f"True ∂u/∂{param_names[i]}")
            plt.colorbar(imt, ax=ax_true)
            imp = ax_pred.pcolormesh(t_pred, x, sen_pred, shading="auto")
            ax_pred.set_title(f"Pred ∂u/∂{param_names[i]}")
            plt.colorbar(imp, ax=ax_pred)

        fig.tight_layout()
        self._finalize_plot(save_path or "pde1d_sample")

    def _plot_pde2d(
        self,
        u_pred: np.ndarray,  # (S_x, S_y)   (no time axis – final snapshot)
        u_true: np.ndarray,
        J_pred: np.ndarray,  # (P, S_x, S_y)
        J_true: np.ndarray,
        eq_cfg: dict,
        save_path: Optional[str],
    ) -> None:
        """Colormesh plots for 2D spatial problem (PDE3)."""
        S_x, S_y = u_pred.shape
        dom = eq_cfg["spatial_domain"]  # [x0, x1, y0, y1]
        x = np.linspace(dom[0], dom[1], S_x)
        y = np.linspace(dom[2], dom[3], S_y)

        n_params = J_pred.shape[0]
        param_names = eq_cfg["param_names"]

        fig, axes = plt.subplots(1 + n_params, 2, figsize=(8, 3 * (1 + n_params)))

        # State
        ax0_true = axes[0, 0]
        ax0_pred = axes[0, 1]
        c0 = ax0_true.pcolormesh(x, y, u_true.T, shading="auto")  # note transpose for pcolormesh
        ax0_true.set_title("True ω (final time)")
        plt.colorbar(c0, ax=ax0_true)
        c1 = ax0_pred.pcolormesh(x, y, u_pred.T, shading="auto")
        ax0_pred.set_title("Pred ω (final time)")
        plt.colorbar(c1, ax=ax0_pred)

        # Sensitivities
        for i in range(n_params):
            ax_true = axes[1 + i, 0]
            ax_pred = axes[1 + i, 1]
            sen_true = J_true[i]
            sen_pred = J_pred[i]
            ct = ax_true.pcolormesh(x, y, sen_true.T, shading="auto")
            ax_true.set_title(f"True ∂ω/∂{param_names[i]}")
            plt.colorbar(ct, ax=ax_true)
            cp = ax_pred.pcolormesh(x, y, sen_pred.T, shading="auto")
            ax_pred.set_title(f"Pred ∂ω/∂{param_names[i]}")
            plt.colorbar(cp, ax=ax_pred)

        fig.tight_layout()
        self._finalize_plot(save_path or "pde2d_sample")

    def _finalize_plot(self, save_path: Optional[str]) -> None:
        """Save or show the current matplotlib figure according to config."""
        if self.save_plots:
            final_path = Path(save_path or "plot").with_suffix(f".{self.plot_format}")
            plt.savefig(final_path, bbox_inches="tight")
            print(f"Saved figure to {final_path}")
        else:
            plt.show()
        plt.close()

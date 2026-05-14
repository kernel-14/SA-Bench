# inversion.py
# ============================================================================
# Purpose: Implement gradient‑based parameter inversion using a trained SC‑FNO
#          surrogate model.  This class reproduces the single‑ and multi‑
#          parameter inversion experiments described in Section 3.1 of the
#          SC‑FNO paper.
# ============================================================================

from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from dataset import PDEDataset
from models.fno import FNO
from solver import Solver
from utils import relative_l2_error, r2_score


class Inversion:
    """Performs gradient‑based parameter inversion with a trained FNO surrogate.

    The class uses the trained model in evaluation mode and optimises the
    physical parameters `p` (or a subset) to minimise the mean‑squared error
    between the model’s solution and an observed solution `u_obs`.  Both
    single‑parameter and multi‑parameter inversion are supported, following
    the experimental setup in the paper.

    The optimisation is performed with the Adam optimiser, and the estimated
    parameters are clamped to the valid training ranges after each step to
    stay within the surrogate’s region of reliability.

    Attributes:
        model:              The trained FNO (frozen, in eval mode).
        solver:             The differentiable solver (used only if `u_obs` is
                            not provided; typically, `u_obs` is supplied from
                            a pre‑existing dataset).
        config:             Global configuration object.
        device:             Torch device (cuda/cpu).
        grid:               Coordinate grid tensor used as input to the model.
                            Its shape matches `model.forward` requirements.
        param_names:        List of parameter names (e.g., ['alpha','beta']).
        lower_bounds, upper_bounds:  Tensors of valid parameter ranges.
        lr, steps, single_param_name, multi_param, num_experiments:  Settings
                            from the `inversion` section of the config.
    """

    def __init__(self, model: FNO, solver: Solver, config: Config) -> None:
        """Initialise the Inversion module.

        Args:
            model:  A trained `FNO` instance (or any SC‑FNO variant).  It must
                    be in the same state as after training (i.e., on the correct
                    device and in eval mode – this constructor will enforce both).
            solver: A `Solver` instance for the selected equation.  Required for
                    the interface, but not used if `u_obs` is always given.
            config: The global configuration object (frozen dataclass).
        """
        # ---------- Model and device --------------------------------------------
        self.model = model.eval()
        self.solver = solver
        self.config = config
        self.device = torch.device(config.global_params["device"])
        self.model.to(self.device)

        # ---------- Build coordinate grid ---------------------------------------
        self.grid = self._build_grid().to(self.device)

        # ---------- Inversion settings from config --------------------------------
        inv_cfg = config.inversion_params
        self.lr: float = inv_cfg.get("lr", 0.01)
        self.steps: int = inv_cfg.get("steps", 500)
        self.single_param_name: str = inv_cfg.get("single_param", "alpha")
        self.multi_param: bool = inv_cfg.get("multi_param", True)
        self.num_experiments: int = inv_cfg.get("num_experiments", 20)

        # ---------- Parameter metadata ------------------------------------------
        eq_params = config.sol_params
        eq = config.equation

        # Handle naming and bounds for all equation types, including the
        # zoned Burgers' case where parameters are spatially varying scalars.
        if eq == "pde2_zoned":
            num_zones = eq_params["num_zones"]
            alpha_rng = eq_params["param_ranges"]["alpha_zonal"]
            delta_rng = eq_params["param_ranges"]["delta_zonal"]
            gamma_rng = eq_params["param_ranges"]["gamma"]
            omega_rng = eq_params["param_ranges"]["omega"]

            # Build explicit parameter names (e.g., alpha_0, ..., delta_39, gamma, omega)
            alpha_names = [f"alpha_{i}" for i in range(num_zones)]
            delta_names = [f"delta_{i}" for i in range(num_zones)]
            self.param_names: List[str] = alpha_names + delta_names + ["gamma", "omega"]
            self.num_params: int = len(self.param_names)

            # Build lower / upper bound tensors
            lower = [alpha_rng[0]] * num_zones + [delta_rng[0]] * num_zones + [gamma_rng[0], omega_rng[0]]
            upper = [alpha_rng[1]] * num_zones + [delta_rng[1]] * num_zones + [gamma_rng[1], omega_rng[1]]
            self.lower_bounds = torch.tensor(lower, dtype=torch.float32)
            self.upper_bounds = torch.tensor(upper, dtype=torch.float32)
        else:
            # Standard scalar‑parameter equations
            self.param_names: List[str] = eq_params["param_names"]
            self.num_params: int = len(self.param_names)

            ranges = eq_params["param_ranges"]
            lower_list = [ranges[name][0] for name in self.param_names]
            upper_list = [ranges[name][1] for name in self.param_names]
            self.lower_bounds = torch.tensor(lower_list, dtype=torch.float32)
            self.upper_bounds = torch.tensor(upper_list, dtype=torch.float32)

    # --------------------------------------------------------------------------
    # Private helpers – grid construction & random initialisation
    # --------------------------------------------------------------------------

    def _build_grid(self) -> torch.Tensor:
        """Create the coordinate grid expected by the FNO's forward method.

        The exact shape depends on the equation type:
          - ODEs: (1, 1, N_time)
          - 1D+time PDEs: (1, 2, S_x, N_time)
          - 2D spatial PDE (Navier‑Stokes): (1, 2, S_x, S_y)
        """
        eq = self.config.equation
        eq_params = self.config.sol_params
        spatial_dims = eq_params.get("spatial_dims", 0)

        if eq in ("ode1", "ode2"):
            t_start, t_end = eq_params["temporal_domain"]
            Nt = eq_params["N_time"]
            t = torch.linspace(t_start, t_end, Nt, dtype=torch.float32)
            return t.view(1, 1, Nt)  # (1, 1, Nt)

        if eq == "pde3":
            dom = eq_params["spatial_domain"]  # [x0, x1, y0, y1]
            x0, x1, y0, y1 = dom[0], dom[1], dom[2], dom[3]
            Sx, Sy = eq_params["S_x"], eq_params["S_y"]
            x = torch.linspace(x0, x1, Sx, dtype=torch.float32)
            y = torch.linspace(y0, y1, Sy, dtype=torch.float32)
            X, Y = torch.meshgrid(x, y, indexing="ij")
            return torch.stack([X, Y], dim=0).unsqueeze(0)  # (1, 2, Sx, Sy)

        # 1D space + time (PDE1, PDE2, PDE4, pde2_zoned)
        x_start, x_end = eq_params["spatial_domain"]
        Sx = eq_params["S_x"]
        x = torch.linspace(x_start, x_end, Sx, dtype=torch.float32)
        t_start, t_end = eq_params["temporal_domain"]
        Nt = eq_params["N_time"]
        t = torch.linspace(t_start, t_end, Nt, dtype=torch.float32)
        X, T = torch.meshgrid(x, t, indexing="ij")
        return torch.stack([X, T], dim=0).unsqueeze(0)  # (1, 2, Sx, Nt)

    def _generate_param_init(
        self, true_p: torch.Tensor, opt_indices: List[int]
    ) -> torch.Tensor:
        """Create an initial parameter guess for inversion.

        The true parameter `true_p` is cloned.  For every index in
        `opt_indices`, the value is replaced by a uniform random sample
        drawn from the corresponding allowed range.  All other indices
        retain their true values (they will be fixed during optimisation).

        Args:
            true_p:      True parameter tensor of shape (num_params,) on CPU.
            opt_indices: List of indices to randomise.

        Returns:
            A tensor of shape (num_params,) with gradients enabled, placed on
            the computation device.
        """
        p_init = true_p.clone().detach().to(self.device)
        for idx in opt_indices:
            lo = self.lower_bounds[idx].item()
            hi = self.upper_bounds[idx].item()
            p_init[idx] = np.random.uniform(lo, hi)
        p_init.requires_grad_(True)
        return p_init

    # --------------------------------------------------------------------------
    # Core inversion routine
    # --------------------------------------------------------------------------

    def run_inversion(
        self,
        p_true: torch.Tensor,
        u_input: torch.Tensor,
        u_obs: Optional[torch.Tensor] = None,
        opt_indices: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Run gradient‑based inversion for a single test sample.

        Args:
            p_true:     True physical parameters, shape (num_params,).
            u_input:    Input initial‑condition segment (the part the model
                        expects as input during forward).  Shape must match
                        the model's input spatial dimensions, e.g.,
                        (S_x, N_time) for PDE1 (with a channel dim maybe).
                        The tensor will be moved to the device and given a
                        batch dimension if necessary.
            u_obs:      Observed solution to match.  If ``None``, the solver
                        is used to generate it (not implemented here).  Shape
                        must align with the model output.
            opt_indices: List of parameter indices to optimise.  If ``None``,
                         the method determines them from the config:
                         - if ``multi_param`` is true, all indices.
                         - otherwise, the index of ``single_param_name``.

        Returns:
            A dictionary containing:
                - ``"estimated_p"``: final optimised parameter tensor (CPU).
                - ``"true_p"``: input true parameter tensor (CPU).
                - ``"metrics"``: per‑parameter and average relative L² and R².
                - ``"loss_history"``: list of loss values during optimisation.
        """
        # ---------- Determine parameter indices to optimise ----------------------
        if opt_indices is None:
            if self.multi_param:
                opt_indices = list(range(self.num_params))
            else:
                if self.single_param_name not in self.param_names:
                    raise ValueError(
                        f"Parameter '{self.single_param_name}' not found in "
                        f"{self.param_names}."
                    )
                opt_idx = self.param_names.index(self.single_param_name)
                opt_indices = [opt_idx]

        # ---------- Move data to device and ensure correct shape ----------------
        p_true_dev = p_true.to(self.device)
        u_input_dev = u_input.to(self.device)
        if u_obs is None:
            # In practice, u_obs should always be supplied from the dataset.
            raise ValueError(
                "u_obs must be provided.  Generation via solver is not implemented "
                "in this version."
            )
        u_obs_dev = u_obs.to(self.device)

        # Add batch dimension if missing (model expects (B, C, *spatial))
        if u_input_dev.dim() == self.grid.dim() - 1:  # missing batch dim
            u_input_dev = u_input_dev.unsqueeze(0)
        if u_obs_dev.dim() == self.grid.dim() - 1:
            u_obs_dev = u_obs_dev.unsqueeze(0)

        # Add channel dimension if missing (model expects (B, 1, ...))
        if u_input_dev.shape[1] != 1:
            u_input_dev = u_input_dev.unsqueeze(1)
        if u_obs_dev.shape[1] != 1:
            u_obs_dev = u_obs_dev.unsqueeze(1)

        # ---------- Initial parameter guess -------------------------------------
        p_est = self._generate_param_init(p_true_dev, opt_indices)

        # ---------- Optimiser setup ---------------------------------------------
        optimizer = torch.optim.Adam([p_est], lr=self.lr)
        loss_history: List[float] = []

        lower = self.lower_bounds.to(self.device)
        upper = self.upper_bounds.to(self.device)

        # ---------- Inversion loop ----------------------------------------------
        for _ in range(self.steps):
            optimizer.zero_grad()

            # Expand the stored grid to the current batch size (1)
            grid_batch = self.grid.expand(1, *self.grid.shape[1:])

            pred = self.model(u_input_dev, p_est.unsqueeze(0), grid_batch)
            loss = F.mse_loss(pred, u_obs_dev)
            loss.backward()

            # Freeze gradients for parameters that are not being optimised
            if p_est.grad is not None:
                grad = p_est.grad
                for i in range(self.num_params):
                    if i not in opt_indices:
                        grad[i] = 0.0

            optimizer.step()

            # Clamp parameters to the valid training ranges
            with torch.no_grad():
                p_est.clamp_(lower, upper)

            loss_history.append(loss.item())

        # ---------- Compute evaluation metrics ----------------------------------
        metrics = self._compute_metrics(p_est.detach(), p_true_dev)

        return {
            "estimated_p": p_est.detach().cpu(),
            "true_p": p_true_dev.cpu(),
            "metrics": metrics,
            "loss_history": loss_history,
        }

    # --------------------------------------------------------------------------
    # Metric calculation
    # --------------------------------------------------------------------------

    def _compute_metrics(
        self, p_est: torch.Tensor, p_true: torch.Tensor
    ) -> Dict[str, Any]:
        """Compute per‑parameter and average relative L² and R² metrics.

        Args:
            p_est:  Estimated parameter vector (detached).
            p_true: True parameter vector.

        Returns:
            Dictionary with keys being parameter names (or 'average') and values
            being sub‑dictionaries of `"relative_l2"` and `"r2"`.
        """
        per_param = {}
        for i, name in enumerate(self.param_names):
            pe = p_est[i].reshape(-1)
            pt = p_true[i].reshape(-1)
            l2_val = relative_l2_error(pe, pt).item()
            r2_val = r2_score(pe, pt).item()
            per_param[name] = {"relative_l2": l2_val, "r2": r2_val}

        avg_l2 = np.mean([v["relative_l2"] for v in per_param.values()])
        avg_r2 = np.mean([v["r2"] for v in per_param.values()])
        per_param["average"] = {"relative_l2": avg_l2, "r2": avg_r2}
        return per_param

    # --------------------------------------------------------------------------
    # Multi‑experiment evaluation (reproduces Figure 2, Table D.11, …)
    # --------------------------------------------------------------------------

    def evaluate_inversion(
        self,
        dataset: PDEDataset,
        num_experiments: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run inversion on a set of random test samples and aggregate results.

        For each sample, the method calls `run_inversion` with the sample’s
        `u_true` as the observation.  After all experiments, per‑parameter
        means and standard deviations are computed and, if configured, scatter
        plots are saved.

        Args:
            dataset:          A PDEDataset instance containing test samples
                              (the split should already be 'test').
            num_experiments:  Number of random samples to use.  Defaults to the
                              value in `config.inversion.num_experiments`.

        Returns:
            A dictionary with keys:
              - `"parameter_metrics"`: per‑parameter aggregate stats.
              - `"overall_metrics"`: mean of all per‑sample average metrics.
              - `"num_experiments"`: actual number of experiments run.
        """
        n_exp = num_experiments if num_experiments is not None else self.num_experiments
        total_samples = len(dataset)
        if n_exp > total_samples:
            n_exp = total_samples
            print(f"Warning: `num_experiments` reduced to {n_exp} (dataset size).")

        indices = random.sample(range(total_samples), n_exp)

        results = []
        for idx in indices:
            p_true, u_input, u_true, _ = dataset[idx]
            res = self.run_inversion(p_true, u_input, u_obs=u_true)
            results.append(res)

        # ---------- Aggregate statistics ----------------------------------------
        param_metrics: Dict[str, Dict[str, float]] = {}
        for name in self.param_names:
            l2_list = [r["metrics"][name]["relative_l2"] for r in results]
            r2_list = [r["metrics"][name]["r2"] for r in results]
            param_metrics[name] = {
                "mean_relative_l2": float(np.mean(l2_list)),
                "std_relative_l2": float(np.std(l2_list)),
                "mean_r2": float(np.mean(r2_list)),
                "std_r2": float(np.std(r2_list)),
            }

        avg_l2_list = [r["metrics"]["average"]["relative_l2"] for r in results]
        avg_r2_list = [r["metrics"]["average"]["r2"] for r in results]
        overall = {
            "mean_rel_l2": float(np.mean(avg_l2_list)),
            "std_rel_l2": float(np.std(avg_l2_list)),
            "mean_r2": float(np.mean(avg_r2_list)),
            "std_r2": float(np.std(avg_r2_list)),
        }

        # ---------- Optional scatter plots (Figure 1 style) --------------------
        if self.config.eval_params.get("save_plots", False):
            out_dir = self.config.global_params["output_dir"]
            os.makedirs(out_dir, exist_ok=True)

            # Per‑parameter plots
            for name in self.param_names:
                idx_param = self.param_names.index(name)
                trues = [r["true_p"][idx_param].item() for r in results]
                estim = [r["estimated_p"][idx_param].item() for r in results]

                import matplotlib.pyplot as plt

                plt.figure()
                plt.scatter(trues, estim, alpha=0.6)
                all_vals = trues + estim
                min_v, max_v = min(all_vals), max(all_vals)
                plt.plot([min_v, max_v], [min_v, max_v], "k--")
                plt.xlabel(f"True {name}")
                plt.ylabel(f"Estimated {name}")
                plt.title(f"Inversion of {name}")
                save_path = os.path.join(out_dir, f"inversion_{name}.pdf")
                plt.savefig(save_path, bbox_inches="tight")
                plt.close()

        return {
            "parameter_metrics": param_metrics,
            "overall_metrics": overall,
            "num_experiments": n_exp,
        }

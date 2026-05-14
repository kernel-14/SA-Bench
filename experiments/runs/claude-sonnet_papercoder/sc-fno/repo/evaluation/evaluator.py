```python
## evaluation/evaluator.py
"""Evaluator class orchestrating all SC-FNO post-training experiments.

Reproduces the key tables and figures from the SC-FNO paper:
  - Tables 1, 2, 3, D.14: Forward simulation quality (u and ∂u/∂p)
  - Table 1 right half, Figure 5: Perturbation robustness
  - Figures 1, 2, Tables D.11: Parameter inversion
  - Figure 4: Data scaling experiment
  - Table 4: High-dimensional zoned PDE2

All evaluation methods return structured dicts with consistent
{'r2': float, 'relative_l2': float} leaf nodes, enabling programmatic
table construction in main.py.

References:
    - SC-FNO paper Section 3.1: Parameter inversion
    - SC-FNO paper Section 3.2: Surrogate quality and perturbation robustness
    - SC-FNO paper Section 3.3: Data scaling
    - config.yaml: perturbation.lambda_values, data_scaling.sample_sizes
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from data.dataset import SCFNODataset
from evaluation.metrics import Metrics
from training.inversion import Inverter
from utils.visualization import Visualizer


class Evaluator:
    """Orchestrates all evaluation experiments for SC-FNO model variants.

    Provides five public evaluation methods corresponding to the five main
    experimental sections of the SC-FNO paper. Each method returns a
    structured dict of metrics and optionally produces visualization figures.

    The evaluator is instantiated per model variant. Comparison across
    variants (e.g., FNO vs SC-FNO) is assembled by main.py by calling the
    same evaluation methods on multiple Evaluator instances.

    Attributes:
        model: The trained FNO/SC-FNO model. Set to eval mode at construction.
        test_loader: DataLoader for the standard held-out test set.
        cfg: The full master configuration dictionary from ConfigLoader.
        metrics: Metrics instance for R² and relative L² computation.
        visualizer: Visualizer instance for producing paper figures.
        device: torch.device for tensor operations.
        equation: Equation identifier string (e.g., 'pde1', 'ode1').
        n_params: Number of physical parameters for the active equation.
        param_names: Ordered list of parameter name strings.
        figures_dir: Directory where figures are saved.

    Example:
        >>> from models.sc_fno import build_model
        >>> from data.dataset import SCFNODataset
        >>> from torch.utils.data import DataLoader
        >>> from utils.config_loader import ConfigLoader
        >>>
        >>> cfg_loader = ConfigLoader('config.yaml')
        >>> cfg = cfg_loader.cfg
        >>> model = build_model(eq_cfg)
        >>> test_ds = SCFNODataset('data/datasets/pde1.pt', 'test', use_jacobian=True)
        >>> test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)
        >>> evaluator = Evaluator(model, test_loader, cfg)
        >>> forward_metrics = evaluator.evaluate_forward()
        >>> sensitivity_metrics = evaluator.evaluate_sensitivity()
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        cfg: dict,
    ) -> None:
        """Initializes the Evaluator.

        Args:
            model: The trained FNO or SC-FNO model instance. Must have a
                   .variant attribute (set by build_model in sc_fno.py) and
                   a .equation attribute. Will be set to eval mode in-place.
            test_loader: DataLoader for the standard held-out test set.
                         Batch dicts must contain 'params', 'u0', 'u_true',
                         'coords', and optionally 'jacobians'.
            cfg: The full master configuration dictionary loaded from
                 config.yaml. Must contain top-level keys:
                   - 'device': str
                   - 'figures_dir': str
                   - 'equation': str (active equation identifier)
                   - 'seed': int
                 Also reads equation-specific sub-configs for n_params and
                 param_names.
        """
        self.model: nn.Module = model
        self.test_loader: DataLoader = test_loader
        self.cfg: dict = cfg

        # ------------------------------------------------------------------
        # Device setup
        # ------------------------------------------------------------------
        device_str: str = str(
            cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.device: torch.device = torch.device(device_str)

        # Move model to device and set to eval mode.
        self.model = self.model.to(self.device)
        self.model.eval()

        # ------------------------------------------------------------------
        # Equation metadata
        # ------------------------------------------------------------------
        self.equation: str = str(cfg.get("equation", "pde1")).lower()

        # Extract n_params and param_names from the equation sub-config.
        eq_cfg: dict = cfg.get(self.equation, {})
        self.n_params: int = int(eq_cfg.get("n_params", cfg.get("n_params", 5)))

        params_cfg: dict = eq_cfg.get("params", cfg.get("params", {}))
        self.param_names: List[str] = list(params_cfg.keys())

        # Guard: if param_names is empty or mismatched, generate generic names.
        if len(self.param_names) != self.n_params:
            self.param_names = [f"p{i}" for i in range(self.n_params)]

        # ------------------------------------------------------------------
        # Metrics and visualization
        # ------------------------------------------------------------------
        self.figures_dir: str = str(cfg.get("figures_dir", "outputs/figures"))
        self.metrics: Metrics = Metrics()
        self.visualizer: Visualizer = Visualizer(save_dir=self.figures_dir)

        # ------------------------------------------------------------------
        # Random seed for reproducibility
        # ------------------------------------------------------------------
        self.seed: int = int(cfg.get("seed", 42))

        # ------------------------------------------------------------------
        # Print initialization summary
        # ------------------------------------------------------------------
        print(
            f"[Evaluator] Initialized for equation='{self.equation}' | "
            f"variant='{getattr(model, 'variant', 'fno')}' | "
            f"device={self.device} | "
            f"n_params={self.n_params} | "
            f"param_names={self.param_names}"
        )

    # ------------------------------------------------------------------
    # Public evaluation methods
    # ------------------------------------------------------------------

    def evaluate_forward(self) -> dict:
        """Evaluates solution accuracy u(t) and sensitivity ∂u/∂p on the test set.

        Reproduces Tables 1, 2, 3, D.14 from the paper. Computes R² and
        relative L² for both the solution field u and each Jacobian ∂u/∂pᵢ
        on the standard held-out test set (same parameter distribution as
        training).

        Returns:
            Nested dict with structure:
            {
                'u': {'r2': float, 'relative_l2': float},
                'du_d{param_name_0}': {'r2': float, 'relative_l2': float},
                'du_d{param_name_1}': {'r2': float, 'relative_l2': float},
                ...
            }
            One entry per parameter, keyed by 'du_d{param_name}'.
            Returns empty dict if test_loader is empty.

        Example:
            >>> metrics = evaluator.evaluate_forward()
            >>> metrics['u']['r2']           # e.g., 0.983 for PDE1 SC-FNO
            >>> metrics['du_dalpha']['r2']   # e.g., 0.925 for PDE1 SC-FNO
        """
        was_training: bool = self.model.training
        self.model.eval()

        # ------------------------------------------------------------------
        # Step 1: Collect solution predictions and ground truth.
        # ------------------------------------------------------------------
        u_pred_list: List[torch.Tensor] = []
        u_true_list: List[torch.Tensor] = []

        # Per-parameter Jacobian accumulators.
        j_pred_lists: List[List[torch.Tensor]] = [[] for _ in range(self.n_params)]
        j_true_lists: List[List[torch.Tensor]] = [[] for _ in range(self.n_params)]

        has_jacobians: bool = False

        for batch in self.test_loader:
            batch_dev: Dict[str, Any] = self._move_batch_to_device(batch)

            params: torch.Tensor = batch_dev["params"]
            u0: torch.Tensor = batch_dev["u0"]
            coords: torch.Tensor = batch_dev["coords"]
            u_true: torch.Tensor = batch_dev["u_true"]
            jacobians: Optional[torch.Tensor] = batch_dev.get("jacobians", None)

            # ------------------------------------------------------------------
            # Solution prediction (no grad needed for u metrics).
            # ------------------------------------------------------------------
            with torch.no_grad():
                u_pred: torch.Tensor = self.model(params, u0, coords)

            u_pred_list.append(u_pred.detach().cpu())
            u_true_list.append(u_true.detach().cpu())

            # ------------------------------------------------------------------
            # Jacobian prediction via AD (requires grad).
            # ------------------------------------------------------------------
            if jacobians is not None:
                has_jacobians = True
                j_pred_batch, j_true_batch = self._compute_batch_jacobians(
                    params=params,
                    u0=u0,
                    coords=coords,
                    jacobians=jacobians,
                )
                # j_pred_batch: list of n_params tensors, each [B, ...]
                # j_true_batch: list of n_params tensors, each [B, ...]
                for i in range(self.n_params):
                    j_pred_lists[i].append(j_pred_batch[i].detach().cpu())
                    j_true_lists[i].append(j_true_batch[i].detach().cpu())

        # ------------------------------------------------------------------
        # Step 2: Concatenate across batches.
        # ------------------------------------------------------------------
        if not u_pred_list:
            self.model.train(was_training)
            return {}

        u_pred_all: torch.Tensor = torch.cat(u_pred_list, dim=0)
        u_true_all: torch.Tensor = torch.cat(u_true_list, dim=0)

        # ------------------------------------------------------------------
        # Step 3: Compute solution metrics.
        # ------------------------------------------------------------------
        result: Dict[str, Any] = {}
        result["u"] = self.metrics.compute_all(u_pred_all, u_true_all)

        # ------------------------------------------------------------------
        # Step 4: Compute per-parameter sensitivity metrics.
        # ------------------------------------------------------------------
        if has_jacobians:
            for i, param_name in enumerate(self.param_names):
                if j_pred_lists[i] and j_true_lists[i]:
                    j_pred_all_i: torch.Tensor = torch.cat(j_pred_lists[i], dim=0)
                    j_true_all_i: torch.Tensor = torch.cat(j_true_lists[i], dim=0)
                    key: str = f"du_d{param_name}"
                    result[key] = self.metrics.compute_all(j_pred_all_i, j_true_all_i)

        self.model.train(was_training)
        return result

    def evaluate_sensitivity(self) -> dict:
        """Evaluates sensitivity accuracy ∂u/∂p on the test set.

        Focused version of the sensitivity portion of evaluate_forward().
        Computes model Jacobians via AD and compares with stored j_true.
        Useful when only sensitivity metrics are needed (e.g., after loading
        a pre-trained model without re-running the full forward evaluation).

        Returns:
            Nested dict with structure:
            {
                'du_d{param_name_0}': {'r2': float, 'relative_l2': float},
                'du_d{param_name_1}': {'r2': float, 'relative_l2': float},
                ...
            }
            Returns empty dict if test_loader has no Jacobians or is empty.

        Example:
            >>> sens_metrics = evaluator.evaluate_sensitivity()
            >>> sens_metrics['du_dalpha']['r2']   # e.g., 0.925 for PDE1 SC-FNO
        """
        was_training: bool = self.model.training
        self.model.eval()

        j_pred_lists: List[List[torch.Tensor]] = [[] for _ in range(self.n_params)]
        j_true_lists: List[List[torch.Tensor]] = [[] for _ in range(self.n_params)]
        has_jacobians: bool = False

        for batch in self.test_loader:
            batch_dev: Dict[str, Any] = self._move_batch_to_device(batch)

            params: torch.Tensor = batch_dev["params"]
            u0: torch.Tensor = batch_dev["u0"]
            coords: torch.Tensor = batch_dev["coords"]
            jacobians: Optional[torch.Tensor] = batch_dev.get("jacobians", None)

            if jacobians is None:
                continue

            has_jacobians = True
            j_pred_batch, j_true_batch = self._compute_batch_jacobians(
                params=params,
                u0=u0,
                coords=coords,
                jacobians=jacobians,
            )

            for i in range(self.n_params):
                j_pred_lists[i].append(j_pred_batch[i].detach().cpu())
                j_true_lists[i].append(j_true_batch[i].detach().cpu())

        result: Dict[str, Any] = {}

        if has_jacobians:
            for i, param_name in enumerate(self.param_names):
                if j_pred_lists[i] and j_true_lists[i]:
                    j_pred_all_i: torch.Tensor = torch.cat(j_pred_lists[i], dim=0)
                    j_true_all_i: torch.Tensor = torch.cat(j_true_lists[i], dim=0)
                    key: str = f"du_d{param_name}"
                    result[key] = self.metrics.compute_all(j_pred_all_i, j_true_all_i)

        self.model.train(was_training)
        return result

    def evaluate_perturbation(self, lambda_vals: List[float]) -> dict:
        """Evaluates model robustness when test parameters exceed training range.

        Reproduces Table 1 (right half) and Figure 5 from the paper. For each
        perturbation ratio λ, the test parameter range becomes [b, (1+λ)·b]
        for the upper tail of each parameter's range.

        Args:
            lambda_vals: List of perturbation ratios to evaluate. Sourced from
                         config.yaml key 'perturbation.lambda_values'
                         (default [0.1, 0.2, 0.3, 0.4]).

        Returns:
            Nested dict with structure:
            {
                0.1: {'u': {'r2': ..., 'relative_l2': ...},
                      'du_dalpha': {'r2': ..., 'relative_l2': ...}, ...},
                0.2: {...},
                0.3: {...},
                0.4: {...},
            }
            Returns empty dict if test_loader dataset does not support
            get_perturbed_split.

        Example:
            >>> lambda_vals = [0.1, 0.2, 0.3, 0.4]
            >>> perturb_results = evaluator.evaluate_perturbation(lambda_vals)
            >>> perturb_results[0.4]['u']['r2']  # e.g., 0.912 for SC-FNO PDE1
        """
        was_training: bool = self.model.training
        self.model.eval()

        # Retrieve the underlying dataset from the test_loader.
        original_dataset = self.test_loader.dataset
        if not hasattr(original_dataset, "get_perturbed_split"):
            print(
                "[Evaluator] WARNING: test_loader.dataset does not support "
                "get_perturbed_split(). Skipping perturbation evaluation."
            )
            self.model.train(was_training)
            return {}

        # Infer batch size from the test_loader.
        batch_size: int = self.test_loader.batch_size or 4

        result: Dict[float, Any] = {}

        for lambda_val in lambda_vals:
            print(
                f"[Evaluator] Evaluating perturbation λ={lambda_val:.2f}..."
            )

            # ------------------------------------------------------------------
            # Create perturbed dataset and DataLoader.
            # ------------------------------------------------------------------
            try:
                perturbed_dataset: SCFNODataset = original_dataset.get_perturbed_split(
                    lambda_perturb=lambda_val
                )
            except Exception as exc:
                print(
                    f"[Evaluator] WARNING: get_perturbed_split(λ={lambda_val}) "
                    f"failed: {exc}. Skipping this λ value."
                )
                continue

            perturbed_loader: DataLoader = DataLoader(
                perturbed_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                drop_last=False,
            )

            # ------------------------------------------------------------------
            # Temporarily swap test_loader and evaluate.
            # ------------------------------------------------------------------
            original_loader: DataLoader = self.test_loader
            self.test_loader = perturbed_loader

            try:
                lambda_metrics: dict = self.evaluate_forward()
            except Exception as exc:
                print(
                    f"[Evaluator] WARNING: evaluate_forward() failed for "
                    f"λ={lambda_val}: {exc}. Skipping."
                )
                lambda_metrics = {}
            finally:
                # Always restore the original test_loader.
                self.test_loader = original_loader

            result[lambda_val] = lambda_metrics

        # ------------------------------------------------------------------
        # Visualization: R² vs perturbation ratio (Figure 5).
        # ------------------------------------------------------------------
        if result:
            # Build r2_dict for the solution field u.
            variant: str = str(getattr(self.model, "variant", "fno"))
            r2_u_vals: List[float] = []
            for lv in lambda_vals:
                if lv in result and "u" in result[lv]:
                    r2_u_vals.append(float(result[lv]["u"].get("r2", float("nan"))))
                else:
                    r2_u_vals.append(float("nan"))

            r2_dict_u: Dict[str, List[float]] = {variant: r2_u_vals}
            title: str = f"{self.equation.upper()} {variant}"
            self.visualizer.plot_r2_vs_perturbation(
                lambda_vals=lambda_vals,
                r2_dict=r2_dict_u,
                metric="u",
                title=title,
            )

        self.model.train(was_training)
        return result

    def evaluate_inversion(self, inverter: Inverter) -> dict:
        """Evaluates parameter inversion accuracy using the trained surrogate.

        Reproduces Section 3.1, Figures 1, 2, and Tables D.11 from the paper.
        Runs both single-parameter inversion (invert α only) and multi-parameter
        inversion (invert all simultaneously) on the test set.

        Args:
            inverter: An Inverter instance initialized with the same model.
                      The Inverter handles the gradient-based optimization loop.

        Returns:
            Nested dict with structure:
            {
                'single_param': {
                    '{param_name_0}': {'r2': float, 'relative_l2': float}
                },
                'multi_param': {
                    '{param_name_0}': {'r2': float, 'relative_l2': float},
                    '{param_name_1}': {'r2': float, 'relative_l2': float},
                    ...
                }
            }
            Returns empty dict if test_loader is empty.

        Example:
            >>> inv_results = evaluator.evaluate_inversion(inverter)
            >>> inv_results['single_param']['alpha']['r2']  # e.g., 0.998 for SC-FNO
            >>> inv_results['multi_param']['alpha']['r2']   # e.g., 0.986 for SC-FNO
        """
        was_training: bool = self.model.training
        self.model.eval()

        # ------------------------------------------------------------------
        # Accumulators for single-parameter inversion (invert param index 0).
        # ------------------------------------------------------------------
        single_p_est_list: List[torch.Tensor] = []
        single_p_true_list: List[torch.Tensor] = []

        # ------------------------------------------------------------------
        # Accumulators for multi-parameter inversion.
        # ------------------------------------------------------------------
        multi_p_est_list: List[torch.Tensor] = []
        multi_p_true_list: List[torch.Tensor] = []

        # Limit the number of test samples for inversion to avoid excessive
        # computation (2000 steps × N_test samples is very expensive).
        # Use config value if available, otherwise default to 100 samples.
        max_inversion_samples: int = int(
            self.cfg.get("inversion", {}).get("max_eval_samples", 100)
        )
        n_evaluated: int = 0

        print(
            f"[Evaluator] Running inversion evaluation "
            f"(max_samples={max_inversion_samples})..."
        )

        for batch in self.test_loader:
            if n_evaluated >= max_inversion_samples:
                break

            batch_dev: Dict[str, Any] = self._move_batch_to_device(batch)

            params: torch.Tensor = batch_dev["params"]       # [B, n_params]
            u0: torch.Tensor = batch_dev["u0"]               # [B, M, Sx] etc.
            coords: torch.Tensor = batch_dev["coords"]       # [T_out, Sx, 2] etc.
            u_true: torch.Tensor = batch_dev["u_true"]       # [B, T_out, Sx] etc.

            B: int = params.shape[0]

            # Process each sample individually (inversion is per-sample).
            for sample_idx in range(B):
                if n_evaluated >= max_inversion_samples:
                    break

                # Extract single-sample tensors (remove batch dim).
                u_obs_i: torch.Tensor = u_true[sample_idx]       # [T_out, Sx] etc.
                u0_i: torch.Tensor = u0[sample_idx]              # [M, Sx] etc.
                true_params_i: torch.Tensor = params[sample_idx] # [n_params]

                # ------------------------------------------------------------------
                # Single-parameter inversion: invert param index 0 (first param).
                # ------------------------------------------------------------------
                try:
                    p_est_single: torch.Tensor = inverter.invert_single_param(
                        u_obs=u_obs_i,
                        u0=u0_i,
                        coords=coords,
                        param_idx=0,
                        true_params=true_params_i,
                    )
                    single_p_est_list.append(p_est_single.unsqueeze(0))
                    single_p_true_list.append(true_params_i.cpu().unsqueeze(0))
                except Exception as exc:
                    print(
                        f"[Evaluator] WARNING: single-param inversion failed "
                        f"for sample {n_evaluated}: {exc}"
                    )

                # ------------------------------------------------------------------
                # Multi-parameter inversion: invert all parameters simultaneously.
                # ------------------------------------------------------------------
                try:
                    p_est_multi: torch.Tensor = inverter.invert_all_params(
                        u_obs=u_obs_i,
                        u0=u0_i,
                        coords=coords,
                        true_params=true_params_i,
                    )
                    multi_p_est_list.append(p_est_multi.unsqueeze(0))
                    multi_p_true_list.append(true_params_i.cpu().unsqueeze(0))
                except Exception as exc:
                    print(
                        f"[Evaluator] WARNING: multi-param inversion failed "
                        f"for sample {n_evaluated}: {exc}"
                    )

                n_evaluated += 1

        print(
            f"[Evaluator] Inversion evaluation complete: "
            f"{n_evaluated} samples processed."
        )

        result: Dict[str, Any] = {
            "single_param": {},
            "multi_param": {},
        }

        # ------------------------------------------------------------------
        # Compute single-parameter inversion metrics (param index 0).
        # ------------------------------------------------------------------
        if single_p_est_list and single_p_true_list:
            p_est_single_all: torch.Tensor = torch.cat(single_p_est_list, dim=0)
            p_true_single_all: torch.Tensor = torch.cat(single_p_true_list, dim=0)

            # Only evaluate the inverted parameter (index 0).
            first_param_name: str = self.param_names[0] if self.param_names else "p0"
            result["single_param"][first_param_name] = self.metrics.compute_all(
                p_est_single_all[:, 0],
                p_true_single_all[:, 0],
            )

            # Visualization: scatter plot for single-parameter inversion (Figure 1a).
            variant: str = str(getattr(self.model, "variant", "fno"))
            self.visualizer.plot_inversion_scatter(
                p_pred=p_est_single_all[:, 0:1].numpy()
                if hasattr(p_est_single_all, "numpy")
                else p_est_single_all[:, 0:1].cpu().numpy(),
                p_true=p_true_single_all[:, 0:1].numpy()
                if hasattr(p_true_single_all, "numpy")
                else p_true_single_all[:, 0:1].cpu().numpy(),
                param_names=[first_param_name],
                model_names=[variant],
                title=f"{self.equation.upper()} Single-Param Inversion ({variant})",
            )

        # ------------------------------------------------------------------
        # Compute multi-parameter inversion metrics (all parameters).
        # ------------------------------------------------------------------
        if multi_p_est_list and multi_p_true_list:
            p_est_multi_all: torch.Tensor = torch.cat(multi_p_est_list, dim=0)
            p_true_multi_all: torch.Tensor = torch.cat(multi_p_true_list, dim=0)

            for i, param_name in enumerate(self.param_names):
                if i < p_est_multi_all.shape[1] and i < p_true_multi_all.shape[1]:
                    result["multi_param"][param_name] = self.metrics.compute_all(
                        p_est_multi_all[:, i],
                        p_true_multi_all[:, i],
                    )

            # Visualization: scatter plot for multi-parameter inversion (Figures 1b, 2).
            variant = str(getattr(self.model, "variant", "fno"))
            n_params_actual: int = min(
                p_est_multi_all.shape[1],
                p_true_multi_all.shape[1],
                len(self.param_names),
            )
            self.visualizer.plot_inversion_scatter(
                p_pred=p_est_multi_all[:, :n_params_actual].cpu().numpy(),
                p_true=p_true_multi_all[:, :n_params_actual].cpu().numpy(),
                param_names=self.param_names[:n_params_actual],
                model_names=[variant],
                title=f"{self.equation.upper()} Multi-Param Inversion ({variant})",
            )

        self.model.train(was_training)
        return result

    def evaluate_data_scaling(
        self,
        sample_sizes: List[int],
        train_dataset: SCFNODataset,
    ) -> dict:
        """Evaluates model performance across varying training data volumes.

        Reproduces Section 3.3 and Figure 4 from the paper. For each sample
        size N, trains a fresh model on a subset of train_dataset and evaluates
        on the fixed test set.

        Args:
            sample_sizes: List of training set sizes to evaluate. Sourced from
                          config.yaml key 'data_scaling.sample_sizes'
                          (default [100, 200, 500, 1000, 2000]).
            train_dataset: The full training dataset. Subsets are created by
                           taking the first N samples (after shuffling with
                           the global seed for reproducibility).

        Returns:
            Nested dict with structure:
            {
                100:  {'u': {'r2': ..., 'relative_l2': ...},
                       'du_d{param_name}': {'r2': ..., 'relative_l2': ...}, ...},
                200:  {...},
                500:  {...},
                1000: {...},
                2000: {...},
            }
            Returns empty dict if training fails for all sample sizes.

        Example:
            >>> sample_sizes = [100, 200, 500, 1000, 2000]
            >>> scaling_results = evaluator.evaluate_data_scaling(
            ...     sample_sizes, train_dataset
            ... )
            >>> scaling_results[500]['u']['r2']  # e.g., 0.9 for SC-FNO PDE1
        """
        # Local import to avoid circular dependency at module level.
        # Trainer imports DataLoss, SensitivityLoss, PINNLoss — none of which
        # import Evaluator, so the circular dependency is only at instantiation.
        from training.trainer import Trainer  # pylint: disable=import-outside-toplevel
        from models.sc_fno import build_model  # pylint: disable=import-outside-toplevel

        result: Dict[int, Any] = {}

        # Infer batch size from the equation-specific config.
        equation: str = self.equation
        eq_cfg: dict = self.cfg.get(equation, {})
        training_cfg: dict = eq_cfg.get("training", self.cfg.get("training", {}))
        batch_size: int = int(training_cfg.get("batch_size", 4))

        # Number of epochs for the scaling experiment.
        # Use a reduced epoch count for speed if specified in config.
        n_epochs_scaling: int = int(
            self.cfg.get("data_scaling", {}).get(
                "n_epochs",
                self.cfg.get("training", {}).get("n_epochs", 500),
            )
        )

        # Current model variant.
        variant: str = str(getattr(self.model, "variant", "fno"))

        # Shuffle indices with fixed seed for reproducibility.
        torch.manual_seed(self.seed)
        n_total: int = len(train_dataset)
        shuffled_indices: torch.Tensor = torch.randperm(n_total)

        # Build a validation DataLoader from the test_loader's dataset
        # (we use the test set as validation for the scaling experiment,
        # since we need a fixed evaluation set across all sample sizes).
        val_loader:
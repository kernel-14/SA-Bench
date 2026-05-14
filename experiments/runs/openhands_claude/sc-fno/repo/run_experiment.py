"""
Main experiment runner for SC-FNO paper reproduction.

Runs all four model variants (FNO, FNO-PINN, SC-FNO, SC-FNO-PINN) on
the specified equation, reproducing the experiments from Section 3.

Usage:
    python run_experiment.py --equation pde1 --n_samples 2000
    python run_experiment.py --equation ode1 --variants FNO SC-FNO
    python run_experiment.py --equation pde2_zoned --n_samples 100
"""

import argparse
import os
import time
from typing import Dict, List, Optional

import torch

from config import (
    ODE1_CONFIG,
    ODE2_CONFIG,
    PDE1_CONFIG,
    PDE2_CONFIG,
    PDE2_ZONED_100_CONFIG,
    PDE2_ZONED_500_CONFIG,
    PDE3_CONFIG,
    PDE4_100_CONFIG,
    PDE4_500_CONFIG,
    PERTURBATION_RATIOS,
    ExperimentConfig,
    TrainingConfig,
)
from data import (
    ODEDataset,
    PDE1DDataset,
    PDE2DDataset,
    make_dataloaders,
    save_dataset,
    load_dataset,
)
from equations import (
    ODE1Solver,
    ODE2Solver,
    PDE1Solver,
    PDE2Solver,
    PDE3Solver,
    PDE4Solver,
)
from evaluate import (
    ParameterInverter,
    compute_metrics,
    evaluate_robustness,
)
from train import Trainer
from utils import (
    build_fno_model,
    get_device,
    get_equation_type,
    print_model_info,
    save_results,
    set_seed,
)


EQUATION_CONFIGS = {
    "ode1": ODE1_CONFIG,
    "ode2": ODE2_CONFIG,
    "pde1": PDE1_CONFIG,
    "pde2": PDE2_CONFIG,
    "pde2_zoned_100": PDE2_ZONED_100_CONFIG,
    "pde2_zoned_500": PDE2_ZONED_500_CONFIG,
    "pde3": PDE3_CONFIG,
    "pde4_100": PDE4_100_CONFIG,
    "pde4_500": PDE4_500_CONFIG,
}


def generate_or_load_data(
    equation: str,
    n_samples: int,
    device: torch.device,
    data_dir: str = "data",
    use_ad: bool = True,
    force_regenerate: bool = False,
) -> Dict:
    """Generate or load cached dataset."""
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, f"{equation}_n{n_samples}.pt")

    if os.path.exists(data_path) and not force_regenerate:
        print(f"Loading cached dataset from {data_path}")
        return load_dataset(data_path)

    print(f"Generating dataset for {equation} with {n_samples} samples...")
    t0 = time.time()

    eq_name = equation.replace("_zoned", "").replace("_100", "").replace("_500", "")
    zoned = "zoned" in equation

    if eq_name == "ode1":
        solver = ODE1Solver(device=device)
        data = solver.generate_dataset(n_samples)
    elif eq_name == "ode2":
        solver = ODE2Solver(device=device)
        data = solver.generate_dataset(n_samples, use_ad=use_ad)
    elif eq_name == "pde1":
        solver = PDE1Solver(device=device)
        data = solver.generate_dataset(n_samples, use_ad=use_ad)
    elif eq_name == "pde2":
        solver = PDE2Solver(device=device, zoned=zoned)
        data = solver.generate_dataset(n_samples, use_ad=use_ad)
    elif eq_name == "pde3":
        solver = PDE3Solver(device=device)
        data = solver.generate_dataset(n_samples, use_ad=False)
    elif eq_name == "pde4":
        solver = PDE4Solver(device=device)
        data = solver.generate_dataset(n_samples, use_ad=use_ad)
    else:
        raise ValueError(f"Unknown equation: {equation}")

    elapsed = time.time() - t0
    print(f"Dataset generated in {elapsed:.1f}s")

    # Move to CPU for storage
    data_cpu = {k: v.cpu() for k, v in data.items()}
    save_dataset(data_cpu, data_path)
    return data_cpu


def build_dataset(equation: str, data: Dict, cfg: ExperimentConfig) -> torch.utils.data.Dataset:
    """Build the appropriate Dataset object from raw data."""
    eq_name = equation.replace("_zoned", "").replace("_100", "").replace("_500", "")

    if eq_name == "ode1":
        from equations.ode1 import ODE1Solver as S
        return ODEDataset(
            params=data["params"],
            u=data["u"],
            jacobian=data["jacobian"],
            M=S.M,
            t_start=S.t_start,
            t_end=S.t_end,
        )
    elif eq_name == "ode2":
        from equations.ode2 import ODE2Solver as S
        return ODEDataset(
            params=data["params"],
            u=data["u"],
            jacobian=data["jacobian"],
            M=S.M,
            t_start=S.t_start,
            t_end=S.t_end,
        )
    elif eq_name == "pde1":
        from equations.pde1 import PDE1Solver as S
        return PDE1DDataset(
            params=data["params"],
            u=data["u"],
            jacobian=data["jacobian"],
            M=S.M,
            x_start=S.x_start,
            x_end=S.x_end,
            t_start=S.t_start,
            t_end=S.t_end,
        )
    elif eq_name == "pde2":
        from equations.pde2 import PDE2Solver as S
        return PDE1DDataset(
            params=data["params"],
            u=data["u"],
            jacobian=data["jacobian"],
            M=S.M,
            x_start=S.x_start,
            x_end=S.x_end,
            t_start=S.t_start,
            t_end=S.t_end,
        )
    elif eq_name == "pde3":
        from equations.pde3 import PDE3Solver as S
        return PDE2DDataset(
            params=data["params"],
            omega0=data["omega0"],
            omega_final=data["u"],
            jacobian=data["jacobian"],
        )
    elif eq_name == "pde4":
        from equations.pde4 import PDE4Solver as S
        return PDE1DDataset(
            params=data["params"],
            u=data["u"],
            jacobian=data["jacobian"],
            M=S.M,
            x_start=S.x_start,
            x_end=S.x_end,
            t_start=S.t_start,
            t_end=S.t_end,
        )
    else:
        raise ValueError(f"Unknown equation: {equation}")


def get_pinn_residual_fn(equation: str, device: torch.device):
    """Get the PINN residual function for a given equation."""
    eq_name = equation.replace("_zoned", "").replace("_100", "").replace("_500", "")

    if eq_name == "ode1":
        solver = ODE1Solver(device=device)
        return solver.pinn_residual
    elif eq_name == "ode2":
        solver = ODE2Solver(device=device)
        return solver.pinn_residual
    elif eq_name == "pde1":
        solver = PDE1Solver(device=device)
        return solver.pinn_residual
    elif eq_name == "pde2":
        solver = PDE2Solver(device=device)
        return solver.pinn_residual
    elif eq_name == "pde4":
        solver = PDE4Solver(device=device)
        return solver.pinn_residual
    else:
        return None  # PDE3 PINN not implemented


def run_experiment(
    equation: str,
    variants: List[str],
    n_samples: Optional[int] = None,
    n_epochs: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
    data_dir: str = "data",
    checkpoint_dir: str = "checkpoints",
    results_dir: str = "results",
    use_ad: bool = True,
    force_regenerate: bool = False,
    seed: int = 42,
) -> Dict:
    """
    Run SC-FNO experiments for a given equation and set of model variants.

    Args:
        equation: equation identifier (e.g., "pde1", "ode1")
        variants: list of model variants to train
        n_samples: number of training samples (uses config default if None)
        n_epochs: number of training epochs (uses config default if None)
        device: computation device
        data_dir: directory for cached datasets
        checkpoint_dir: directory for model checkpoints
        results_dir: directory for experiment results
        use_ad: use automatic differentiation for Jacobians
        force_regenerate: regenerate dataset even if cached
        seed: random seed

    Returns:
        all_results: dict mapping variant → metrics
    """
    set_seed(seed)
    os.makedirs(results_dir, exist_ok=True)

    # Get configuration
    cfg_key = equation
    if cfg_key not in EQUATION_CONFIGS:
        # Try to find a matching config
        for k in EQUATION_CONFIGS:
            if equation in k:
                cfg_key = k
                break
    cfg = EQUATION_CONFIGS.get(cfg_key, PDE1_CONFIG)

    if n_samples is not None:
        cfg.n_samples = n_samples
    if n_epochs is not None:
        cfg.fno.n_epochs = n_epochs

    equation_type = get_equation_type(cfg.equation)

    # Generate or load data
    data = generate_or_load_data(
        equation=cfg.equation,
        n_samples=cfg.n_samples,
        device=device,
        data_dir=data_dir,
        use_ad=use_ad,
        force_regenerate=force_regenerate,
    )

    # Build dataset and dataloaders
    dataset = build_dataset(cfg.equation, data, cfg)
    train_loader, val_loader, test_loader = make_dataloaders(
        dataset,
        batch_size=cfg.batch_size,
        train_frac=cfg.training.train_frac,
        val_frac=cfg.training.val_frac,
        seed=seed,
    )

    print(f"\nDataset: {len(dataset)} samples | "
          f"Train: {len(train_loader.dataset)} | "
          f"Val: {len(val_loader.dataset)} | "
          f"Test: {len(test_loader.dataset)}")

    # PINN residual function
    pinn_fn = get_pinn_residual_fn(cfg.equation, device)

    all_results = {}

    for variant in variants:
        print(f"\n{'='*60}")
        print(f"Training {variant} on {equation}")
        print(f"{'='*60}")

        # Build model
        model = build_fno_model(cfg.equation, cfg.fno, cfg.n_params, device)
        print_model_info(model, variant)

        # Build trainer
        trainer = Trainer(
            model=model,
            variant=variant,
            equation_type=equation_type,
            c1=cfg.training.c1,
            c2=cfg.training.c2,
            c3=cfg.training.c3,
            learning_rate=cfg.fno.learning_rate,
            n_spatial_samples=cfg.training.n_spatial_samples,
            n_time_samples=cfg.training.n_time_samples,
            pinn_residual_fn=pinn_fn if "PINN" in variant else None,
            device=device,
            checkpoint_dir=os.path.join(checkpoint_dir, equation),
        )

        # Train
        exp_name = f"{equation}_{variant.replace('-', '_')}"
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=cfg.fno.n_epochs,
            save_best=True,
            experiment_name=exp_name,
        )

        # Evaluate on test set
        print(f"\nEvaluating {variant} on test set...")
        test_metrics = trainer.evaluate(test_loader)
        print(f"Test metrics: {test_metrics}")

        all_results[variant] = {
            "history": history,
            "test_metrics": test_metrics,
        }

    # Save results
    results_path = os.path.join(results_dir, f"{equation}_results.pt")
    save_results(all_results, results_path)
    print(f"\nResults saved to {results_path}")

    return all_results


def run_inversion_experiment(
    equation: str,
    model_checkpoints: Dict[str, str],
    n_test_samples: int = 100,
    device: torch.device = torch.device("cpu"),
    data_dir: str = "data",
    results_dir: str = "results",
    invert_single_param: bool = True,
    param_idx: int = 0,
) -> Dict:
    """
    Run parameter inversion experiments (Section 3.1).

    Args:
        equation: equation identifier
        model_checkpoints: dict mapping variant → checkpoint path
        n_test_samples: number of test samples for inversion
        device: computation device
        data_dir: directory for cached datasets
        results_dir: directory for results
        invert_single_param: if True, invert only one parameter (others fixed)
        param_idx: index of parameter to invert (for single-param case)

    Returns:
        inversion_results: dict mapping variant → inversion metrics
    """
    from config import PARAM_RANGES

    cfg_key = equation
    cfg = EQUATION_CONFIGS.get(cfg_key, PDE1_CONFIG)
    equation_type = get_equation_type(cfg.equation)

    # Load test data
    data = generate_or_load_data(
        equation=cfg.equation,
        n_samples=cfg.n_samples,
        device=device,
        data_dir=data_dir,
    )

    dataset = build_dataset(cfg.equation, data, cfg)
    _, _, test_loader = make_dataloaders(
        dataset, batch_size=1, train_frac=cfg.training.train_frac, val_frac=cfg.training.val_frac
    )

    param_ranges = PARAM_RANGES.get(cfg.equation, {})
    inversion_results = {}

    for variant, ckpt_path in model_checkpoints.items():
        print(f"\nRunning inversion with {variant}...")

        model = build_fno_model(cfg.equation, cfg.fno, cfg.n_params, device)
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        inverter = ParameterInverter(
            model=model,
            equation_type=equation_type,
            param_ranges=param_ranges,
            device=device,
            n_iter=1000,
            lr=1e-2,
        )

        p_pred_list = []
        p_true_list = []
        count = 0

        for batch in test_loader:
            if count >= n_test_samples:
                break

            u_obs = batch["u_out"].to(device)
            fno_input = batch["fno_input"].to(device)
            p_true = batch["params"].to(device)

            fixed_params = None
            if invert_single_param:
                # Fix all parameters except param_idx
                fixed_params = {
                    i: p_true[0, i].item()
                    for i in range(cfg.n_params)
                    if i != param_idx
                }

            p_opt, _ = inverter.invert(u_obs, fno_input, fixed_params=fixed_params)
            p_pred_list.append(p_opt.cpu())
            p_true_list.append(p_true.cpu())
            count += 1

        p_pred_all = torch.cat(p_pred_list, dim=0)
        p_true_all = torch.cat(p_true_list, dim=0)

        param_names = list(param_ranges.keys())
        metrics = {}
        for p_idx, name in enumerate(param_names):
            metrics[name] = compute_metrics(p_pred_all[:, p_idx], p_true_all[:, p_idx])

        inversion_results[variant] = {
            "p_pred": p_pred_all,
            "p_true": p_true_all,
            "metrics": metrics,
        }
        print(f"  {variant} inversion metrics: {metrics}")

    return inversion_results


def main():
    parser = argparse.ArgumentParser(description="SC-FNO Experiment Runner")
    parser.add_argument(
        "--equation",
        type=str,
        default="pde1",
        choices=list(EQUATION_CONFIGS.keys()),
        help="Equation to run experiments on",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["FNO", "SC-FNO"],
        choices=["FNO", "FNO-PINN", "SC-FNO", "SC-FNO-PINN"],
        help="Model variants to train",
    )
    parser.add_argument("--n_samples", type=int, default=None, help="Number of training samples")
    parser.add_argument("--n_epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--results_dir", type=str, default="results", help="Results directory")
    parser.add_argument("--use_fd", action="store_true", help="Use finite differences instead of AD")
    parser.add_argument("--force_regenerate", action="store_true", help="Regenerate dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run_inversion", action="store_true", help="Run inversion experiments")

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    results = run_experiment(
        equation=args.equation,
        variants=args.variants,
        n_samples=args.n_samples,
        n_epochs=args.n_epochs,
        device=device,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
        use_ad=not args.use_fd,
        force_regenerate=args.force_regenerate,
        seed=args.seed,
    )

    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    for variant, res in results.items():
        print(f"\n{variant}:")
        print(f"  Test L_u: {res['test_metrics']['L_u']:.4f}")


if __name__ == "__main__":
    main()

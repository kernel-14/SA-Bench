#!/usr/bin/env python3
"""Main training script for SC-FNO experiments.

Usage:
    python train.py --equation pde1 --model sc-fno
    python train.py --equation pde1 --model fno
    python train.py --equation pde1 --model fno-pinn
    python train.py --equation pde1 --model sc-fno-pinn
    python train.py --equation pde1 --model sc-fno --solver fd
    python train.py --equation pde1 --model sc-fno --invert

All models share the same FNO architecture (per Section 2.4).
Only the loss function configuration differs between models.
"""

import argparse
import json
import os
import random
from typing import Any, Dict, List

import numpy as np
import torch

from sc_fno.configs import get_config
from sc_fno.equations import (
    ODE1Solver,
    ODE2Solver,
    PDE1Solver,
    PDE2Solver,
    PDE3Solver,
    PDE4Solver,
    generate_dataset,
)
from sc_fno.models import FNO
from sc_fno.training import (
    FNOTrainer,
    FNOTrainerPINN,
    SCFNOTrainer,
    SCFNOTrainerPINN,
    invert_parameters,
    invert_parameters_multi,
    train_model,
)
from sc_fno.utils import compute_all_metrics, prepare_dataloaders


SOLVER_MAP = {
    "ode1": ODE1Solver,
    "ode2": ODE2Solver,
    "pde1": PDE1Solver,
    "pde2": PDE2Solver,
    "pde3": PDE3Solver,
    "pde4": PDE4Solver,
}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(config: Dict[str, Any]) -> FNO:
    """Build FNO model from configuration."""
    mp = config["model_params"]
    return FNO(
        modes1=mp["modes1"],
        modes2=mp.get("modes2", 0),
        modes_t=mp.get("modes_t", 0),
        width=mp["width"],
        n_layers=mp["n_layers"],
        input_channels=mp["input_channels"],
        output_channels=mp["output_channels"],
        ndim=mp.get("ndim", "1d"),
    )


def build_trainer(
    model: FNO,
    device: torch.device,
    config: Dict[str, Any],
    model_type: str,
) -> Any:
    """Build the appropriate trainer for the model type.

    Args:
        model: FNO model.
        device: Torch device.
        config: Configuration dict.
        model_type: One of "fno", "sc-fno", "fno-pinn", "sc-fno-pinn".

    Returns:
        Trainer instance.
    """
    tp = config["training_params"]
    c1 = tp.get("c1", 1.0)
    c2 = tp.get("c2", 1.0)
    c3 = tp.get("c3", 0.1)
    eq_type = config["equation"]

    if model_type == "fno":
        return FNOTrainer(model, device, lr=tp["lr"], c1=c1, c2=c2, c3=c3)
    elif model_type == "sc-fno":
        return SCFNOTrainer(
            model, device,
            param_names=config["param_names"],
            lr=tp["lr"], c1=c1, c2=c2, c3=c3,
        )
    elif model_type == "fno-pinn":
        return FNOTrainerPINN(
            model, device,
            eq_type=eq_type,
            lr=tp["lr"], c1=c1, c2=c2, c3=c3,
        )
    elif model_type == "sc-fno-pinn":
        return SCFNOTrainerPINN(
            model, device,
            param_names=config["param_names"],
            eq_type=eq_type,
            lr=tp["lr"], c1=c1, c2=c2, c3=c3,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def evaluate_model(
    model: FNO,
    test_loader: torch.utils.data.DataLoader,
    param_names: List[str],
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on test set for both solution paths and Jacobians.

    Returns metrics matching those in Tables 1-4 and Table D.14.
    """
    model.eval()
    all_u_pred = []
    all_u_true = []

    all_du_pred = {name: [] for name in param_names}
    all_du_true = {name: [] for name in param_names}

    for batch in test_loader:
        num_items = len(batch)
        x = batch[0].to(device)
        u_true = batch[1].to(device)

        x.requires_grad_(True)
        u_pred = model(x)

        all_u_pred.append(u_pred.detach())
        all_u_true.append(u_true)

        n_params = len(param_names)
        param_start_idx = x.shape[1] - n_params

        u_flat = u_pred.reshape(u_pred.shape[0], -1)
        n_total_out = u_flat.shape[1]

        du_pred_dict = {name: torch.zeros_like(u_pred) for name in param_names}

        for pt in range(min(n_total_out, 200)):
            grad_out = torch.zeros_like(u_pred)
            grad_out_flat = grad_out.reshape(x.shape[0], -1)
            grad_out_flat[:, pt] = 1.0

            grad_x = torch.autograd.grad(
                u_pred, x,
                grad_outputs=grad_out,
                create_graph=False,
                retain_graph=True,
            )[0]

            for j, name in enumerate(param_names):
                du_pred_dict[name].reshape(x.shape[0], -1)[:, pt] = \
                    grad_x[:, param_start_idx + j, ...].reshape(x.shape[0], -1).sum(dim=1)

        for j, name in enumerate(param_names):
            if num_items >= 3 + j:
                du_true = batch[2 + j].to(device)
                all_du_pred[name].append(du_pred_dict[name].detach())
                all_du_true[name].append(du_true)

    u_pred_all = torch.cat(all_u_pred, dim=0)
    u_true_all = torch.cat(all_u_true, dim=0)
    u_metrics = compute_all_metrics(u_pred_all, u_true_all, "u")

    jac_metrics = {}
    for name in param_names:
        if all_du_pred[name]:
            du_p = torch.cat(all_du_pred[name], dim=0)
            du_t = torch.cat(all_du_true[name], dim=0)
            jac_metrics[name] = compute_all_metrics(du_p, du_t, name)

    avg_jac_r2 = np.mean([m[f"{n}_R2"] for n, m in jac_metrics.items()]) if jac_metrics else 0.0
    avg_jac_l2 = np.mean([m[f"{n}_relative_L2"] for n, m in jac_metrics.items()]) if jac_metrics else 0.0

    jac_metrics["avg_R2"] = avg_jac_r2
    jac_metrics["avg_relative_L2"] = avg_jac_l2

    return {"u": u_metrics, "jacobians": jac_metrics}


def evaluate_perturbed(
    model: FNO,
    config: Dict[str, Any],
    param_names: List[str],
    device: torch.device,
    perturbation_lambda: float = 0.4,
) -> Dict[str, float]:
    """Evaluate model on perturbed parameter ranges.

    Perturbed range: (a, (1+λ)*b) where [a, b] is the training range.
    Per Section 3.2 and Table 1.
    """
    solver_cls = SOLVER_MAP[config["equation"]]
    solver_params = config["solver_params"]
    tp = config["training_params"]
    M = tp["M"]
    n_test = 200

    orig_ranges = config["param_ranges"]
    perturbed_ranges = {}
    for name, (lo, hi) in orig_ranges.items():
        perturbed_ranges[name] = [lo, hi * (1 + perturbation_lambda)]

    test_data = generate_dataset(
        solver_cls, perturbed_ranges, n_samples=n_test, solver_params=solver_params
    )

    batch_size = tp["batch_size"]

    loader, _, _, _ = prepare_dataloaders(
        test_data, param_names, config["equation"], M,
        batch_size=batch_size,
        train_ratio=1.0, val_ratio=0.0, test_ratio=0.0,
    )

    return evaluate_model(model, loader, param_names, device)


def run_inversion(
    model: FNO,
    config: Dict[str, Any],
    device: torch.device,
    single_param: bool = True,
) -> Dict[str, Any]:
    """Run parameter inversion experiments (Section 3.1).

    Args:
        model: Trained model.
        config: Configuration.
        device: Torch device.
        single_param: If True, invert only alpha. Otherwise invert all params.

    Returns:
        Dict with inversion results.
    """
    solver_cls = SOLVER_MAP[config["equation"]]
    solver_params = config["solver_params"]
    tp = config["training_params"]
    M = tp["M"]
    param_names = config["param_names"]
    param_ranges = config["param_ranges"]

    n_test = 100
    test_data = generate_dataset(
        solver_cls, param_ranges, n_samples=n_test, solver_params=solver_params
    )

    results = []
    for sample in test_data:
        params_true = sample["params"]
        u_true = sample["u"]

        if config["equation"] in ("ode1", "ode2"):
            x_in = _build_ode_input(sample, param_names, M).to(device)
            u_target = u_true[M:].to(device)
        elif config["equation"] == "pde3":
            x_in = _build_pde3_input(sample, param_names).to(device)
            u_target = u_true.to(device)
        else:
            x_in = _build_pde_input(sample, param_names, M).to(device)
            u_target = u_true[:, M:].to(device)

        if single_param:
            invert_names = [pn for pn in param_names if "alpha" in pn][:1]
            if not invert_names:
                invert_names = param_names[:1]
        else:
            invert_names = param_names

        param_init = {
            name: float(torch.rand(1).item() * (param_ranges[name][1] - param_ranges[name][0]) + param_ranges[name][0])
            for name in invert_names
        }

        optimized = invert_parameters(
            model, u_target.unsqueeze(0), x_in.unsqueeze(0),
            invert_names, param_init, param_ranges,
            n_iterations=200, lr=0.01, verbose=False,
        )

        for name in invert_names:
            results.append({
                "param": name,
                "true": params_true[name],
                "predicted": optimized[name],
            })

    r2_by_param = {}
    l2_by_param = {}
    for name in param_names:
        entries = [r for r in results if r["param"] == name]
        if entries:
            pred = torch.tensor([e["predicted"] for e in entries])
            true = torch.tensor([e["true"] for e in entries])
            r2_by_param[name] = 1 - torch.sum((true - pred)**2) / (torch.sum((true - true.mean())**2) + 1e-12)
            l2_by_param[name] = torch.norm(true - pred) / (torch.norm(true) + 1e-12)
            r2_by_param[name] = max(min(r2_by_param[name].item(), 1.0), -20.0)
            l2_by_param[name] = l2_by_param[name].item()

    return {"R2": r2_by_param, "rel_L2": l2_by_param}


def _build_ode_input(sample, param_names, M):
    from sc_fno.utils.data import build_input_tensor_ode
    return build_input_tensor_ode(sample["u"][:M], sample["t"], sample["params"], param_names)


def _build_pde_input(sample, param_names, M):
    from sc_fno.utils.data import build_input_tensor_pde
    return build_input_tensor_pde(sample["u"][:, :M], sample["x"], sample["t"], sample["params"], param_names, M)


def _build_pde3_input(sample, param_names):
    from sc_fno.utils.data import build_input_tensor_pde3
    return build_input_tensor_pde3(sample["omega"], sample["x"], sample["y"], sample["params"], param_names)


def main():
    parser = argparse.ArgumentParser(description="SC-FNO Training and Evaluation")
    parser.add_argument("--equation", type=str, default="pde1",
                        choices=["ode1", "ode2", "pde1", "pde2", "pde3", "pde4", "pde2_zoned"])
    parser.add_argument("--model", type=str, default="sc-fno",
                        choices=["fno", "sc-fno", "fno-pinn", "sc-fno-pinn"])
    parser.add_argument("--solver", type=str, default="ad",
                        choices=["ad", "fd"])
    parser.add_argument("--invert", action="store_true",
                        help="Run parameter inversion after training.")
    parser.add_argument("--perturb", action="store_true",
                        help="Evaluate on perturbed parameter range.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Override training sample count.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu).")

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = get_config(args.equation)
    param_names = config["param_names"]

    if args.n_samples is not None:
        config["training_params"]["n_train_samples"] = args.n_samples

    tp = config["training_params"]
    n_samples = tp["n_train_samples"]

    print(f"Configuration: {args.equation} | Model: {args.model} | N samples: {n_samples}")
    print(f"Parameters: {param_names}")
    print(f"Parameter ranges: {config['param_ranges']}")

    solver_cls = SOLVER_MAP[args.equation]
    solver_params = config["solver_params"]

    print(f"Generating {n_samples} samples...")
    dataset = generate_dataset(
        solver_cls, config["param_ranges"], n_samples=n_samples,
        solver_params=solver_params,
    )

    M = tp["M"]

    train_loader, val_loader, test_loader, _ = prepare_dataloaders(
        dataset, param_names, args.equation, M,
        batch_size=tp["batch_size"],
    )

    print(f"Data split: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")

    model = build_model(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params}")

    trainer = build_trainer(model, device, config, args.model)

    print(f"Training {args.model} for {tp['n_epochs']} epochs...")
    history = train_model(
        model, train_loader, val_loader, trainer,
        n_epochs=tp["n_epochs"], verbose=True,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, f"{args.equation}_{args.model}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    print("\nEvaluating on test set...")
    metrics = evaluate_model(model, test_loader, param_names, device)
    print(f"  Solution u: R²={metrics['u']['u_R2']:.4f}, rel_L2={metrics['u']['u_relative_L2']:.4f}")
    if metrics["jacobians"]:
        print(f"  Avg Jacobian: R²={metrics['jacobians']['avg_R2']:.4f}, rel_L2={metrics['jacobians']['avg_relative_L2']:.4f}")
        for name in param_names:
            if name in metrics["jacobians"]:
                jm = metrics["jacobians"][name]
                print(f"    du/d{name}: R²={jm[f'{name}_R2']:.4f}, rel_L2={jm[f'{name}_relative_L2']:.4f}")

    if args.perturb:
        print("\nEvaluating on perturbed parameter range (λ=0.4)...")
        pert_metrics = evaluate_perturbed(model, config, param_names, device, 0.4)
        print(f"  Solution u: R²={pert_metrics['u']['u_R2']:.4f}, rel_L2={pert_metrics['u']['u_relative_L2']:.4f}")
        if pert_metrics["jacobians"]:
            print(f"  Avg Jacobian: R²={pert_metrics['jacobians']['avg_R2']:.4f}")

    if args.invert:
        print("\nRunning single-parameter inversion...")
        inv_single = run_inversion(model, config, device, single_param=True)
        print("  Single parameter inversion results:")
        for name in inv_single["R2"]:
            print(f"    {name}: R²={inv_single['R2'][name]:.4f}, rel_L2={inv_single['rel_L2'][name]:.4f}")

        print("\nRunning multi-parameter inversion...")
        inv_multi = run_inversion(model, config, device, single_param=False)
        print("  Multi-parameter inversion results:")
        for name in inv_multi["R2"]:
            print(f"    {name}: R²={inv_multi['R2'][name]:.4f}, rel_L2={inv_multi['rel_L2'][name]:.4f}")

    all_metrics = {
        "test_metrics": {k: v for k, v in metrics.items()},
    }
    if args.invert:
        all_metrics["inversion_single"] = inv_single
        all_metrics["inversion_multi"] = inv_multi

    metrics_path = os.path.join(args.output_dir, f"{args.equation}_{args.model}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\nMetrics saved to {metrics_path}")


if __name__ == "__main__":
    main()

"""
Standalone script to pre-generate all datasets for SC-FNO experiments.

Datasets are saved to disk and reused during training.
This is the "one-time cost per equation" described in Section 2.4.

Usage:
    python generate_data.py --equations ode1 pde1 pde2 --n_samples 2000
    python generate_data.py --all --use_fd  # use finite differences
"""

import argparse
import os
import time

import torch

from equations import ODE1Solver, ODE2Solver, PDE1Solver, PDE2Solver, PDE3Solver, PDE4Solver
from data import save_dataset
from utils import get_device, set_seed


SOLVERS = {
    "ode1": (ODE1Solver, 2000, False),   # (solver_class, default_n, needs_zoned)
    "ode2": (ODE2Solver, 2000, False),
    "pde1": (PDE1Solver, 2000, False),
    "pde2": (PDE2Solver, 2000, False),
    "pde2_zoned": (PDE2Solver, 500, True),
    "pde3": (PDE3Solver, 1000, False),
    "pde4": (PDE4Solver, 500, False),
}


def generate_dataset(
    equation: str,
    n_samples: int,
    device: torch.device,
    use_ad: bool = True,
    data_dir: str = "data",
    force: bool = False,
) -> None:
    """Generate and save a dataset for the given equation."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f"{equation}_n{n_samples}.pt")

    if os.path.exists(path) and not force:
        print(f"  Skipping {equation} (already exists at {path})")
        return

    solver_cls, _, zoned = SOLVERS[equation]
    print(f"  Generating {equation} with {n_samples} samples (AD={use_ad})...")
    t0 = time.time()

    if equation == "pde2_zoned":
        solver = solver_cls(device=device, zoned=True)
    else:
        solver = solver_cls(device=device)

    if equation == "ode1":
        data = solver.generate_dataset(n_samples)
    elif equation == "pde3":
        data = solver.generate_dataset(n_samples, use_ad=False)
    else:
        data = solver.generate_dataset(n_samples, use_ad=use_ad)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # Print dataset info
    for k, v in data.items():
        print(f"    {k}: {v.shape}")

    data_cpu = {k: v.cpu() for k, v in data.items()}
    save_dataset(data_cpu, path)
    print(f"  Saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate SC-FNO datasets")
    parser.add_argument(
        "--equations",
        nargs="+",
        default=["ode1"],
        choices=list(SOLVERS.keys()),
        help="Equations to generate data for",
    )
    parser.add_argument("--all", action="store_true", help="Generate all datasets")
    parser.add_argument("--n_samples", type=int, default=None, help="Override sample count")
    parser.add_argument("--use_fd", action="store_true", help="Use finite differences")
    parser.add_argument("--data_dir", type=str, default="data", help="Output directory")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--force", action="store_true", help="Regenerate existing datasets")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)

    set_seed(args.seed)
    print(f"Using device: {device}")

    equations = list(SOLVERS.keys()) if args.all else args.equations

    for eq in equations:
        _, default_n, _ = SOLVERS[eq]
        n = args.n_samples if args.n_samples is not None else default_n
        print(f"\n[{eq}]")
        generate_dataset(
            equation=eq,
            n_samples=n,
            device=device,
            use_ad=not args.use_fd,
            data_dir=args.data_dir,
            force=args.force,
        )

    print("\nAll datasets generated successfully.")


if __name__ == "__main__":
    main()

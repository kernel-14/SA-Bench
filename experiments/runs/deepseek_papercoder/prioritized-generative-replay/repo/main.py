## main.py

"""
Entry point for the Prioritized Generative Replay (PGR) experiment.

This script:
    1. Parses command‑line arguments (and an optional YAML configuration file)
       to build a `Config` object.
    2. Sets global random seeds for reproducibility.
    3. Optionally initialises Weights & Biases logging.
    4. Creates an instance of `PGRAlgorithm` and runs the outer/inner loop
       training process.
"""

import dataclasses
import sys

from config import parse_args                                   # from config.py
from pgr import PGRAlgorithm                                    # from pgr.py
from utils import set_seeds                                     # from utils.py


def main() -> None:
    """Main entry point."""
    # ------------------------------------------------------------------
    # 1. Load configuration (YAML defaults + command‑line overrides)
    # ------------------------------------------------------------------
    cfg = parse_args()   # returns a fully populated Config instance

    # ------------------------------------------------------------------
    # 2. Reproducibility – seed every random number generator used
    # ------------------------------------------------------------------
    set_seeds(cfg.environment.random_seed)

    # ------------------------------------------------------------------
    # 3. Optional Weights & Biases logging
    # ------------------------------------------------------------------
    if cfg.logging.use_wandb:
        import wandb
        wandb.init(
            project=cfg.logging.project_name,
            name=cfg.logging.run_name,
            config=dataclasses.asdict(cfg),
        )

    # ------------------------------------------------------------------
    # 4. Create the PGR algorithm (this initialises all modules:
    #    environment, replay buffers, policy, relevance function,
    #    conditional diffusion model, and the outer/inner loop logic)
    # ------------------------------------------------------------------
    algo = PGRAlgorithm(cfg)

    # ------------------------------------------------------------------
    # 5. Execute the training/evaluation loop
    # ------------------------------------------------------------------
    try:
        algo.run()
    except KeyboardInterrupt:
        print("Training interrupted by user. Shutting down gracefully...")
    finally:
        if cfg.logging.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()


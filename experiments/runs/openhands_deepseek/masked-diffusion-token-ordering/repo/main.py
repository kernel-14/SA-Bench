import os
import sys
import argparse
import torch
import numpy as np
import random

from config import ExperimentConfig, get_default_config
from models import MaskedDiffusionModel, get_num_params
from diffusion import get_noise_schedule
from inference import get_alpha_schedule, sample_mdm
from data import get_dataloader, sample_permutation
from train import train_mdm, train_pi_learner, train_arm
from evaluate import run_full_evaluation


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Masked Diffusion: Token Ordering")
    parser.add_argument("--preset", type=str, default="base",
                        choices=["base", "sudoku_6m", "zebra_19m", "lonaesat_19m",
                                 "text_170m", "arm_42m_sudoku", "arm_42m_zebra",
                                 "text_pi_learner"])
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "train_pi", "train_arm", "eval"])
    parser.add_argument("--strategy", type=str, default="top_probability_margin",
                        choices=["vanilla", "top_probability", "top_probability_margin"])
    parser.add_argument("--order_info", action="store_true",
                        help="For ARM training with ordering information")
    parser.add_argument("--hard", action="store_true",
                        help="Use hard Sudoku test set")
    parser.add_argument("--pi_distribution", type=str, default="uniform",
                        help="Permutation distribution for pi-learner")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to dataset file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint to load")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    cfg = get_default_config(args.preset)
    cfg.output_dir = args.output_dir
    if args.wandb:
        cfg.wandb_project = "mdm-token-ordering"

    if args.pi_distribution:
        cfg.data.pi_distribution = args.pi_distribution

    set_seed(cfg.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Preset: {args.preset}")
    print(f"Model: {cfg.model}")
    print(f"Params: ~{get_num_params(MaskedDiffusionModel(cfg.model)) / 1e6:.1f}M")
    print(f"Device: {device}")

    if args.mode in ["train", "train_pi", "train_arm"]:
        dataloader = get_dataloader(
            cfg.data, cfg.model, cfg.training.batch_size,
            split="train", data_path=args.data_path,
        )

        if args.mode == "train":
            model = train_mdm(cfg, dataloader, device=device)
        elif args.mode == "train_pi":
            L = cfg.model.max_seq_len
            pi = sample_permutation(
                L, cfg.data.pi_distribution,
                np.random.RandomState(cfg.seed)
            )
            print(f"Training π-learner with distribution: {cfg.data.pi_distribution}")
            model = train_pi_learner(cfg, dataloader, pi, device=device)
        elif args.mode == "train_arm":
            print(f"Training ARM {'with' if args.order_info else 'without'} ordering info")
            model = train_arm(cfg, dataloader, device=device, order_info=args.order_info)

    elif args.mode == "eval":
        model = MaskedDiffusionModel(cfg.model).to(device)
        if args.checkpoint:
            ckpt = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        model.eval()

        cfg.inference.strategy = args.strategy

        if args.hard:
            dataloader = get_dataloader(
                cfg.data, cfg.model, cfg.training.batch_size,
                split="hard_test", data_path=args.data_path,
            )

        results = run_full_evaluation(model, cfg, args.output_dir, device=device)
        print("\n=== Results ===")
        for k, v in results.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

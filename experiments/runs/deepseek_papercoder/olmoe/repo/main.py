# main.py
"""
OLMoE-1B-7B Reproduction Entry Point

This script is the single entry point for all stages of the OLMoE pipeline:
  - pretrain : full pretraining of the Mixture-of-Experts transformer
  - eval     : OLMES or instruct benchmark evaluation
  - adapt    : instruction tuning (SFT) followed by preference tuning (DPO)

All hyperparameters are read from a single YAML configuration file that follows
the structure of ``config.yaml`` (see the project root).

Usage (examples):
    # Pretraining on 8 GPUs (launched with torchrun)
    torchrun --nproc_per_node=8 main.py --mode pretrain --config config.yaml

    # Evaluate a pretrained checkpoint on OLMES
    python main.py --mode eval --config config.yaml --checkpoint path/to/model.pt

    # Run adaptation (SFT + DPO) starting from a pretrained annealed checkpoint
    torchrun --nproc_per_node=8 main.py --mode adapt --config config.yaml \\
             --checkpoint path/to/annealed_model.pt

The script handles distributed initialisation, sets random seeds, and manages
the high‑level workflow by delegating to the specialised classes:
  - DataLoader, MoETransformer, PretrainTrainer, AdaptationTrainer,
    DownstreamEvaluator, InLoopEvaluator
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import yaml
from transformers import AutoTokenizer

# Project imports – these modules must be importable (i.e., the project root
# is on PYTHONPATH).
from data.dataset_loader import DataLoader
from model.moe_transformer import MoETransformer
from trainer.adaptation_trainer import AdaptationTrainer
from trainer.pretrain_trainer import PretrainTrainer
from evaluation.downstream_eval import DownstreamEvaluator
from evaluation.in_loop_eval import InLoopEvaluator
from utils.logging_utils import init_wandb, log_metrics

# ---------------------------------------------------------------------------
# Logging configuration (console only; W&B is initialised later)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("OLMoE.main")


# ===========================================================================
# 1. Command‑line argument parsing
# ===========================================================================
def parse_args() -> argparse.Namespace:
    """Parse CLI arguments and return a populated namespace."""
    parser = argparse.ArgumentParser(
        description="OLMoE-1B-7B Reproduction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["pretrain", "eval", "adapt"],
        help="Pipeline stage to execute.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml).",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a model checkpoint. Required for 'eval' and 'adapt' modes.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for saving checkpoints, logs, and final outputs. "
             "Overrides config['output_dir'] if set.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    parser.add_argument(
        "--eval_type",
        type=str,
        default="olmes",
        choices=["olmes", "instruct"],
        help="Which evaluation benchmark suite to run in 'eval' mode "
             "(default: olmes).",
    )

    parser.add_argument(
        "--evaluate_after_adapt",
        action="store_true",
        help="Run instruct benchmarks immediately after adaptation completes.",
    )

    return parser.parse_args()


# ===========================================================================
# 2. Configuration loading and validation
# ===========================================================================
def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load the YAML configuration from *config_path*.

    Returns the configuration as a nested dictionary.  Raises ``FileNotFoundError``
    if the path does not exist and ``yaml.YAMLError`` on parse errors.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with path.open("r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Configuration file {config_path} is empty.")

    # Validate essential top‑level sections
    required_sections = ["model", "pretraining", "adaptation", "evaluation", "logging", "fsdp"]
    for section in required_sections:
        if section not in config:
            raise KeyError(
                f"Missing required configuration section: '{section}'. "
                f"Please check {config_path}."
            )

    logger.info("Configuration loaded from %s", config_path)
    return config


# ===========================================================================
# 3. Distributed and environment setup
# ===========================================================================
def setup_environment(config: Dict[str, Any], seed: int) -> None:
    """
    Initialise the distributed process group (if applicable) and set global
    random seeds.

    This function should be called *before* any model or data‑loader
    instantiation.  It detects whether the script was launched with
    ``torchrun`` (by checking the ``LOCAL_RANK`` environment variable) and,
    if so, calls ``torch.distributed.init_process_group`` with the appropriate
    backend.

    Args:
        config: Full configuration dictionary.
        seed:   Random seed for PyTorch, NumPy, and Python's ``random`` module.
    """
    # -- Random seeds --------------------------------------------------------
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # -- Distributed initialisation ------------------------------------------
    local_rank = os.environ.get("LOCAL_RANK")
    world_size = os.environ.get("WORLD_SIZE")

    if local_rank is not None and world_size is not None:
        # Launched via torchrun or torch.distributed.launch
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(int(local_rank))
            logger.info(
                "Distributed initialised: rank=%d, local_rank=%d, world_size=%d",
                dist.get_rank(),
                int(local_rank),
                dist.get_world_size(),
            )
    else:
        # Single‑process execution (e.g., eval on a single GPU)
        logger.info("Running in single‑process mode (no distributed env detected).")

    # -- CUDA availability check ---------------------------------------------
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available; running on CPU (training will be slow).")


# ===========================================================================
# 4. High‑level mode runners
# ===========================================================================
def run_pretrain(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """
    Execute the full pretraining pipeline.

    1. Build the tokenised pretraining dataset (with filters).
    2. Instantiate the MoETransformer model (truncated‑normal init).
    3. Wrap with FSDP, create the optimizer + LR scheduler, and start the
       training loop with periodic in‑loop evaluation and checkpointing.
    4. Optionally, run the OLMES benchmark on the final checkpoint.
    """
    logger.info("=== Starting PRETRAINING ===")

    # ---- Tokenizer ---------------------------------------------------------
    tokenizer_path = config["pretraining"].get(
        "tokenizer", "allenai/OLMo-1B-0724-hf"
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # ---- Data --------------------------------------------------------------
    data_loader = DataLoader(config, tokenizer=tokenizer)
    logger.info("DataLoader created; preparing pretraining corpus...")
    # The paper reshuffles before annealing; for reproducibility we call
    # build_tokenized_data with a fixed seed (the trainer will handle
    # the reshuffle during the annealing phase).
    data_loader.build_tokenized_data(shuffle_seed=args.seed)

    # ---- Model -------------------------------------------------------------
    logger.info("Building MoETransformer...")
    model_cfg = config["model"]
    model = MoETransformer(model_cfg)

    # Resume from checkpoint if requested (optional)
    if args.checkpoint is not None:
        logger.info("Loading initial checkpoint from %s", args.checkpoint)
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        del ckpt

    # ---- In‑loop evaluator (will be called by the trainer) -----------------
    in_loop_evaluator = InLoopEvaluator(
        tokenizer=tokenizer,
        tasks=config["evaluation"].get("in_loop_tasks", []),
        eval_config=config["evaluation"],
        max_seq_length=config["pretraining"]["seq_length"],
        seed=args.seed,
    )

    # ---- Trainer -----------------------------------------------------------
    trainer = PretrainTrainer(
        model=model,
        data_loader=data_loader,
        config=config,
    )
    # Optionally patch the evaluator into the trainer (the trainer already
    # instantiates its own InLoopEvaluator; here we replace it with the one
    # we created above to reuse the same task list).
    trainer.in_loop_evaluator = in_loop_evaluator

    # ---- Launch training loop ----------------------------------------------
    trainer.train()

    # ---- Final OLMES evaluation --------------------------------------------
    logger.info("Pretraining finished. Running final OLMES evaluation...")
    # After training, the FSDP‑wrapped model is inside trainer.fsdp_model.
    # For evaluation we use the unwrapped raw model to avoid FSDP overhead.
    final_evaluator = DownstreamEvaluator(
        model=trainer.raw_model,
        tokenizer=tokenizer,
        config=config,
    )
    olmes_scores = final_evaluator.run_olmes()
    logger.info("Final OLMES scores: %s", olmes_scores)

    # Cleanup W&B
    import wandb
    if wandb.run is not None:
        wandb.finish()
    logger.info("PRETRAINING complete.")


def run_eval(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """
    Evaluate a pretrained or instruction‑tuned model.

    For ``--eval_type olmes`` the OLMES standard benchmark is run (5‑shot,
    max‑MCF‑CF).  For ``--eval_type instruct`` the paper's adaptation
    benchmarks are used (MMLU 0‑shot, GSM8k CoT, BBH, HumanEval, etc.).
    """
    logger.info("=== Starting EVALUATION (mode=%s) ===", args.eval_type)

    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for eval mode.")

    # ---- Tokenizer ---------------------------------------------------------
    tokenizer_path = config["pretraining"].get("tokenizer", "allenai/OLMo-1B-0724-hf")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # ---- Model -------------------------------------------------------------
    model_cfg = config["model"]
    model = MoETransformer(model_cfg)
    logger.info("Loading checkpoint from %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    # Handle both FSDP‑wrapped state dicts (prefixed with "_fsdp_wrapped_module.")
    # and plain state dicts.
    state_dict = ckpt["model"]
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("_fsdp_wrapped_module."):
            new_state[k[len("_fsdp_wrapped_module."):]] = v
        else:
            new_state[k] = v
    model.load_state_dict(new_state, strict=True)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # ---- Evaluation --------------------------------------------------------
    if args.eval_type == "olmes":
        evaluator = DownstreamEvaluator(
            model=model,
            tokenizer=tokenizer,
            config=config,
        )
        scores = evaluator.run_olmes()
        logger.info("OLMES benchmark results:\n%s", yaml.dump(scores, default_flow_style=False))
    else:  # instruct
        evaluator = DownstreamEvaluator(
            model=model,
            tokenizer=tokenizer,
            config=config,
        )
        scores = evaluator.run_instruct_eval()
        logger.info("Instruct benchmark results:\n%s", yaml.dump(scores, default_flow_style=False))

    import wandb
    if wandb.run is not None:
        wandb.finish()
    logger.info("EVALUATION complete.")


def run_adapt(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """
    Run adaptation: instruction tuning (SFT) followed by preference tuning (DPO).

    The pipeline:
    1. Load the pretrained *annealed* checkpoint as the base model.
    2. Perform SFT for 2 epochs on the combined instruction datasets.
    3. Perform DPO for 3 epochs using the binarized UltraFeedback dataset.
    4. (Optionally) evaluate the final INSTRUCT model.
    """
    logger.info("=== Starting ADAPTATION ===")

    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for adapt mode (pretrained annealed model).")

    # ---- Tokenizer ---------------------------------------------------------
    tokenizer_path = config["pretraining"].get("tokenizer", "allenai/OLMo-1B-0724-hf")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # ---- Base model --------------------------------------------------------
    model_cfg = config["model"]
    model = MoETransformer(model_cfg)
    logger.info("Loading pretrained checkpoint from %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt["model"]
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("_fsdp_wrapped_module."):
            new_state[k[len("_fsdp_wrapped_module."):]] = v
        else:
            new_state[k] = v
    model.load_state_dict(new_state, strict=True)
    # Note: the model will be moved to the correct device inside the trainer.

    # ---- Data --------------------------------------------------------------
    data_loader = DataLoader(config, tokenizer=tokenizer)

    # ---- Adaptation trainer ------------------------------------------------
    adapt_trainer = AdaptationTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
    )

    # ----- SFT --------------------------------------------------------------
    logger.info("Loading SFT dataset...")
    sft_dataset = data_loader.get_adaptation_dataset("sft")
    logger.info("SFT dataset size: %d samples", len(sft_dataset))
    adapt_trainer.sft_train(sft_dataset)

    # ----- DPO --------------------------------------------------------------
    logger.info("Loading DPO dataset...")
    dpo_dataset = data_loader.get_adaptation_dataset("dpo")
    logger.info("DPO dataset size: %d samples", len(dpo_dataset))
    adapt_trainer.dpo_train(dpo_dataset)

    # ----- Optional post‑adaptation evaluation ------------------------------
    if args.evaluate_after_adapt:
        logger.info("Running post‑adaptation instruct benchmarks...")
        instruct_path = os.path.join(
            adapt_trainer.checkpoint_dir, "instruct_model.pt"
        )
        if not os.path.isfile(instruct_path):
            logger.error("INSTRUCT checkpoint not found at %s", instruct_path)
        else:
            # Load the final instruct model for evaluation
            eval_model = MoETransformer(model_cfg)
            ckpt = torch.load(instruct_path, map_location="cpu")
            state_dict = ckpt["model"]
            new_state = {}
            for k, v in state_dict.items():
                if k.startswith("_fsdp_wrapped_module."):
                    new_state[k[len("_fsdp_wrapped_module."):]] = v
                else:
                    new_state[k] = v
            eval_model.load_state_dict(new_state, strict=True)
            eval_model.to("cuda" if torch.cuda.is_available() else "cpu")
            eval_model.eval()

            evaluator = DownstreamEvaluator(
                model=eval_model,
                tokenizer=tokenizer,
                config=config,
            )
            scores = evaluator.run_instruct_eval()
            logger.info("Post‑adaptation instruct scores:\n%s",
                        yaml.dump(scores, default_flow_style=False))

    import wandb
    if wandb.run is not None:
        wandb.finish()
    logger.info("ADAPTATION complete.")


# ===========================================================================
# 5. Main entry point
# ===========================================================================
def main() -> None:
    """
    Parse arguments, load configuration, setup the environment, and dispatch
    to the requested pipeline stage.
    """
    args = parse_args()

    # ---- Load configuration ------------------------------------------------
    config = load_config(args.config)

    # ---- Override output directory if supplied via CLI ----------------------
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    else:
        # Provide a default output directory if none is set in config
        config.setdefault("output_dir", os.path.join(os.getcwd(), "output"))

    # Ensure the output directory exists
    os.makedirs(config["output_dir"], exist_ok=True)

    # ---- Set random seeds & distributed environment ------------------------
    setup_environment(config, args.seed)

    # ---- Initialise Weights & Biases ---------------------------------------
    init_wandb(config)

    # ---- Dispatch to the selected mode -------------------------------------
    try:
        if args.mode == "pretrain":
            run_pretrain(config, args)
        elif args.mode == "eval":
            run_eval(config, args)
        elif args.mode == "adapt":
            run_adapt(config, args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Exiting gracefully.")
        sys.exit(130)
    except Exception as e:
        logger.exception("Fatal error in '%s' mode: %s", args.mode, str(e))
        sys.exit(1)


# ===========================================================================
# 6. Script entry point
# ===========================================================================
if __name__ == "__main__":
    main()


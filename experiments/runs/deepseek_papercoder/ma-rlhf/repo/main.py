# main.py
"""
Main entry point for MA‑RLHF reproduction pipeline.

Usage:
    python main.py --config config.yaml --experiment tldr_2b [--output_dir ./outputs] [--skip_sft] [--skip_rm] [--skip_ppo] [--eval_only] [--ckpt_path <path>] [--compare_ckpt <path>] [--seed 42]

The pipeline:
  1. SFT: supervised fine‑tuning on human demonstrations.
  2. RM: reward model training on preference pairs (skipped for code generation).
  3. PPO: policy optimisation with or without macro‑actions.
  4. Evaluation: reward model scoring, GPT‑4 pairwise win rate, pass@k.

All hyperparameters are read from the YAML configuration file, following the paper's
Table 5 and Appendix B.2 / B.5.
"""

import argparse
import logging
import os
import random
import sys
import yaml
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

# Local modules (assumed to be in the same directory)
from data_utils import DataProcessor
from sft_trainer import SFTTrainer
from rm_trainer import RMTrainer, RewardModel
from ppo_trainer import PPOTrainer, CriticModel
from evaluate import Evaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration utilities
# ---------------------------------------------------------------------------

def load_and_merge_config(config_path: str, experiment_name: str) -> Dict[str, Any]:
    """
    Load the YAML file and merge the experiment-specific block with defaults.

    The `defaults` section provides common values; the chosen experiment overrides
    them. Nested blocks (sft, rm, ppo, macro) are taken wholly from the experiment
    if present, otherwise from `defaults`. If the experiment defines a `macro` block,
    it replaces the default macro configuration entirely.
    """
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    defaults = raw.get("defaults", {})
    experiments = raw.get("experiments", {})
    if experiment_name not in experiments:
        raise KeyError(f"Experiment '{experiment_name}' not found in config file.")

    exp = experiments[experiment_name]

    # Start with defaults
    cfg = dict(defaults)  # shallow copy, top-level keys

    # Override top-level keys (data_splits, tokenizer_kwargs)
    for key in ["data_splits", "tokenizer_kwargs"]:
        if key in exp:
            cfg[key] = exp[key]

    # Model and dataset info always from experiment
    cfg["model_name"] = exp["model_name"]
    cfg["dataset_name"] = exp["dataset_name"]
    cfg["max_prompt_length"] = exp.get("max_prompt_length", 512)
    cfg["max_response_length"] = exp.get("max_response_length", 512)

    # Nested blocks: sft, rm, ppo, macro
    for block in ["sft", "rm", "ppo", "macro"]:
        if block in exp:
            cfg[block] = exp[block]
        elif block in defaults:
            cfg[block] = defaults[block]
        else:
            cfg[block] = {}  # allow empty for optional blocks like macro

    # If the experiment explicitly sets rm: null, keep it
    if "rm" in exp and exp["rm"] is None:
        cfg["rm"] = None

    return cfg


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class Main:
    """
    Orchestrates the SFT → RM → PPO → Evaluation pipeline.

    Args:
        config: Merged configuration dictionary (from load_and_merge_config).
        args: Parsed command-line arguments (Namespace).
    """

    def __init__(self, config: Dict[str, Any], args: argparse.Namespace) -> None:
        self.config = config
        self.args = args
        self.experiment = args.experiment
        self.output_root = args.output_dir
        self.exp_dir = os.path.join(self.output_root, self.experiment)

        # Create directory structure
        self.sft_dir = os.path.join(self.exp_dir, "sft_model")
        self.rm_dir = os.path.join(self.exp_dir, "rm_model")
        self.ppo_dir = os.path.join(self.exp_dir, "ppo_model")
        self.log_dir = os.path.join(self.exp_dir, "logs")
        for d in [self.exp_dir, self.sft_dir, self.rm_dir, self.ppo_dir, self.log_dir]:
            os.makedirs(d, exist_ok=True)

        # Setup logging to file
        fh = logging.FileHandler(os.path.join(self.log_dir, "pipeline.log"))
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logging.getLogger().addHandler(fh)

        logger.info(f"Experiment: {self.experiment}")
        logger.info(f"Output directory: {self.exp_dir}")

        set_seed(args.seed)

        # Tokenizer
        tokenizer_kwargs = config.get("tokenizer_kwargs", {})
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model_name"],
            **tokenizer_kwargs,
        )
        # Ensure pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            logger.info("Set tokenizer pad_token to eos_token.")

        # Data processor (shared across stages)
        self.data_processor = DataProcessor(config, self.tokenizer)

        # Paths for checkpoints (updated after each stage)
        self.sft_path = None
        self.rm_path = None
        self.ppo_path = None

        # Save effective config for reproducibility
        import json
        with open(os.path.join(self.log_dir, "effective_config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def run_sft(self) -> None:
        """Supervised Fine‑Tuning stage."""
        if self.args.skip_sft:
            logger.info("SFT skipped by user.")
            return
        logger.info("=== Starting SFT ===")

        sft_cfg = self.config.get("sft", {})
        if not sft_cfg:
            raise ValueError("SFT configuration missing.")

        trainer = SFTTrainer(
            model_path=self.config["model_name"],
            tokenizer=self.tokenizer,
            config=sft_cfg,
        )
        train_ds = self.data_processor.load_sft_data(split="train")
        trainer.train(train_ds)
        trainer.save_checkpoint(self.sft_dir)
        self.sft_path = self.sft_dir
        logger.info(f"SFT checkpoint saved to {self.sft_path}")

    def run_rm(self) -> None:
        """Reward Model training stage (skipped if rm is null or flag set)."""
        if self.args.skip_rm or self.config.get("rm") is None:
            logger.info("RM stage skipped (either flag or config sets rm to null).")
            return
        logger.info("=== Starting RM ===")
        if self.sft_path is None:
            raise RuntimeError("SFT checkpoint not available; run SFT first.")

        rm_cfg = self.config.get("rm", {})
        trainer = RMTrainer(
            sft_checkpoint=self.sft_path,
            tokenizer=self.tokenizer,
            config=self.config,  # full config contains max lengths etc.
        )
        train_ds = self.data_processor.load_rm_data(split="train")
        trainer.train(train_ds)
        trainer.save_checkpoint(self.rm_dir)
        self.rm_path = self.rm_dir
        logger.info(f"RM checkpoint saved to {self.rm_path}")

    def run_ppo(self) -> None:
        """PPO stage (vanilla or macro‑action)."""
        if self.args.skip_ppo:
            logger.info("PPO skipped by user.")
            return
        logger.info("=== Starting PPO ===")
        if self.sft_path is None:
            raise RuntimeError("SFT checkpoint not available; run SFT first.")

        # Determine base paths for critic and reward
        actor_path = self.sft_path
        critic_base_path = self.rm_path if self.rm_path else self.sft_path
        reward_path = self.rm_path if self.rm_path else None

        # Load reward model (if exists)
        reward_model = None
        if reward_path is not None:
            try:
                # The reward model was saved as a custom RewardModel.
                # We load the transformer and the head separately.
                reward_transformer = AutoModel.from_pretrained(reward_path)
                head_path = os.path.join(reward_path, "v_head.pt")
                v_head = torch.nn.Linear(
                    reward_transformer.config.hidden_size, 1, bias=False
                )
                v_head.load_state_dict(torch.load(head_path, map_location="cpu"))
                # Create a RewardModel instance; it will re‑load the transformer,
                # but we can replace its internal transformer to avoid duplication.
                reward_model = RewardModel(reward_path, self.tokenizer)
                reward_model.transformer = reward_transformer
                reward_model.v_head = v_head
                logger.info("Loaded reward model from checkpoint.")
            except Exception as e:
                logger.error(f"Failed to load reward model: {e}")
                raise
        else:
            logger.info("No reward model path; using compiler reward (code generation).")

        # Load critic
        critic_base = AutoModel.from_pretrained(critic_base_path)
        critic = CriticModel(critic_base, self.tokenizer)
        # The critic head should be randomly init (as per paper). We'll move it to appropriate device later.

        # Load actor (policy)
        actor = AutoModelForCausalLM.from_pretrained(actor_path)

        # Load reference model (frozen SFT)
        ref_model = AutoModelForCausalLM.from_pretrained(self.sft_path)

        # Macro configuration (can be overridden by command line? Not needed)
        macro_cfg = self.config.get("macro", {"enabled": False})
        if not macro_cfg.get("enabled", False):
            logger.info("Macro actions disabled; running vanilla PPO.")

        # Instantiate PPO trainer
        models = {
            "actor": actor,
            "critic": critic,
            "reward": reward_model,
            "ref": ref_model,
        }

        ppo_config = self.config  # full config includes ppo and macro keys
        ppo_trainer = PPOTrainer(
            models=models,
            tokenizer=self.tokenizer,
            config=ppo_config,
            macro_config=macro_cfg,
        )

        # Load PPO prompts dataset
        ppo_ds = self.data_processor.load_ppo_data(split="train")
        ppo_trainer.train(ppo_ds)
        ppo_trainer.save_checkpoint(self.ppo_dir)
        self.ppo_path = self.ppo_dir
        logger.info(f"PPO checkpoint saved to {self.ppo_path}")

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def _load_eval_prompts(self) -> List[str]:
        """Load raw prompt strings for evaluation."""
        dataset_name = self.config["dataset_name"]
        try:
            if dataset_name == "tldr":
                ds = load_dataset("openai/summarize_from_feedback", "comparisons", split="validation")
                prompts = [sample["info"]["post"] for sample in ds if "info" in sample and "post" in sample["info"]]
                # Use first 2000 as in paper
                prompts = prompts[:2000]
            elif dataset_name == "hhrlhf":
                ds = load_dataset("Anthropic/hh-rlhf", "helpful-base", split="test")
                prompts = []
                for sample in ds:
                    chosen_text = sample["chosen"]
                    # Extract prompt: all text before the last "Assistant:"
                    last_assistant = chosen_text.rfind("Assistant:")
                    if last_assistant != -1:
                        prompt = chosen_text[:last_assistant].strip()
                        prompts.append(prompt)
                prompts = prompts[:2000]
            elif dataset_name == "webgpt":
                ds = load_dataset("openai/webgpt_comparisons", split="train")
                # Use a 5% held‑out subset as validation (paper does the same)
                split_ds = ds.train_test_split(test_size=0.05, seed=42)["test"]
                prompts = [sample["question"] for sample in split_ds if "question" in sample]
            elif dataset_name == "apps":
                # APPS evaluation is handled separately via pass_k_eval.
                return None
            else:
                raise ValueError(f"Unknown dataset for evaluation: {dataset_name}")
            logger.info(f"Loaded {len(prompts)} prompts for evaluation.")
            return prompts
        except Exception as e:
            logger.error(f"Failed to load evaluation prompts: {e}")
            raise

    def _generate_responses(
        self,
        actor_model: torch.nn.Module,
        prompts: List[str],
        sample_temperature: float,
    ) -> List[str]:
        """Helper to generate responses in batches using the provided actor."""
        responses = []
        actor_model.eval()
        device = next(actor_model.parameters()).device
        batch_size = 8

        # Tokenizer settings
        max_prompt_length = self.config.get("max_prompt_length", 512)
        max_response_length = self.config.get("max_response_length", 512)

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            tokenized = self.tokenizer(
                batch_prompts,
                max_length=max_prompt_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            prompt_ids = tokenized["input_ids"].to(device)
            prompt_mask = tokenized["attention_mask"].to(device)

            with torch.no_grad():
                generated = actor_model.generate(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    max_new_tokens=max_response_length,
                    do_sample=True,
                    temperature=sample_temperature,
                    top_p=1.0,
                    top_k=50,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    return_dict_in_generate=True,
                )

            prompt_len = prompt_ids.shape[1]
            for j in range(generated.sequences.size(0)):
                resp_tokens = generated.sequences[j, prompt_len:]
                text = self.tokenizer.decode(resp_tokens, skip_special_tokens=True)
                responses.append(text)
        return responses

    def evaluate(self, checkpoint_path: Optional[str] = None, compare_ckpt: Optional[str] = None) -> None:
        """
        Run evaluation on the trained policy.

        If `compare_ckpt` is provided, a pairwise GPT‑4 evaluation is performed
        between the primary model and the comparison model.

        For text tasks: computes RM score and optionally GPT‑4 win rates.
        For code generation: computes pass@1 and pass@5 on APPS test set.
        """
        logger.info("=== Starting Evaluation ===")
        if checkpoint_path is None:
            checkpoint_path = self.ppo_path
        if not checkpoint_path:
            raise RuntimeError(
                "No PPO checkpoint available. Train a policy or provide --ckpt_path."
            )

        dataset_name = self.config["dataset_name"]

        # If code generation, use specialised evaluation
        if dataset_name == "apps":
            self._evaluate_code(checkpoint_path)
            return

        # Text tasks: load prompts and reward model
        prompts = self._load_eval_prompts()
        if prompts is None:
            logger.warning("No prompts loaded; skipping evaluation.")
            return

        reward_path = self.rm_path if self.rm_path else None
        # Instantiate Evaluator (needs actor and reward paths)
        evaluator = Evaluator(
            actor_path=checkpoint_path,
            reward_path=reward_path,
            tokenizer=self.tokenizer,
            config=self.config,
        )

        # RM scoring
        if evaluator.reward_model is not None:
            rm_result = evaluator.compute_rm_score(prompts)
            logger.info(
                f"RM score: mean = {rm_result['mean']:.4f} ± {rm_result['std']:.4f}"
            )
            # Optionally save distribution
            with open(os.path.join(self.log_dir, "rm_scores.json"), "w") as f:
                import json
                json.dump(rm_result, f, indent=2)

        # GPT‑4 pairwise evaluation (if a comparison checkpoint is provided)
        if compare_ckpt:
            logger.info("Running GPT‑4 pairwise evaluation...")
            # Load primary actor (already loaded by evaluator, but we need it for generation)
            primary_actor = evaluator.actor  # already loaded, can reuse
            # Load comparison actor
            logger.info(f"Loading comparison actor from {compare_ckpt}")
            compar_actor = AutoModelForCausalLM.from_pretrained(
                compare_ckpt,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            temperature = self.config.get("ppo", {}).get("temperature", 0.8)
            # Generate responses from both models
            logger.info("Generating responses from primary model...")
            responses_primary = self._generate_responses(primary_actor, prompts, temperature)
            logger.info("Generating responses from comparison model...")
            responses_compar = self._generate_responses(compar_actor, prompts, temperature)

            # Randomise order for each pair (to avoid position bias)
            # Evaluator expects responses_a and responses_b; we'll assign randomly.
            responses_a = []
            responses_b = []
            for p, c in zip(responses_primary, responses_compar):
                if random.random() < 0.5:
                    responses_a.append(p)
                    responses_b.append(c)
                else:
                    responses_a.append(c)
                    responses_b.append(p)

            gpt_result = evaluator.gpt4_eval(prompts, responses_a, responses_b)
            logger.info(
                f"GPT‑4 win rates: A={gpt_result['win_A']}, B={gpt_result['win_B']}, tie={gpt_result['tie']}"
            )
            with open(os.path.join(self.log_dir, "gpt4_results.json"), "w") as f:
                json.dump(gpt_result, f, indent=2)

        logger.info("Evaluation completed.")

    def _evaluate_code(self, checkpoint_path: str) -> None:
        """Specialised evaluation for APPS code generation (pass@k)."""
        logger.info("Evaluating code generation on APPS test set.")
        evaluator = Evaluator(
            actor_path=checkpoint_path,
            reward_path=None,  # no reward model
            tokenizer=self.tokenizer,
            config=self.config,
        )
        # Load APPS test set
        test_ds = load_dataset("codeparrot/apps", "all", split="test")
        # compute pass@1
        pass1 = evaluator.pass_k_eval(test_ds, k=1)
        pass5 = evaluator.pass_k_eval(test_ds, k=5)
        logger.info(f"Pass@1: {pass1['pass@1']:.4f}, Pass@5: {pass5['pass@5']:.4f}")
        with open(os.path.join(self.log_dir, "pass_at_k.json"), "w") as f:
            json.dump({**pass1, **pass5}, f, indent=2)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MA‑RLHF Reproduction Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file (config.yaml).",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Experiment key in the config file (e.g., tldr_2b).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Root directory for outputs (default: ./outputs).",
    )
    parser.add_argument("--skip_sft", action="store_true", help="Skip SFT stage.")
    parser.add_argument("--skip_rm", action="store_true", help="Skip RM stage.")
    parser.add_argument("--skip_ppo", action="store_true", help="Skip PPO stage.")
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only run evaluation (requires a trained model).",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to a specific PPO checkpoint for evaluation. If not given, uses the most recent.",
    )
    parser.add_argument(
        "--compare_ckpt",
        type=str,
        default=None,
        help="Path to a second (baseline) PPO checkpoint for pairwise GPT‑4 evaluation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Load and merge config
    try:
        cfg = load_and_merge_config(args.config, args.experiment)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    pipeline = Main(cfg, args)

    if args.eval_only:
        pipeline.evaluate(checkpoint_path=args.ckpt_path, compare_ckpt=args.compare_ckpt)
    else:
        # Sequential training stages
        pipeline.run_sft()
        pipeline.run_rm()
        pipeline.run_ppo()
        # After training, evaluate using the final checkpoint
        pipeline.evaluate(compare_ckpt=args.compare_ckpt)

    logger.info("Pipeline finished.")


if __name__ == "__main__":
    main()

```python
## main.py
"""Main entry point for SCoRe: Self-Correction via Reinforcement Learning.

This module orchestrates the full SCoRe training and evaluation pipeline,
exposing a CLI with four subcommands:
    train    — Full two-stage SCoRe RL training + evaluation.
    eval     — Load a checkpoint and run the full evaluation suite.
    ablation — Reproduce Table 4 ablation studies.
    baseline — Run STaR and Pair-SFT SFT baselines (Table 1/2).

All hyperparameters are sourced from config.yaml via Config.from_dict().
No values are hardcoded in this file.

Usage:
    python main.py --config config.yaml train
    python main.py --config config.yaml eval --checkpoint outputs/stage2_best
    python main.py --config config.yaml ablation --type all
    python main.py --config config.yaml baseline --method all
"""

import argparse
import copy
import logging
import os
import random
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

from config import Config
from data.dataset_loader import DatasetLoader
from data.prompt_templates import PromptTemplates
from evaluation.evaluator import Evaluator
from models.model_wrapper import ModelWrapper
from rewards import RewardFunction
from training.reinforce_trainer import REINFORCETrainer
from training.rollout_buffer import RolloutBuffer
from training.score_trainer import SCoReTrainer
from training.sft_baseline_trainer import SFTBaselineTrainer
from utils.checkpoint_utils import CheckpointUtils
from utils.logging_utils import LoggingUtils

logger = logging.getLogger(__name__)


class Main:
    """Top-level orchestrator for the SCoRe training and evaluation pipeline.

    Reads the YAML config, wires together all subsystems, and dispatches
    to the appropriate experiment method based on the CLI subcommand.

    Attributes:
        config: The global Config instance parsed from config.yaml.
        logger: LoggingUtils instance for wandb + Python logging.
        checkpoint_utils: CheckpointUtils instance for checkpoint management.
    """

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialize Main by loading the YAML config and setting up logging.

        Args:
            config_path: Path to the YAML configuration file. Defaults to
                "config.yaml" in the current working directory.

        Raises:
            FileNotFoundError: If config_path does not exist.
            ValueError: If the config contains invalid values (propagated
                from Config.from_dict()).
        """
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"Config file not found: '{config_path}'. "
                "Ensure the path is correct and the file exists."
            )

        # ------------------------------------------------------------------
        # Step 1: Load and parse the YAML config file.
        # ------------------------------------------------------------------
        with open(config_path, "r", encoding="utf-8") as fh:
            raw_config: Dict[str, Any] = yaml.safe_load(fh)

        if raw_config is None:
            raise ValueError(
                f"Config file '{config_path}' is empty or contains only "
                "null values. Provide a valid YAML configuration."
            )

        self.config: Config = Config.from_dict(raw_config)

        # ------------------------------------------------------------------
        # Step 2: Initialize logging (wandb + Python logging).
        # Must be done before _setup_environment so all setup steps are logged.
        # ------------------------------------------------------------------
        self.logger: LoggingUtils = LoggingUtils(self.config)

        # ------------------------------------------------------------------
        # Step 3: Initialize checkpoint utilities.
        # ------------------------------------------------------------------
        self.checkpoint_utils: CheckpointUtils = CheckpointUtils()

        # ------------------------------------------------------------------
        # Step 4: Set up reproducibility and distributed training environment.
        # ------------------------------------------------------------------
        self._setup_environment()

        logger.info(
            "Main initialized. task='%s', model='%s', output_dir='%s'.",
            self.config.task,
            self.config.model_name,
            self.config.output_dir,
        )

    # -------------------------------------------------------------------------
    # Environment setup
    # -------------------------------------------------------------------------

    def _setup_environment(self) -> None:
        """Set random seeds, configure distributed training, and create output dirs.

        Implements reproducibility setup:
            - Python random, numpy, torch, and CUDA seeds from config.seed.
            - cuDNN deterministic mode.
            - Output directory creation.
            - DeepSpeed config path validation (actual init happens in ModelWrapper).

        All values sourced from config.yaml:
            experiment.seed = 42
            experiment.output_dir = "outputs/"
            distributed.deepspeed_config = "configs/ds_zero2_config.json"
        """
        seed: int = self.config.seed

        # ------------------------------------------------------------------
        # Reproducibility: set all random seeds.
        # ------------------------------------------------------------------
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # cuDNN deterministic mode for reproducibility.
        # Note: may reduce performance slightly.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        logger.info(
            "_setup_environment: Random seeds set to %d. "
            "cuDNN deterministic=True, benchmark=False.",
            seed,
        )

        # ------------------------------------------------------------------
        # Create output directory structure.
        # ------------------------------------------------------------------
        output_dir: str = self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(
            "_setup_environment: Output directory created/verified: '%s'.",
            output_dir,
        )

        # ------------------------------------------------------------------
        # Validate DeepSpeed config path (actual init happens in ModelWrapper).
        # ------------------------------------------------------------------
        deepspeed_config: str = self.config.deepspeed_config
        if deepspeed_config and deepspeed_config.strip():
            if not os.path.isfile(deepspeed_config):
                logger.warning(
                    "_setup_environment: DeepSpeed config file '%s' not found. "
                    "Training will proceed without DeepSpeed. "
                    "To enable DeepSpeed, create the config file at the "
                    "specified path.",
                    deepspeed_config,
                )
            else:
                logger.info(
                    "_setup_environment: DeepSpeed config found at '%s'.",
                    deepspeed_config,
                )

        # ------------------------------------------------------------------
        # Set Python logging level from config.
        # ------------------------------------------------------------------
        numeric_level: int = getattr(
            logging, self.config.log_level.upper(), logging.INFO
        )
        logging.getLogger().setLevel(numeric_level)
        logger.info(
            "_setup_environment: Python logging level set to '%s'.",
            self.config.log_level,
        )

    # -------------------------------------------------------------------------
    # Main experiment: SCoRe two-stage RL training
    # -------------------------------------------------------------------------

    def run_score(self) -> None:
        """Execute the full SCoRe two-stage RL training pipeline.

        Implements the complete SCoRe experiment:
            1. Load train/test data.
            2. Initialize reward function.
            3. Initialize policy and reference models.
            4. Run SCoReTrainer (Stage I + Stage II).
            5. Evaluate the trained model on the test set.
            6. Optionally run inference-compute scaling and multi-attempt eval.
            7. Log all results and close the wandb run.

        All hyperparameters sourced from config.yaml:
            training.math.stage1_steps = 1500
            training.math.stage2_steps = 1500
            score.math.alpha = 10.0
            score.math.beta1 = 0.01
            score.math.beta2 = 0.1
            model.eval_temperature = 0.0
        """
        logger.info("run_score: Starting SCoRe two-stage RL training.")

        try:
            # ------------------------------------------------------------------
            # Step 1: Load data.
            # ------------------------------------------------------------------
            loader: DatasetLoader = DatasetLoader(self.config)
            train_data: List[Dict[str, Any]] = loader.load_train_data()
            test_data: List[Dict[str, Any]] = loader.load_test_data()

            logger.info(
                "run_score: Loaded %d training examples and %d test examples.",
                len(train_data),
                len(test_data),
            )

            # ------------------------------------------------------------------
            # Step 2: Initialize reward function.
            # ------------------------------------------------------------------
            reward_fn: RewardFunction = RewardFunction(task=self.config.task)

            # ------------------------------------------------------------------
            # Step 3: Run SCoRe training.
            # SCoReTrainer internally creates ModelWrapper instances,
            # RolloutBuffer, SCoReStage1Trainer, and SCoReStage2Trainer.
            # ------------------------------------------------------------------
            score_trainer: SCoReTrainer = SCoReTrainer(self.config)
            trained_model: ModelWrapper = score_trainer.train(train_data)

            logger.info("run_score: SCoRe training complete.")

            # ------------------------------------------------------------------
            # Step 4: Main evaluation (greedy decoding, temperature=0.0).
            # Section 6: "greedy decoding (i.e. temperature 0)"
            # ------------------------------------------------------------------
            prompt_templates: PromptTemplates = PromptTemplates()
            evaluator: Evaluator = Evaluator(
                model=trained_model,
                reward_fn=reward_fn,
                prompt_templates=prompt_templates,
                config=self.config,
            )

            eval_results: Dict[str, Any] = evaluator.evaluate(
                test_data=test_data,
                temperature=self.config.eval_temperature,
            )

            logger.info(
                "run_score: Evaluation results — "
                "acc@t1=%.3f, acc@t2=%.3f, delta=%.3f, "
                "i2c=%.3f, c2i=%.3f.",
                eval_results.get("accuracy_t1", 0.0),
                eval_results.get("accuracy_t2", 0.0),
                eval_results.get("delta_t1_t2", 0.0),
                eval_results.get("i2c_rate", 0.0),
                eval_results.get("c2i_rate", 0.0),
            )

            # Log evaluation results to wandb
            loggable_eval: Dict[str, Any] = {
                k: v
                for k, v in eval_results.items()
                if isinstance(v, (int, float, bool))
            }
            self.logger.log_metrics(loggable_eval, step=self.config.stage2_steps)

            # ------------------------------------------------------------------
            # Step 5: Save evaluation results to disk.
            # ------------------------------------------------------------------
            self._save_results_json(eval_results, filename="score_eval_results.json")

            # ------------------------------------------------------------------
            # Step 6: Optional inference-compute scaling (Section 6.2, Figure 1).
            # config.yaml: evaluation.inference_scaling.enabled = false
            # ------------------------------------------------------------------
            if self.config.inference_scaling_enabled:
                logger.info(
                    "run_score: Running inference-compute scaling experiments "
                    "(Section 6.2, Figure 1 right)."
                )
                scaling_results: Dict[str, Any] = {}
                for k_val in self.config.inference_scaling_k_values:
                    k_result: Dict[str, Any] = evaluator.evaluate_inference_scaling(
                        test_data=test_data,
                        k=k_val,
                    )
                    scaling_results[f"k_{k_val}"] = k_result
                    self.logger.log_metrics(
                        {
                            f"scaling/k{k_val}_parallel_acc": k_result.get(
                                "parallel_accuracy", 0.0
                            ),
                            f"scaling/k{k_val}_sequential_acc": k_result.get(
                                "sequential_accuracy", 0.0
                            ),
                        },
                        step=self.config.stage2_steps,
                    )
                    logger.info(
                        "run_score: Scaling k=%d — parallel=%.3f, sequential=%.3f.",
                        k_val,
                        k_result.get("parallel_accuracy", 0.0),
                        k_result.get("sequential_accuracy", 0.0),
                    )
                self._save_results_json(
                    scaling_results, filename="score_scaling_results.json"
                )

            # ------------------------------------------------------------------
            # Step 7: Optional multi-attempt evaluation (Appendix A.1, Figure 8).
            # config.yaml: evaluation.multi_attempt.enabled = false
            # ------------------------------------------------------------------
            if self.config.multi_attempt_enabled:
                logger.info(
                    "run_score: Running multi-attempt evaluation "
                    "(Appendix A.1, Figure 8). num_attempts=%d.",
                    self.config.multi_attempt_num_attempts,
                )
                multi_results: Dict[str, Any] = evaluator.evaluate_multi_attempt(
                    test_data=test_data,
                    num_attempts=self.config.multi_attempt_num_attempts,
                )
                per_turn_acc: List[float] = multi_results.get(
                    "per_turn_accuracy", []
                )
                for turn_idx, turn_acc in enumerate(per_turn_acc):
                    self.logger.log_metrics(
                        {f"multi_attempt/turn_{turn_idx + 1}_accuracy": turn_acc},
                        step=turn_idx,
                    )
                logger.info(
                    "run_score: Multi-attempt per-turn accuracies: %s.",
                    [f"{a:.3f}" for a in per_turn_acc],
                )
                self._save_results_json(
                    multi_results, filename="score_multi_attempt_results.json"
                )

            # ------------------------------------------------------------------
            # Step 8: MBPP-R offline repair evaluation (code task only).
            # Table 3: SCoRe achieves 60.6% on MBPP-R.
            # ------------------------------------------------------------------
            if self.config.task == "code":
                mbpp_r_data: List[Dict[str, Any]] = loader.load_mbpp_r_data()
                if mbpp_r_data:
                    mbpp_r_accuracy: float = evaluator.evaluate_mbpp_r(mbpp_r_data)
                    self.logger.log_metrics(
                        {"mbpp_r_accuracy": mbpp_r_accuracy},
                        step=self.config.stage2_steps,
                    )
                    logger.info(
                        "run_score: MBPP-R offline repair accuracy = %.3f (%.1f%%).",
                        mbpp_r_accuracy,
                        mbpp_r_accuracy * 100.0,
                    )

        except Exception as exc:
            logger.error(
                "run_score: Training/evaluation failed with exception: %s.",
                exc,
                exc_info=True,
            )
            raise
        finally:
            self.logger.finish()

    # -------------------------------------------------------------------------
    # SFT baselines: STaR and Pair-SFT
    # -------------------------------------------------------------------------

    def run_sft_baselines(self, method: str = "all") -> None:
        """Run STaR and Pair-SFT SFT baselines for comparison against SCoRe.

        Reproduces Table 1 (Section 4) and Table 2 (Section 6) baselines.
        Each baseline starts from the same base model for fair comparison.

        Args:
            method: Which baseline(s) to run. One of:
                'star'     — STaR only (Zelikman et al., 2022).
                'pair_sft' — Pair-SFT only (Welleck et al., 2023 variant).
                'all'      — Both baselines + base model evaluation.

        Config values used:
            sft_baselines.star.num_iterations = 3
            sft_baselines.star.include_correct_pairs = false
            sft_baselines.pair_sft.include_correct_pairs = false
        """
        logger.info(
            "run_sft_baselines: Starting SFT baselines. method='%s'.", method
        )

        try:
            # ------------------------------------------------------------------
            # Step 1: Load data and reward function.
            # ------------------------------------------------------------------
            loader: DatasetLoader = DatasetLoader(self.config)
            train_data: List[Dict[str, Any]] = loader.load_train_data()
            test_data: List[Dict[str, Any]] = loader.load_test_data()
            reward_fn: RewardFunction = RewardFunction(task=self.config.task)
            prompt_templates: PromptTemplates = PromptTemplates()

            logger.info(
                "run_sft_baselines: Loaded %d train, %d test examples.",
                len(train_data),
                len(test_data),
            )

            all_results: Dict[str, Dict[str, Any]] = {}

            # ------------------------------------------------------------------
            # Step 2: Base model evaluation (no fine-tuning).
            # This gives the "Base model" row in Table 1/2.
            # ------------------------------------------------------------------
            logger.info(
                "run_sft_baselines: Evaluating base model (no fine-tuning)."
            )
            base_model: ModelWrapper = ModelWrapper(self.config, freeze=False)
            base_evaluator: Evaluator = Evaluator(
                model=base_model,
                reward_fn=reward_fn,
                prompt_templates=prompt_templates,
                config=self.config,
            )
            base_results: Dict[str, Any] = base_evaluator.evaluate(
                test_data=test_data,
                temperature=self.config.eval_temperature,
            )
            all_results["base_model"] = base_results
            loggable_base: Dict[str, Any] = {
                f"base_model/{k}": v
                for k, v in base_results.items()
                if isinstance(v, (int, float, bool))
            }
            self.logger.log_metrics(loggable_base, step=0)
            logger.info(
                "run_sft_baselines: Base model — "
                "acc@t1=%.3f, acc@t2=%.3f, delta=%.3f.",
                base_results.get("accuracy_t1", 0.0),
                base_results.get("accuracy_t2", 0.0),
                base_results.get("delta_t1_t2", 0.0),
            )
            # Free base model memory before training baselines
            del base_model

            # ------------------------------------------------------------------
            # Step 3: STaR baseline.
            # Section 4: "We run 3 iterations for STaR following the protocol
            # in Singh et al. (2024)."
            # ------------------------------------------------------------------
            if method in ("star", "all"):
                logger.info(
                    "run_sft_baselines: Running STaR baseline "
                    "(num_iterations=%d, include_correct_pairs=%s).",
                    self.config.star_num_iterations,
                    self.config.star_include_correct_pairs,
                )

                star_base_model: ModelWrapper = ModelWrapper(
                    self.config, freeze=False
                )
                star_trainer: SFTBaselineTrainer = SFTBaselineTrainer(
                    policy_model=star_base_model,
                    config=self.config,
                    method="star",
                )
                star_dataset: List[Dict[str, Any]] = (
                    star_trainer.build_star_dataset(
                        base_model=star_base_model,
                        train_data=train_data,
                        reward_fn=reward_fn,
                        num_iterations=self.config.star_num_iterations,
                    )
                )

                if star_dataset:
                    star_model: ModelWrapper = star_trainer.train_sft(star_dataset)
                    star_evaluator: Evaluator = Evaluator(
                        model=star_model,
                        reward_fn=reward_fn,
                        prompt_templates=prompt_templates,
                        config=self.config,
                    )
                    star_results: Dict[str, Any] = star_evaluator.evaluate(
                        test_data=test_data,
                        temperature=self.config.eval_temperature,
                    )
                    all_results["star"] = star_results
                    loggable_star: Dict[str, Any] = {
                        f"star/{k}": v
                        for k, v in star_results.items()
                        if isinstance(v, (int, float, bool))
                    }
                    self.logger.log_metrics(loggable_star, step=0)
                    logger.info(
                        "run_sft_baselines: STaR — "
                        "acc@t1=%.3f, acc@t2=%.3f, delta=%.3f.",
                        star_results.get("accuracy_t1", 0.0),
                        star_results.get("accuracy_t2", 0.0),
                        star_results.get("delta_t1_t2", 0.0),
                    )
                    del star_model
                else:
                    logger.warning(
                        "run_sft_baselines: STaR dataset is empty. "
                        "No i→c transitions found. Skipping STaR evaluation."
                    )

                del star_base_model

            # ------------------------------------------------------------------
            # Step 4: Pair-SFT baseline.
            # Section 4: "only one iteration for Pair-SFT, following the
            # protocol in Welleck et al. (2023)."
            # ------------------------------------------------------------------
            if method in ("pair_sft", "all"):
                logger.info(
                    "run_sft_baselines: Running Pair-SFT baseline "
                    "(include_correct_pairs=%s).",
                    self.config.pair_sft_include_correct_pairs,
                )

                pair_base_model: ModelWrapper = ModelWrapper(
                    self.config, freeze=False
                )
                pair_trainer: SFTBaselineTrainer = SFTBaselineTrainer(
                    policy_model=pair_base_model,
                    config=self.config,
                    method="pair_sft",
                )
                pair_dataset: List[Dict[str, Any]] = (
                    pair_trainer.build_pair_sft_dataset(
                        base_model=pair_base_model,
                        train_data=train_data,
                        reward_fn=reward_fn,
                        include_correct_pairs=self.config.pair_sft_include_correct_pairs,
                    )
                )

                if pair_dataset:
                    pair_model: ModelWrapper = pair_trainer.train_sft(pair_dataset)
                    pair_evaluator: Evaluator = Evaluator(
                        model=pair_model,
                        reward_fn=reward_fn,
                        prompt_templates=prompt_templates,
                        config=self.config,
                    )
                    pair_results: Dict[str, Any] = pair_evaluator.evaluate(
                        test_data=test_data,
                        temperature=self.config.eval_temperature,
                    )
                    all_results["pair_sft"] = pair_results
                    loggable_pair: Dict[str, Any] = {
                        f"pair_sft/{k}": v
                        for k, v in pair_results.items()
                        if isinstance(v, (int, float, bool))
                    }
                    self.logger.log_metrics(loggable_pair, step=0)
                    logger.info(
                        "run_sft_baselines: Pair-SFT — "
                        "acc@t1=%.3f, acc@t2=%.3f, delta=%.3f.",
                        pair_results.get("accuracy_t1", 0.0),
                        pair_results.get("accuracy_t2", 0.0),
                        pair_results.get("delta_t1_t2", 0.0),
                    )
                    del pair_model
                else:
                    logger.warning(
                        "run_sft_baselines: Pair-SFT dataset is empty. "
                        "Skipping Pair-SFT evaluation."
                    )

                del pair_base_model

            # ------------------------------------------------------------------
            # Step 5: Save all baseline results to disk.
            # ------------------------------------------------------------------
            self._save_results_json(
                all_results, filename="sft_baselines_results.json"
            )
            logger.info(
                "run_sft_baselines: All baseline results saved. "
                "Methods evaluated: %s.",
                list(all_results.keys()),
            )

        except Exception as exc:
            logger.error(
                "run_sft_baselines: Failed with exception: %s.",
                exc,
                exc_info=True,
            )
            raise
        finally:
            self.logger.finish()

    # -------------------------------------------------------------------------
    # Ablation studies (Table 4)
    # -------------------------------------------------------------------------

    def run_ablations(self, ablation_type: str = "all") -> None:
        """Reproduce Table 4 ablation studies.

        Four ablations, each modifying one component of SCoRe:
            1. w/o multi-turn training: single-turn RL only.
            2. w/o Stage I: Stage II directly from base model.
            3. w/o reward shaping: alpha=0 in Stage II.
            4. STaR instead of REINFORCE in Stage II.

        Args:
            ablation_type: Which ablation(s) to run. One of:
                'single_turn'       — Ablation 1: w/o multi-turn training.
                'no_stage1'         — Ablation 2: w/o Stage I.
                'no_reward_shaping' — Ablation 3: w/o reward shaping.
                'star_stage2'       — Ablation 4: STaR in Stage II.
                'all'               — All four ablations.

        Config values used:
            ablations.single_turn_only.enabled
            ablations.skip_stage1.enabled
            ablations.no_reward_shaping.enabled
            ablations.star_stage2.enabled
        """
        logger.info(
            "run_ablations: Starting ablation studies. type='%s'.",
            ablation_type,
        )

        try:
            # ------------------------------------------------------------------
            # Shared setup: data and reward function.
            # ------------------------------------------------------------------
            loader: DatasetLoader = DatasetLoader(self.config)
            train_data: List[Dict[str, Any]] = loader.load_train_data()
            test_data: List[Dict[str, Any]] = loader.load_test_data()
            reward_fn: RewardFunction = RewardFunction(task=self.config.task)
            prompt_templates: PromptTemplates = PromptTemplates()

            logger.info(
                "run_ablations: Loaded %d train, %d test examples.",
                len(train_data),
                len(test_data),
            )

            all_ablation_results: Dict[str, Dict[str, Any]] = {}

            # ------------------------------------------------------------------
            # Ablation 1: w/o multi-turn training (single-turn RL only).
            # Table 4: acc@t1=61.8%, acc@t2=59.4%, Δ=-2.4%.
            # Uses REINFORCETrainer in single-turn mode (only turn-2 reward).
            # ------------------------------------------------------------------
            if ablation_type in ("single_turn", "all"):
                logger.info(
                    "run_ablations: Running ablation 1 — w/o multi-turn training."
                )
                ablation1_results: Dict[str, Any] = (
                    self._run_ablation_single_turn(
                        train_data=train_data,
                        test_data=test_data,
                        reward_fn=reward_fn,
                        prompt_templates=prompt_templates,
                    )
                )
                all_ablation_results["single_turn_only"] = ablation1_results
                loggable_a1: Dict[str, Any] = {
                    f"ablation_single_turn/{k}": v
                    for k, v in ablation1_results.items()
                    if isinstance(v, (int, float, bool))
                }
                self.logger.log_metrics(loggable_a1, step=0)
                logger.info(
                    "run_ablations: Ablation 1 (single-turn) — "
                    "acc@t1=%.3f, acc@t2=%.3f, delta=%.3f.",
                    ablation1_results.get("accuracy_t1", 0.0),
                    ablation1_results.get("accuracy_t2", 0.0),
                    ablation1_results.get("delta_t1_t2", 0.0),
                )

            # ------------------------------------------------------------------
            # Ablation 2: w/o Stage I (Stage II directly from base model).
            # Table 4: acc@t1=59.2%, acc@t2=61.4%, Δ=2.2%.
            # ------------------------------------------------------------------
            if ablation_type in ("no_stage1", "all"):
                logger.info(
                    "run_ablations: Running ablation 2 — w/o Stage I."
                )
                ablation2_results: Dict[str, Any] = (
                    self._run_ablation_no_stage1(
                        train_data=train_data,
                        test_data=test_data,
                        reward_fn=reward_fn,
                        prompt_templates=prompt_templates,
                    )
                )
                all_ablation_results["no_stage1"] = ablation2_results
                loggable_a2: Dict[str, Any] = {
                    f"ablation_no_stage1/{k}": v
                    for k, v in ablation2_results.items()
                    if isinstance(v, (int, float, bool))
                }
                self.logger.log_metrics(loggable_a2, step=0)
                logger.info(
                    "run_ablations: Ablation 2 (no Stage I) — "
                    "acc@t1=%.3f, acc@t2=%.3f, delta=%.3f.",
                    ablation2_results.get("accuracy_t1", 0.0),
                    ablation2_results.get("accuracy_t2", 0.0),
                    ablation2_results.get("delta_t1_t2", 0.0),
                )

            # ------------------------------------------------------------------
            # Ablation 3: w/o reward shaping (alpha=0 in Stage II).
            # Table 4: acc@t1=60.0%, acc@t2=62.6%, Δ=2.6%.
            # ------------------------------------------------------------------
            if ablation_type in ("no_reward_shaping", "all"):
                logger.info(
                    "run_ablations: Running ablation 3 — w/o reward shaping "
                    "(alpha=0)."
                )
                ablation3_results: Dict[str, Any] = (
                    self._run_ablation_no_reward_shaping(
                        train_data=train_data,
                        test_data=test_data,
                        reward_fn=reward_fn,
                        prompt_templates=prompt_templates,
                    )
                )
                all_ablation_results["no_reward_shaping"] = ablation3_results
                loggable_a3: Dict[str, Any] = {
                    f"ablation_no_reward_shaping/{k}": v
                    for k, v in ablation3_results.items()
                    if isinstance(v, (int, float, bool))
                }
                self.logger.log_metrics
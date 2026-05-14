```python
## main.py
"""Entry point for Adjoint Matching fine-tuning experiments.

This module orchestrates the entire experimental pipeline:
- CLI argument parsing with config.yaml defaults
- Config loading and override application
- Dataset initialization with reproducible train/test splits
- Single experiment execution (train + evaluate)
- Ablation studies: lambda (Table 2), noise schedule (Table 7)
- All-baselines comparison (Table 2 full comparison)

Configuration alignment (config.yaml):
    All default values sourced from config.yaml sections:
    model, flow_matching, noise_schedule, sampling, training,
    reward, loss, algorithms, data, evaluation, inference, paths, ablations

Dependencies:
    - config.py: Config, VALID_ALGORITHMS
    - prompt_dataset.py: PromptDataset
    - trainer.py: Trainer
    - evaluator.py: Evaluator
    - utils.py: set_seed

Usage:
    # Single experiment with defaults (Adjoint Matching, lambda=12500)
    python main.py

    # Override algorithm and lambda
    python main.py --algorithm draft_1 --lambda_reward 1000

    # Evaluation only from checkpoint
    python main.py --eval_only --checkpoint outputs/checkpoints/iter_1000.pt

    # Lambda ablation (Table 2)
    python main.py --run_ablation_lambda

    # Noise schedule ablation (Table 7)
    python main.py --run_ablation_noise

    # All baselines comparison (Table 2 full)
    python main.py --run_all_baselines
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

from config import Config, VALID_ALGORITHMS
from prompt_dataset import PromptDataset
from utils import set_seed

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants (from config.yaml)
# ---------------------------------------------------------------------------

# Default config file path
_DEFAULT_CONFIG_PATH: str = "config.yaml"

# Lambda values for ablation (config.yaml ablations.lambda_ablation.values)
_LAMBDA_ABLATION_VALUES: List[float] = [1000.0, 2500.0, 12500.0]

# Noise schedule values for ablation (config.yaml ablations.noise_schedule_ablation.schedules)
_NOISE_SCHEDULE_ABLATION_VALUES: List[str] = ["memoryless", "constant", "zero"]

# All algorithms for baseline comparison (config.yaml algorithms section)
_ALL_ALGORITHMS: List[str] = [
    "adjoint_matching",
    "draft_1",
    "draft_40",
    "refl",
    "dpo",
    "cont_adjoint",
    "disc_adjoint",
]

# Default iteration counts per algorithm (config.yaml algorithms section)
_ALGORITHM_ITERATIONS: Dict[str, int] = {
    "adjoint_matching": 1000,
    "draft_1": 4000,
    "draft_40": 500,
    "refl": 1500,
    "dpo": 1000,
    "cont_adjoint": 750,
    "disc_adjoint": 1000,
}

# Discrete adjoint learning rate override (config.yaml algorithms.disc_adjoint.learning_rate_override)
_DISC_ADJOINT_LR: float = 1.0e-5

# Default lambda for noise schedule ablation (Table 7 uses lambda=12500)
_NOISE_ABLATION_LAMBDA: float = 12500.0

# CFG guidance weights for evaluation (config.yaml inference.guidance_weights)
_CFG_GUIDANCE_WEIGHTS: List[float] = [0.0, 1.0, 4.0]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _load_config(config_path: str) -> Config:
    """Load configuration from a YAML file and construct a Config object.

    Reads the YAML file at config_path, passes the nested dict to
    Config.from_dict() which flattens and validates all fields.

    Args:
        config_path: Path to the config.yaml file.
            Default: "config.yaml" (config.yaml root).

    Returns:
        Fully initialized Config object with all derived fields computed
        (h, lct, lct_cont_adjoint, sigma_offset, num_iterations).

    Raises:
        FileNotFoundError: If config_path does not exist.
        yaml.YAMLError: If the YAML file is malformed.
        ValueError: If Config validation fails (invalid algorithm, etc.).
    """
    if not os.path.isfile(config_path):
        logger.warning(
            "Config file '%s' not found. Using Config defaults.", config_path
        )
        return Config()

    logger.info("Loading config from '%s'...", config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        yaml_dict: Dict[str, Any] = yaml.safe_load(f) or {}

    config: Config = Config.from_dict(yaml_dict)
    logger.info("Config loaded: %s", config)
    return config


def _apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Apply CLI argument overrides on top of the loaded config.

    Modifies the config object in-place. When lambda_reward changes,
    LCT is recomputed automatically (LCT = lct_constant * lambda^2).
    This recomputation is mandatory — forgetting it would silently
    produce wrong clipping behavior (Appendix G.3).

    Override precedence: config.yaml defaults < CLI arguments.

    Args:
        config: Config object loaded from config.yaml.
        args: Parsed argparse.Namespace with CLI arguments.

    Returns:
        Modified Config object with CLI overrides applied.
        Note: Config is modified in-place AND returned for convenience.
    """
    # --- Algorithm override ---
    if args.algorithm is not None:
        if args.algorithm not in VALID_ALGORITHMS:
            raise ValueError(
                f"Invalid algorithm '{args.algorithm}'. "
                f"Must be one of: {VALID_ALGORITHMS}"
            )
        config.algorithm = args.algorithm
        # Reset num_iterations to algorithm default if not explicitly set
        if args.num_iterations is None:
            config.num_iterations = _ALGORITHM_ITERATIONS.get(
                args.algorithm, config.num_iterations
            )

    # --- Lambda reward override ---
    # CRITICAL: Must recompute LCT whenever lambda changes (Appendix G.3)
    if args.lambda_reward is not None:
        if args.lambda_reward <= 0.0:
            raise ValueError(
                f"lambda_reward must be positive, got {args.lambda_reward}."
            )
        config.lambda_reward = float(args.lambda_reward)
        # Recompute LCT = lct_constant * lambda^2 (Appendix G.3)
        config.lct = config.lct_constant * (config.lambda_reward ** 2)
        # Also recompute continuous adjoint LCT
        config.lct_cont_adjoint = (
            config.lct_constant_cont_adjoint * (config.lambda_reward ** 2)
        )
        logger.info(
            "Lambda override: lambda=%.1f, LCT=%.2e, LCT_cont=%.2e",
            config.lambda_reward,
            config.lct,
            config.lct_cont_adjoint,
        )

    # --- Num iterations override ---
    if args.num_iterations is not None:
        if args.num_iterations <= 0:
            raise ValueError(
                f"num_iterations must be positive, got {args.num_iterations}."
            )
        config.num_iterations = args.num_iterations

    # --- Output directory override ---
    if args.output_dir is not None:
        config.output_dir = args.output_dir
        config.checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
        config.results_dir = os.path.join(args.output_dir, "results")
        config.log_dir = os.path.join(args.output_dir, "logs")

    # --- Seed override ---
    if args.seed is not None:
        config.seed = args.seed

    # --- Wandb override ---
    if getattr(args, "no_wandb", False):
        config.wandb_project = ""

    if getattr(args, "wandb_project", None) is not None:
        config.wandb_project = args.wandb_project

    return config


def _setup_directories(config: Config) -> None:
    """Create all required output directories.

    Creates the following directories from config.yaml paths section:
        paths.output_dir: "outputs"
        paths.checkpoint_dir: "outputs/checkpoints"
        paths.results_dir: "outputs/results"
        paths.log_dir: "outputs/logs"

    Uses exist_ok=True so re-running doesn't fail if directories exist.

    Args:
        config: Config object with output path fields.
    """
    for dir_path in [
        config.output_dir,
        config.checkpoint_dir,
        config.results_dir,
        config.log_dir,
    ]:
        os.makedirs(dir_path, exist_ok=True)
        logger.debug("Directory ensured: '%s'", dir_path)

    logger.info(
        "Output directories created under '%s'.", config.output_dir
    )


def _save_results(
    results: Dict[str, Any],
    results_dir: str,
    name: str,
) -> None:
    """Save experiment results to a JSON file.

    Uses json.dump with default=str to handle non-serializable types
    (numpy floats, torch tensors, etc.) that might appear in results.

    Args:
        results: Dictionary of experiment results to serialize.
        results_dir: Directory path where the JSON file will be saved.
            Created if it doesn't exist.
        name: Base name for the JSON file (without extension).
            File will be saved as: {results_dir}/{name}.json
    """
    os.makedirs(results_dir, exist_ok=True)
    path: str = os.path.join(results_dir, f"{name}.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("Results saved to '%s'.", path)
    print(f"Results saved to {path}")


def _print_results(
    results: Dict[str, Any],
    algorithm: str,
    lambda_val: float,
) -> None:
    """Print a human-readable results summary matching Table 2 format.

    Prints metrics for both sampling modes (sigma=0 and sigma=memoryless)
    and all CFG guidance weights tested.

    Args:
        results: Results dict from evaluator.run_evaluation_suite().
            Expected structure:
                {
                    "sigma_zero": {
                        "clipscore": float,
                        "pickscore": float,
                        "hpsv2": float,
                        "dreamsim_diversity": float,
                        "imagereward": float,
                    },
                    "sigma_memoryless": { ... },
                    "cfg_w1.0": { ... },  # optional
                    ...
                }
        algorithm: Algorithm name for display (e.g., "adjoint_matching").
        lambda_val: Lambda reward scaling factor for display.
    """
    separator: str = "=" * 65
    print(f"\n{separator}")
    print(f"  Algorithm: {algorithm}  |  Lambda: {lambda_val:.0f}")
    print(separator)

    # Define display order for sampling modes
    sampling_modes: List[Tuple[str, str]] = [
        ("sigma_zero", "σ(t) = 0 (ODE)"),
        ("sigma_memoryless", "σ(t) = √(2η_t) (Memoryless SDE)"),
    ]

    for mode_key, mode_label in sampling_modes:
        if mode_key not in results:
            continue
        r: Dict[str, Any] = results[mode_key]
        print(f"\n  Sampling: {mode_label}")
        print(f"    ClipScore:          {r.get('clipscore', 'N/A')}")
        print(f"    PickScore:          {r.get('pickscore', 'N/A')}")
        print(f"    HPSv2:              {r.get('hpsv2', 'N/A')}")
        print(f"    DreamSim Diversity: {r.get('dreamsim_diversity', 'N/A')}")
        print(f"    ImageReward:        {r.get('imagereward', 'N/A')}")

    # Print CFG ablation results if present
    for w in _CFG_GUIDANCE_WEIGHTS:
        cfg_key: str = f"cfg_w{w}"
        if cfg_key in results:
            r = results[cfg_key]
            print(f"\n  CFG w={w}:")
            print(f"    ClipScore:          {r.get('clipscore', 'N/A')}")
            print(f"    PickScore:          {r.get('pickscore', 'N/A')}")
            print(f"    HPSv2:              {r.get('hpsv2', 'N/A')}")
            print(f"    DreamSim Diversity: {r.get('dreamsim_diversity', 'N/A')}")
            print(f"    ImageReward:        {r.get('imagereward', 'N/A')}")

    print(f"{separator}\n")


def _make_config_for_algorithm(
    base_config: Config,
    algorithm: str,
    lambda_reward: Optional[float] = None,
    sigma_schedule: Optional[str] = None,
    num_iterations: Optional[int] = None,
) -> Config:
    """Create a deep-copied Config variant for a specific algorithm/setting.

    Uses copy.deepcopy to ensure the base config is never mutated.
    Applies algorithm-specific overrides including:
    - Algorithm name and iteration count
    - Lambda reward and LCT recomputation
    - Noise schedule type
    - Discrete adjoint learning rate override (Table 6)
    - Continuous adjoint LCT override (Appendix G.3)

    Args:
        base_config: Original Config object (never mutated).
        algorithm: Algorithm name from VALID_ALGORITHMS.
        lambda_reward: Optional lambda override. If None, uses base_config value.
            When set, LCT is automatically recomputed.
        sigma_schedule: Optional noise schedule override ("memoryless", "constant", "zero").
            If None, uses base_config.sigma_schedule.
        num_iterations: Optional iteration count override.
            If None, uses the algorithm's default from _ALGORITHM_ITERATIONS.

    Returns:
        New Config object (deep copy) with all specified overrides applied
        and derived fields (h, lct, lct_cont_adjoint) recomputed.
    """
    # Deep copy to prevent mutation of base config
    cfg: Config = copy.deepcopy(base_config)

    # Set algorithm
    cfg.algorithm = algorithm

    # Set iteration count
    if num_iterations is not None:
        cfg.num_iterations = num_iterations
    else:
        cfg.num_iterations = _ALGORITHM_ITERATIONS.get(algorithm, cfg.num_iterations)

    # Set lambda and recompute LCT (CRITICAL: must happen whenever lambda changes)
    if lambda_reward is not None:
        cfg.lambda_reward = float(lambda_reward)

    # Always recompute LCT after any lambda change
    cfg.lct = cfg.lct_constant * (cfg.lambda_reward ** 2)
    cfg.lct_cont_adjoint = cfg.lct_constant_cont_adjoint * (cfg.lambda_reward ** 2)

    # For continuous adjoint, use the larger LCT constant (Appendix G.3)
    # The lct field in Config is used by AdjointMatchingLoss.compute()
    # For cont_adjoint, trainer.py uses config.lct_cont_adjoint instead
    # This is handled in trainer.py's train_step_cont_adjoint()

    # Set noise schedule
    if sigma_schedule is not None:
        cfg.sigma_schedule = sigma_schedule

    # Discrete adjoint learning rate override (Table 6, Appendix G)
    # config.yaml algorithms.disc_adjoint.learning_rate_override: 1e-5
    if algorithm == "disc_adjoint":
        cfg.disc_adjoint_lr = _DISC_ADJOINT_LR
        logger.info(
            "Discrete Adjoint: applying LR override %.2e (Table 6).",
            _DISC_ADJOINT_LR,
        )

    return cfg


def _build_dataset(config: Config) -> Tuple[PromptDataset, List[str], List[str]]:
    """Initialize the prompt dataset and perform train/test split.

    Implements the paper's protocol (Appendix G):
        - Total pool: 100k prompts (config.yaml data.total_prompts: 100000)
        - Training: 40k prompts per run (config.yaml data.num_train_prompts: 40000)
        - Test: 1k prompts per run (config.yaml data.num_test_prompts: 1000)
        - 3 independent runs with different seeds

    Args:
        config: Config object with data and training fields.
            Key fields:
                config.prompts_file (data.prompts_file: "data/prompts.txt")
                config.total_prompts (data.total_prompts: 100000)
                config.num_train_prompts (data.num_train_prompts: 40000)
                config.num_eval_prompts (data.num_test_prompts: 1000)
                config.seed (training.seed: 42)

    Returns:
        Tuple (dataset, train_prompts, test_prompts) where:
            dataset: PromptDataset instance with all prompts loaded.
            train_prompts: List of 40k training prompt strings.
            test_prompts: List of 1k test prompt strings.
    """
    logger.info(
        "Building prompt dataset: file='%s', total=%d, seed=%d",
        config.prompts_file,
        config.total_prompts,
        config.seed,
    )

    dataset: PromptDataset = PromptDataset(
        prompts_file=config.prompts_file,
        num_prompts=config.total_prompts,
        seed=config.seed,
    )

    # Perform train/test split (Appendix G: 40k train / 1k test per run)
    train_prompts: List[str]
    test_prompts: List[str]
    train_prompts, test_prompts = dataset.get_train_test_split(
        train_size=config.num_train_prompts,
        test_size=config.num_eval_prompts,
    )

    logger.info(
        "Dataset split: %d train prompts, %d test prompts.",
        len(train_prompts),
        len(test_prompts),
    )

    return dataset, train_prompts, test_prompts


def _run_training_and_evaluation(
    config: Config,
    dataset: PromptDataset,
    test_prompts: List[str],
    checkpoint_path: Optional[str] = None,
    eval_only: bool = False,
) -> Dict[str, Any]:
    """Run training (optional) and evaluation for a single experiment.

    This is the core execution unit called by all experiment modes.
    Instantiates Trainer and Evaluator, runs training if not eval_only,
    then runs the full evaluation suite.

    Args:
        config: Config object for this experiment variant.
        dataset: PromptDataset with train/test split already performed.
        test_prompts: List of test prompt strings for evaluation.
        checkpoint_path: Optional path to a checkpoint file.
            If provided, loads checkpoint before training/evaluation.
        eval_only: If True, skip training and only run evaluation.
            Requires checkpoint_path to be set for meaningful results.

    Returns:
        Results dict from evaluator.run_evaluation_suite(test_prompts).
        Structure:
            {
                "sigma_zero": {"clipscore": float, "pickscore": float, ...},
                "sigma_memoryless": {"clipscore": float, ...},
                "cfg_w0.0": {...},
                "cfg_w1.0": {...},
                "cfg_w4.0": {...},
            }
    """
    # Import here to avoid circular imports at module level
    from trainer import Trainer
    from evaluator import Evaluator

    # ------------------------------------------------------------------
    # Step 1: Instantiate Trainer (loads all models from HuggingFace)
    # Each call creates fresh model weights from config.model_id.
    # This is critical for ablation studies — weights must not carry over.
    # ------------------------------------------------------------------
    logger.info(
        "Instantiating Trainer: algorithm=%s, lambda=%.1f, K=%d",
        config.algorithm,
        config.lambda_reward,
        config.K,
    )
    trainer: Trainer = Trainer(config=config, dataset=dataset)

    # ------------------------------------------------------------------
    # Step 2: Load checkpoint if provided
    # ------------------------------------------------------------------
    if checkpoint_path is not None:
        if os.path.isfile(checkpoint_path):
            logger.info("Loading checkpoint from '%s'...", checkpoint_path)
            trainer.load_checkpoint(checkpoint_path)
        else:
            logger.warning(
                "Checkpoint file '%s' not found. Starting from scratch.",
                checkpoint_path,
            )

    # ------------------------------------------------------------------
    # Step 3: Run training (unless eval_only)
    # ------------------------------------------------------------------
    if not eval_only:
        logger.info(
            "Starting training: %d iterations, batch_size=%d",
            config.num_iterations,
            config.batch_size,
        )
        trainer.train()
        logger.info("Training complete.")
    else:
        logger.info("eval_only=True: skipping training.")

    # ------------------------------------------------------------------
    # Step 4: Instantiate Evaluator with the trained v_theta
    # The Evaluator receives the TRAINED v_theta from Trainer.
    # ------------------------------------------------------------------
    logger.info("Instantiating Evaluator for metric computation...")
    evaluator: Evaluator = Evaluator(
        config=config,
        v_theta=trainer.v_theta,
        v_base=trainer.v_base,
        vae=trainer.vae,
        text_encoder=trainer.text_encoder,
        tokenizer=trainer.tokenizer,
    )

    # ------------------------------------------------------------------
    # Step 5: Run full evaluation suite
    # Computes all 5 metrics for both sampling modes and CFG weights.
    # ------------------------------------------------------------------
    logger.info(
        "Running evaluation suite on %d test prompts...",
        len(test_prompts),
    )
    results: Dict[str, Any] = evaluator.run_evaluation_suite(test_prompts)
    logger.info("Evaluation complete.")

    return results


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class Main:
    """Top-level orchestrator for Adjoint Matching experiments.

    Handles CLI argument parsing, config loading, dataset initialization,
    and dispatching to the appropriate experiment mode:
        - Single experiment (default)
        - Lambda ablation (Table 2, --run_ablation_lambda)
        - Noise schedule ablation (Table 7, --run_ablation_noise)
        - All baselines comparison (Table 2 full, --run_all_baselines)

    Attributes:
        config: Loaded and override-applied Config object.
        args: Parsed CLI arguments.
        dataset: PromptDataset with train/test split.
        train_prompts: List of training prompt strings.
        test_prompts: List of test prompt strings.
    """

    def __init__(self, config_path: str = _DEFAULT_CONFIG_PATH) -> None:
        """Initialize Main by parsing CLI args and loading config.

        Args:
            config_path: Default path to config.yaml. Can be overridden
                by the --config CLI argument.
        """
        # Parse CLI arguments
        self.args: argparse.Namespace = self._parse_args()

        # Use --config argument if provided, otherwise use config_path parameter
        effective_config_path: str = (
            self.args.config
            if self.args.config is not None
            else config_path
        )

        # Load config from YAML
        self.config: Config = _load_config(effective_config_path)

        # Apply CLI overrides on top of YAML config
        self.config = _apply_overrides(self.config, self.args)

        # Determine device (CUDA if available, else CPU)
        device_str: str = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.config.device = device_str
        logger.info("Using device: %s", device_str)

        # Set random seed for reproducibility (Appendix G: 3 independent runs)
        # config.yaml training.seed: 42
        set_seed(self.config.seed)

        # Create output directories
        _setup_directories(self.config)

        # Initialize dataset (lazy — split happens in run())
        self.dataset: Optional[PromptDataset] = None
        self.train_prompts: List[str] = []
        self.test_prompts: List[str] = []

    def _parse_args(self) -> argparse.Namespace:
        """Parse command-line arguments.

        All arguments are optional; config.yaml provides defaults.
        CLI arguments override config.yaml values.

        Returns:
            Parsed argparse.Namespace with all CLI argument values.
        """
        parser = argparse.ArgumentParser(
            description=(
                "Adjoint Matching: Fine-tuning Flow and Diffusion Generative "
                "Models with Memoryless Stochastic Optimal Control"
            ),
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        # Config file
        parser.add_argument(
            "--config",
            type=str,
            default=None,
            help=(
                "Path to config.yaml file. "
                f"Default: '{_DEFAULT_CONFIG_PATH}'"
            ),
        )

        # Algorithm selection
        parser.add_argument(
            "--algorithm",
            type=str,
            default=None,
            choices=VALID_ALGORITHMS,
            help=(
                "Fine-tuning algorithm to use. "
                "Overrides config.yaml algorithm field. "
                f"Choices: {VALID_ALGORITHMS}"
            ),
        )

        # Lambda reward scaling
        parser.add_argument(
            "--lambda_reward",
            type=float,
            default=None,
            help=(
                "Reward scaling factor λ. r(x) = λ * RewardModel(x). "
                "Overrides config.yaml reward.lambda_reward. "
                "Paper tests: 1000, 2500, 12500 (Table 2). "
                "Default from config.yaml: 12500."
            ),
        )

        # Number of training iterations
        parser.add_argument(
            "--num_iterations",
            type=int,
            default=None,
            help=(
                "Number of fine-tuning iterations. "
                "Overrides the algorithm's default from config.yaml. "
                "Default varies by algorithm (e.g., 1000 for adjoint_matching)."
            ),
        )

        # Evaluation only mode
        parser.add_argument(
            "--eval_only",
            action="store_true",
            default=False,
            help=(
                "Skip training and only run evaluation. "
                "Requires --checkpoint to be set for meaningful results."
            ),
        )

        # Checkpoint path
        parser.add_argument(
            "--checkpoint",
            type=str,
            default=None,
            help=(
                "Path to a checkpoint file (.pt) for resuming training "
                "or evaluation-only mode. "
                "Example: outputs/checkpoints/iter_1000.pt"
            ),
        )

        # Ablation: lambda sweep (Table 2)
        parser.add_argument(
            "--run_ablation_lambda",
            action="store_true",
            default=False,
            help=(
                "Run lambda ablation study (Table 2). "
                "Trains and evaluates Adjoint Matching for "
                f"λ ∈ {_LAMBDA_ABLATION_VALUES}. "
                "Mutually exclusive with --run_ablation_noise and --run_all_baselines."
            ),
        )

        # Ablation: noise schedule (Table 7)
        parser.add_argument(
            "--run_ablation_noise",
            action="store_true",
            default=False,
            help=(
                "Run noise schedule ablation study (Table 7). "
                "Compares memoryless, constant (σ=1), and zero (ODE) schedules. "
                "Mutually exclusive with --run_ablation_lambda and --run_all_baselines."
            ),
        )

        # All baselines comparison (Table 2 full)
        parser.add_argument(
            "--run_all_baselines",
            action="store_true",
            default=False,
            help=(
                "Run all baseline methods for comparison (Table 2). "
                f"Trains and evaluates: {_ALL_ALGORITHMS}. "
                "Mutually exclusive with --run_ablation_lambda and --run_ablation_noise."
            ),
        )

        # Output directory override
        parser.add_argument(
            "--output_dir",
            type=str,
            default=None,
            help=(
                "Override output directory for all experiment artifacts. "
                "Overrides config.yaml paths.output_dir. "
                "Default from config.yaml: 'outputs'."
            ),
        )

        # Random seed override
        parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help=(
                "Random seed for reproducibility. "
                "Overrides config.yaml training.seed. "
                "For 3 independent runs (Appendix G), use seeds 42, 43, 44. "
                "Default from config.yaml: 42."
            ),
        )

        # Wandb project override
        parser.add_argument(
            "--wandb_project",
            type=str,
            default=None,
            help=(
                "Weights & Biases project name. "
                "Overrides config.yaml paths.wandb_project. "
                "Default from config.yaml: 'adjoint-matching'."
            ),
        )

        # Disable wandb
        parser.add_argument(
            "--no_wandb",
            action="store_true",
            default=False,
            help=(
                "Disable Weights & Biases logging. "
                "Useful for debugging or offline runs."
            ),
        )

        return parser.parse_args()

    def run(self) -> None:
        """Main entry point — dispatch to the appropriate experiment mode.

        Validates that ablation flags are mutually exclusive, initializes
        the dataset, then dispatches to the correct experiment function.

        Dispatch logic:
            --run_ablation_lambda  → run_ablation_lambda()
            --run_ablation_noise   → run_ablation_noise_schedule()
            --run_all_baselines    → run_all_baselines()
            (default)              → run_training()  [single experiment]
        """
        # Validate mutual exclusivity of ablation flags
        ablation_flags: List[bool] = [
            self.args.run_ablation_lambda,
            self.args.run_ablation_noise,
            self.args.run_all_baselines,
        ]
        if sum(ablation_flags) > 1:
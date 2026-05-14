```python
## config.py
"""Configuration dataclass for SCoRe: Self-Correction via Reinforcement Learning.

This module defines the central Config dataclass that consolidates all
hyperparameters and settings for the SCoRe training pipeline. Values are
sourced directly from Table 5 (Appendix B) and methodology sections of the
paper. The from_dict() classmethod handles flattening of the nested YAML
structure into a flat dataclass, with task-conditional selection of
hyperparameters (math vs. code).

Typical usage:
    import yaml
    with open("config.yaml") as f:
        raw = yaml.safe_load(f)
    config = Config.from_dict(raw)
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Central configuration for the SCoRe training and evaluation pipeline.

    All numeric defaults correspond to the MATH task values from Table 5
    (Appendix B) of the paper. When task="code", from_dict() overrides
    these with the MBPP column values.

    Attributes:
        task: Task identifier, either "math" or "code".
        run_name: Human-readable name for the experiment run.
        output_dir: Directory for saving checkpoints and results.
        seed: Global random seed for reproducibility.
        wandb_project: Weights & Biases project name for logging.
        log_level: Python logging level string (e.g., "INFO", "DEBUG").
        model_name: HuggingFace model identifier. Defaults to the open-source
            substitute for Gemini 1.5 Flash (MATH task).
        torch_dtype: PyTorch dtype string for model loading ("bfloat16",
            "float16", "float32").
        device_map: HuggingFace device_map argument for model placement.
        max_new_tokens: Maximum number of new tokens to generate per turn.
        eval_temperature: Sampling temperature for evaluation. 0.0 = greedy
            decoding (Section 6: "greedy decoding (temperature 0)").
        sampling_temperature: Sampling temperature for training rollouts.
            Table 5: "Sampling temperature: 1.0".
        scaling_temperature: Temperature for inference-compute scaling
            experiments. Section 6.2: "temperature to be 0.7".
        learning_rate: Adam optimizer learning rate. Table 5: 5e-6 (MATH),
            1e-5 (code).
        stage1_steps: Number of gradient steps for Stage I training. Derived
            from Table 5 total steps with 50/50 split assumption.
        stage2_steps: Number of gradient steps for Stage II training.
        batch_size: Per-device training batch size. Table 5: 512 (MATH),
            128 (code).
        adam_beta1: Adam optimizer beta1 coefficient.
        adam_beta2: Adam optimizer beta2 coefficient.
        adam_epsilon: Adam optimizer epsilon for numerical stability.
        weight_decay: L2 weight decay coefficient.
        max_grad_norm: Maximum gradient norm for gradient clipping.
        gradient_accumulation_steps: Number of steps to accumulate gradients
            before an optimizer update.
        alpha: Reward shaping multiplier for Stage II. Table 5: α = 10.
            Applied as: shaped_r2 = r2 + alpha * (r2 - r1). Paper states
            "α is a positive constant multiplier, ideally larger than 1.0".
        beta1: Standard KL divergence penalty weight applied to both turns.
            Table 5: β₁ = 0.01. Used in Equations 2 and 4.
        beta2: Stage I first-turn KL penalty weight. Table 5: β₂ = 0.1
            (MATH), 0.25 (code). Must be > beta1 to enforce strict first-turn
            constraint from Equation 3.
        normalize_rewards: Whether to apply per-batch reward normalization
            (subtract mean, divide by std) before computing policy gradient.
            Standard REINFORCE variance reduction practice.
        reward_norm_eps: Epsilon for numerical stability in reward
            normalization denominator.
        offline_data_mix_ratio: Fraction of offline base model first-attempt
            solutions to mix into Stage II batches. Section 5.3. Paper does
            not specify ratio; defaults to 0.0 (disabled).
        gamma: Discount factor for reward computation. Appendix A.2 explicitly
            states γ = 0 (instantaneous reward only).
        use_lora: Whether to apply LoRA for parameter-efficient fine-tuning.
            Not specified in paper; added for practical GPU memory constraints.
        lora_rank: LoRA rank (r). Not specified in paper.
        lora_alpha: LoRA alpha scaling factor. Not specified in paper.
        lora_dropout: LoRA dropout probability. Not specified in paper.
        lora_target_modules: List of module names to apply LoRA to. Typical
            attention and MLP projection layers for LLaMA/Qwen/DeepSeek.
        eval_every_n_steps: Frequency of evaluation during training.
        save_every_n_steps: Frequency of checkpoint saving during training.
        deepspeed_config: Path to DeepSpeed ZeRO configuration JSON file.
    """

    # -------------------------------------------------------------------------
    # Identity and experiment tracking
    # -------------------------------------------------------------------------
    task: str = "math"
    run_name: str = "score_experiment"
    output_dir: str = "outputs/"
    seed: int = 42
    wandb_project: str = "score_self_correction"
    log_level: str = "INFO"

    # -------------------------------------------------------------------------
    # Model configuration
    # -------------------------------------------------------------------------
    # Default: open-source substitute for Gemini 1.5 Flash (MATH task)
    model_name: str = "Qwen/Qwen2.5-Math-7B-Instruct"
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    max_new_tokens: int = 1024
    # Section 6: "greedy decoding (i.e. temperature 0)"
    eval_temperature: float = 0.0
    # Table 5: "Sampling temperature: 1.0"
    sampling_temperature: float = 1.0
    # Section 6.2: "we set temperature to be 0.7"
    scaling_temperature: float = 0.7

    # -------------------------------------------------------------------------
    # Training hyperparameters (defaults = MATH values from Table 5)
    # -------------------------------------------------------------------------
    # Table 5 MATH: 5e-6; Table 5 MBPP: 1e-5
    learning_rate: float = 5e-6
    # Table 5 MATH total: 3000 steps; 50/50 split assumption
    stage1_steps: int = 1500
    stage2_steps: int = 1500
    # Table 5 MATH: 512; Table 5 MBPP: 128
    batch_size: int = 512
    # Standard Adam defaults (not specified in paper)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    # -------------------------------------------------------------------------
    # SCoRe-specific hyperparameters (defaults = MATH values from Table 5)
    # -------------------------------------------------------------------------
    # Table 5: α = 10 for both MATH and MBPP
    # shaped_r2 = r2 + alpha * (r2 - r1)
    alpha: float = 10.0
    # Table 5: β₁ = 0.01 for both MATH and MBPP
    # Standard KL penalty weight (Equations 2 and 4)
    beta1: float = 0.01
    # Table 5: β₂ = 0.1 (MATH), 0.25 (MBPP)
    # Stage I first-turn KL penalty weight (Equation 3)
    beta2: float = 0.1
    # Per-batch reward normalization for REINFORCE variance reduction
    normalize_rewards: bool = True
    reward_norm_eps: float = 1e-8
    # Section 5.3: offline base model data augmentation in Stage II
    # Paper does not specify ratio; 0.0 = disabled by default
    offline_data_mix_ratio: float = 0.0
    # Appendix A.2: "equivalent to returns with discount factor γ = 0"
    gamma: float = 0.0

    # -------------------------------------------------------------------------
    # LoRA configuration (not in paper; added for practical GPU usage)
    # -------------------------------------------------------------------------
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # -------------------------------------------------------------------------
    # Evaluation and checkpointing
    # -------------------------------------------------------------------------
    eval_every_n_steps: int = 100
    save_every_n_steps: int = 500
    deepspeed_config: str = "configs/ds_zero2_config.json"

    # -------------------------------------------------------------------------
    # Reward function configuration
    # -------------------------------------------------------------------------
    use_sympy: bool = True
    execution_timeout: int = 5
    max_memory_mb: int = 512

    # -------------------------------------------------------------------------
    # Dataset configuration
    # -------------------------------------------------------------------------
    # MATH: "hendrycks/competition_math"
    math_dataset_name: str = "hendrycks/competition_math"
    # Lightman et al. (2023) split: 4500 test → train, 500 remain as MATH500
    num_test_problems: int = 500
    num_extra_train_from_test: int = 4500
    # Code: training on MBPP, testing on HumanEval
    train_dataset_name: str = "google-research-datasets/mbpp"
    test_dataset_name: str = "openai_humaneval"
    mbpp_r_dataset_name: str = "mbpp_r"

    # -------------------------------------------------------------------------
    # SFT baseline configuration (Section 4, Table 1)
    # -------------------------------------------------------------------------
    # "We run 3 iterations for STaR following the protocol in Singh et al. (2024)"
    star_num_iterations: int = 3
    star_include_correct_pairs: bool = False
    # "only one iteration for Pair-SFT, following the protocol in Welleck et al. (2023)"
    pair_sft_num_iterations: int = 1
    pair_sft_include_correct_pairs: bool = False

    # -------------------------------------------------------------------------
    # Inference-compute scaling (Section 6.2, Figure 1 right)
    # -------------------------------------------------------------------------
    inference_scaling_enabled: bool = False
    inference_scaling_solution_budget: int = 32
    inference_scaling_k_values: List[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 16, 32]
    )

    # -------------------------------------------------------------------------
    # Multi-attempt evaluation (Appendix A.1, Figure 8)
    # -------------------------------------------------------------------------
    multi_attempt_enabled: bool = False
    multi_attempt_num_attempts: int = 10

    # -------------------------------------------------------------------------
    # Edit distance analysis (Section 4, Figure 4)
    # -------------------------------------------------------------------------
    edit_distance_analysis_enabled: bool = True
    edit_distance_no_edit_threshold: float = 0.01
    edit_distance_large_edit_threshold: float = 0.5

    # -------------------------------------------------------------------------
    # Checkpoint selection metric
    # -------------------------------------------------------------------------
    # Section 6: "selected checkpoints with the highest training reward"
    checkpoint_metric: str = "train_reward_t2"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        """Construct a Config from a nested YAML-parsed dictionary.

        Flattens the hierarchical YAML structure into the flat Config dataclass.
        Task-conditional hyperparameters (training, SCoRe) are selected based
        on the top-level 'task' field.

        Args:
            d: Nested dictionary parsed from config.yaml via yaml.safe_load().

        Returns:
            A fully populated Config instance.

        Raises:
            ValueError: If task is not "math" or "code", or if hyperparameter
                constraints are violated.
        """
        # ------------------------------------------------------------------
        # Step 1: Read task first — determines all conditional selections
        # ------------------------------------------------------------------
        task: str = d.get("task", "math")
        if task not in ("math", "code"):
            raise ValueError(
                f"Invalid task '{task}'. Must be 'math' or 'code'."
            )

        # ------------------------------------------------------------------
        # Step 2: Experiment identity
        # ------------------------------------------------------------------
        experiment: Dict[str, Any] = d.get("experiment", {})
        run_name: str = experiment.get("run_name", "score_experiment")
        output_dir: str = experiment.get("output_dir", "outputs/")
        seed: int = int(experiment.get("seed", 42))
        wandb_project: str = experiment.get(
            "wandb_project", "score_self_correction"
        )
        log_level: str = experiment.get("log_level", "INFO")

        # ------------------------------------------------------------------
        # Step 3: Model configuration (task-conditional model name)
        # ------------------------------------------------------------------
        model_cfg: Dict[str, Any] = d.get("model", {})
        if task == "math":
            model_name: str = model_cfg.get(
                "math_model_name", "Qwen/Qwen2.5-Math-7B-Instruct"
            )
        else:
            model_name = model_cfg.get(
                "code_model_name",
                "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
            )
        torch_dtype: str = model_cfg.get("torch_dtype", "bfloat16")
        device_map: str = model_cfg.get("device_map", "auto")
        max_new_tokens: int = int(model_cfg.get("max_new_tokens", 1024))
        eval_temperature: float = float(model_cfg.get("eval_temperature", 0.0))
        sampling_temperature: float = float(
            model_cfg.get("train_temperature", 1.0)
        )
        scaling_temperature: float = float(
            model_cfg.get("scaling_temperature", 0.7)
        )

        # ------------------------------------------------------------------
        # Step 4: Training hyperparameters (task-conditional)
        # ------------------------------------------------------------------
        training_cfg: Dict[str, Any] = d.get("training", {})
        task_training: Dict[str, Any] = training_cfg.get(task, {})

        if task == "math":
            default_lr: float = 5e-6
            default_total_steps: int = 3000
            default_stage1_steps: int = 1500
            default_stage2_steps: int = 1500
            default_batch_size: int = 512
        else:
            default_lr = 1e-5
            default_total_steps = 1500
            default_stage1_steps = 750
            default_stage2_steps = 750
            default_batch_size = 128

        learning_rate: float = float(
            task_training.get("learning_rate", default_lr)
        )
        stage1_steps: int = int(
            task_training.get("stage1_steps", default_stage1_steps)
        )
        stage2_steps: int = int(
            task_training.get("stage2_steps", default_stage2_steps)
        )
        batch_size: int = int(
            task_training.get("batch_size", default_batch_size)
        )
        adam_beta1: float = float(task_training.get("adam_beta1", 0.9))
        adam_beta2: float = float(task_training.get("adam_beta2", 0.999))
        adam_epsilon: float = float(task_training.get("adam_epsilon", 1e-8))
        weight_decay: float = float(task_training.get("weight_decay", 0.0))
        max_grad_norm: float = float(task_training.get("max_grad_norm", 1.0))

        # ------------------------------------------------------------------
        # Step 5: SCoRe-specific hyperparameters (task-conditional)
        # ------------------------------------------------------------------
        score_cfg: Dict[str, Any] = d.get("score", {})
        task_score: Dict[str, Any] = score_cfg.get(task, {})

        if task == "math":
            default_alpha: float = 10.0
            default_beta1: float = 0.01
            default_beta2: float = 0.1
        else:
            default_alpha = 10.0
            default_beta1 = 0.01
            default_beta2 = 0.25

        alpha: float = float(task_score.get("alpha", default_alpha))
        beta1: float = float(task_score.get("beta1", default_beta1))
        beta2: float = float(task_score.get("beta2", default_beta2))
        normalize_rewards: bool = bool(
            score_cfg.get("normalize_rewards", True)
        )
        reward_norm_eps: float = float(
            score_cfg.get("reward_norm_eps", 1e-8)
        )
        offline_data_mix_ratio: float = float(
            score_cfg.get("offline_data_mix_ratio", 0.0)
        )
        gamma: float = float(score_cfg.get("gamma", 0.0))

        # ------------------------------------------------------------------
        # Step 6: LoRA configuration
        # ------------------------------------------------------------------
        lora_cfg: Dict[str, Any] = d.get("lora", {})
        use_lora: bool = bool(lora_cfg.get("use_lora", True))
        lora_rank: int = int(lora_cfg.get("r", 16))
        lora_alpha_val: int = int(lora_cfg.get("lora_alpha", 32))
        lora_dropout: float = float(lora_cfg.get("lora_dropout", 0.05))
        lora_target_modules: List[str] = list(
            lora_cfg.get(
                "target_modules",
                [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            )
        )

        # ------------------------------------------------------------------
        # Step 7: Evaluation and checkpointing
        # ------------------------------------------------------------------
        eval_cfg: Dict[str, Any] = d.get("evaluation", {})
        eval_every_n_steps: int = int(
            eval_cfg.get("eval_every_n_steps", 100)
        )
        save_every_n_steps: int = int(
            eval_cfg.get("save_every_n_steps", 500)
        )
        checkpoint_metric: str = eval_cfg.get(
            "checkpoint_metric", "train_reward_t2"
        )

        # Inference scaling
        inf_scaling_cfg: Dict[str, Any] = eval_cfg.get(
            "inference_scaling", {}
        )
        inference_scaling_enabled: bool = bool(
            inf_scaling_cfg.get("enabled", False)
        )
        inference_scaling_solution_budget: int = int(
            inf_scaling_cfg.get("solution_budget", 32)
        )
        inference_scaling_k_values: List[int] = list(
            inf_scaling_cfg.get("k_values", [1, 2, 4, 8, 16, 32])
        )

        # Multi-attempt
        multi_attempt_cfg: Dict[str, Any] = eval_cfg.get(
            "multi_attempt", {}
        )
        multi_attempt_enabled: bool = bool(
            multi_attempt_cfg.get("enabled", False)
        )
        multi_attempt_num_attempts: int = int(
            multi_attempt_cfg.get("num_attempts", 10)
        )

        # Edit distance analysis
        edit_dist_cfg: Dict[str, Any] = eval_cfg.get(
            "edit_distance_analysis", {}
        )
        edit_distance_analysis_enabled: bool = bool(
            edit_dist_cfg.get("enabled", True)
        )
        edit_distance_no_edit_threshold: float = float(
            edit_dist_cfg.get("no_edit_threshold", 0.01)
        )
        edit_distance_large_edit_threshold: float = float(
            edit_dist_cfg.get("large_edit_threshold", 0.5)
        )

        # ------------------------------------------------------------------
        # Step 8: Distributed training
        # ------------------------------------------------------------------
        dist_cfg: Dict[str, Any] = d.get("distributed", {})
        deepspeed_config: str = dist_cfg.get(
            "deepspeed_config", "configs/ds_zero2_config.json"
        )
        gradient_accumulation_steps: int = int(
            dist_cfg.get("gradient_accumulation_steps", 1)
        )

        # ------------------------------------------------------------------
        # Step 9: Reward function configuration
        # ------------------------------------------------------------------
        reward_cfg: Dict[str, Any] = d.get("reward", {})
        task_reward: Dict[str, Any] = reward_cfg.get(task, {})
        use_sympy: bool = bool(task_reward.get("use_sympy", True))
        execution_timeout: int = int(task_reward.get("execution_timeout", 5))
        max_memory_mb: int = int(task_reward.get("max_memory_mb", 512))

        # ------------------------------------------------------------------
        # Step 10: Dataset configuration
        # ------------------------------------------------------------------
        data_cfg: Dict[str, Any] = d.get("data", {})
        math_data: Dict[str, Any] = data_cfg.get("math", {})
        code_data: Dict[str, Any] = data_cfg.get("code", {})

        math_dataset_name: str = math_data.get(
            "dataset_name", "hendrycks/competition_math"
        )
        num_test_problems: int = int(
            math_data.get("num_test_problems", 500)
        )
        num_extra_train_from_test: int = int(
            math_data.get("num_extra_train_from_test", 4500)
        )
        train_dataset_name: str = code_data.get(
            "train_dataset_name", "google-research-datasets/mbpp"
        )
        test_dataset_name: str = code_data.get(
            "test_dataset_name", "openai_humaneval"
        )
        mbpp_r_dataset_name: str = code_data.get(
            "mbpp_r_dataset_name", "mbpp_r"
        )

        # ------------------------------------------------------------------
        # Step 11: SFT baseline configuration
        # ------------------------------------------------------------------
        sft_cfg: Dict[str, Any] = d.get("sft_baselines", {})
        star_cfg: Dict[str, Any] = sft_cfg.get("star", {})
        pair_sft_cfg: Dict[str, Any] = sft_cfg.get("pair_sft", {})

        star_num_iterations: int = int(star_cfg.get("num_iterations", 3))
        star_include_correct_pairs: bool = bool(
            star_cfg.get("include_correct_pairs", False)
        )
        pair_sft_num_iterations: int = int(
            pair_sft_cfg.get("num_iterations", 1)
        )
        pair_sft_include_correct_pairs: bool = bool(
            pair_sft_cfg.get("include_correct_pairs", False)
        )

        # ------------------------------------------------------------------
        # Step 12: Validation
        # ------------------------------------------------------------------
        if alpha <= 0.0:
            raise ValueError(
                f"alpha must be positive (got {alpha}). Paper states: "
                "'α is a positive constant multiplier, ideally larger than 1.0'."
            )
        if beta2 <= beta1:
            raise ValueError(
                f"beta2 ({beta2}) must be greater than beta1 ({beta1}). "
                "Stage I first-turn KL penalty must be stricter than the "
                "standard KL penalty to enforce the decoupling constraint "
                "from Equation 3."
            )
        if not (0.0 <= offline_data_mix_ratio <= 1.0):
            raise ValueError(
                f"offline_data_mix_ratio must be in [0, 1] "
                f"(got {offline_data_mix_ratio})."
            )
        if gamma != 0.0:
            warnings.warn(
                f"gamma={gamma} is non-zero. Appendix A.2 of the paper "
                "explicitly shows that γ > 0 does not prevent behavior "
                "collapse and recommends γ = 0 (instantaneous reward only). "
                "Proceeding with the provided value.",
                UserWarning,
                stacklevel=2,
            )

        # ------------------------------------------------------------------
        # Step 13: Construct and return the Config instance
        # ------------------------------------------------------------------
        return cls(
            task=task,
            run_name=run_name,
            output_dir=output_dir,
            seed=seed,
            wandb_project=wandb_project,
            log_level=log_level,
            model_name=model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            max_new_tokens=max_new_tokens,
            eval_temperature=eval_temperature,
            sampling_temperature=sampling_temperature,
            scaling_temperature=scaling_temperature,
            learning_rate=learning_rate,
            stage1_steps=stage1_steps,
            stage2_steps=stage2_steps,
            batch_size=batch_size,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_epsilon=adam_epsilon,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            gradient_accumulation_steps=gradient_accumulation_steps,
            alpha=alpha,
            beta1=beta1,
            beta2=beta2,
            normalize_rewards=normalize_rewards,
            reward_norm_eps=reward_norm_eps,
            offline_data_mix_ratio=offline_data_mix_ratio,
            gamma=gamma,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha_val,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            eval_every_n_steps=eval_every_n_steps,
            save_every_n_steps=save_every_n_steps,
            deepspeed_config=deepspeed_config,
            use_sympy=use_sympy,
            execution_timeout=execution_timeout,
            max_memory_mb=max_memory_mb,
            math_dataset_name=math_dataset_name,
            num_test_problems=num_test_problems,
            num_extra_train_from_test=num_extra_train_from_test,
            train_dataset_name=train_dataset_name,
            test_dataset_name=test_dataset_name,
            mbpp_r_dataset_name=mbpp_r_dataset_name,
            star_num_iterations=star_num_iterations,
            star_include_correct_pairs=star_include_correct_pairs,
            pair_sft_num_iterations=pair_sft_num_iterations,
            pair_sft_include_correct_pairs=pair_sft_include_correct_pairs,
            inference_scaling_enabled=inference_scaling_enabled,
            inference_scaling_solution_budget=inference_scaling_solution_budget,
            inference_scaling_k_values=inference_scaling_k_values,
            multi_attempt_enabled=multi_attempt_enabled,
            multi_attempt_num_attempts=multi_attempt_num_attempts,
            edit_distance_analysis_enabled=edit_distance_analysis_enabled,
            edit_distance_no_edit_threshold=edit_distance_no_edit_threshold,
            edit_distance_large_edit_threshold=edit_distance_large_edit_threshold,
            checkpoint_metric=checkpoint_metric,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the Config to a plain Python dictionary.

        Used by LoggingUtils to log the full config to wandb at run start,
        and by CheckpointUtils to save config alongside model checkpoints
        for reproducibility.

        Returns:
            A flat dictionary mapping field names to their values. List fields
            (e.g., lora_target_modules) are preserved as lists.
        """
        return {
            # Identity
            "task": self.task,
            "run_name": self.run_name,
            "output_dir": self.output_dir,
            "seed": self.seed,
            "wandb_project": self.wandb_project,
            "log_level": self.log_level,
            # Model
            "model_name": self.model_name,
            "torch_dtype": self.torch_dtype,
            "device_map": self.device_map,
            "max_new_tokens": self.max_new_tokens,
            "eval_temperature": self.eval_temperature,
            "sampling_temperature": self.sampling_temperature,
            "scaling_temperature": self.scaling_temperature,
            # Training
            "learning_rate": self.learning_rate,
            "stage1_steps": self.stage1_steps,
            "stage2_steps": self.stage2_steps,
            "batch_size": self.batch_size,
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "adam_epsilon": self.adam_epsilon,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            # SCoRe hyperparameters
            "alpha": self.alpha,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "normalize_rewards": self.normalize_rewards,
            "reward_norm_eps": self.reward_norm_eps,
            "offline_data_mix_ratio": self.offline_data_mix_ratio,
            "gamma": self.gamma,
            # LoRA
            "use_lora": self.use_lora,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": list(self.lora_target_modules),
            # Evaluation and checkpointing
            "eval_every_n_steps": self.eval_every_n_steps,
            "save_every_n_steps": self.save_every_n_steps,
            "deepspeed_config": self.deepspeed_config,
            "checkpoint_metric": self.checkpoint_metric,
            # Reward function
            "use_sympy": self.use_sympy,
            "execution_timeout": self.execution_timeout,
            "max_memory_mb": self.max_memory_mb,
            # Dataset
            "math_dataset_name": self.math_dataset_name,
            "num_test_problems": self.num_test_problems,
            "num_extra_train_from_test": self.num_extra_train_from_test,
            "train_dataset_name": self.train_dataset_
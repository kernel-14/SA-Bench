```python
## lora_sb/model_builder.py
"""Model builder for LoRA-SB and all baseline methods.

This module implements ModelBuilder, the central factory class responsible for:
1. Loading pre-trained models and tokenizers in bfloat16 precision
2. Constructing all method variants: LoRA-SB, LoRA-XS, LoRA, PiSSA, rsLoRA,
   DoRA, LoRA-Pro, and Full FT
3. Providing utilities for layer name discovery and trainable parameter counting

The two replacement strategies used are:
- Custom layer-by-layer replacement: LoRA-SB, LoRA-XS, LoRA-Pro
  (PEFT lacks native support for frozen R between B and A)
- PEFT-based wrapping: LoRA, PiSSA, rsLoRA, DoRA
  (standard adapters fully supported by the PEFT library)

References:
    Paper Section 3: Experimental setup and model configurations
    Paper Appendix H: Hyperparameter settings (bfloat16, target modules)
    config.yaml: hardware.precision: bfloat16, target_modules per experiment
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from peft import LoraConfig, TaskType, get_peft_model

from config import Config
from lora_sb.module import LoRASBLinear
from baselines.lora_xs import LoRAXSLinear
from baselines.lora_pro import LoRAProLinear

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GLUE task label counts (required for AutoModelForSequenceClassification)
# ---------------------------------------------------------------------------
_GLUE_NUM_LABELS: Dict[str, int] = {
    "cola": 2,
    "sst2": 2,
    "mrpc": 2,
    "stsb": 1,   # regression task
    "qqp": 2,
    "mnli": 3,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
}

# Default number of labels for unknown GLUE tasks
_DEFAULT_GLUE_NUM_LABELS: int = 2

# PEFT TaskType mapping from config task strings
_TASK_TO_PEFT_TASK_TYPE: Dict[str, TaskType] = {
    "math": TaskType.CAUSAL_LM,
    "commonsense": TaskType.CAUSAL_LM,
    "glue": TaskType.SEQ_CLS,
}


class ModelBuilder:
    """Factory class for loading models and constructing all fine-tuning variants.

    Provides a unified interface for building any of the eight method variants
    evaluated in the paper: LoRA-SB, LoRA-XS, LoRA, PiSSA, rsLoRA, DoRA,
    LoRA-Pro, and Full FT. Each build_* method receives a base_model and
    returns a model with the appropriate adapter structure and parameter
    freezing applied.

    The builder is stateless with respect to model weights — each build_*
    method receives base_model as an argument, allowing the same ModelBuilder
    instance to be reused across seeds without re-instantiation.

    Attributes:
        config: Experiment configuration. Provides rank, target_modules,
            scaling, dropout, task, model_name, and all other hyperparameters.
        device: Target compute device. Used for tensor placement during
            initialization data assignment.

    Example:
        >>> builder = ModelBuilder(config, device)
        >>> base_model, tokenizer = builder.load_base_model()
        >>> # For LoRA-SB:
        >>> delta_w_dict = gradient_estimator.estimate(init_dataloader)
        >>> init_data = {name: initializer.initialize(dw)
        ...              for name, dw in delta_w_dict.items()}
        >>> model = builder.build_lora_sb(base_model, init_data)
        >>> print(builder.count_trainable_params(model))  # r² × num_layers
    """

    def __init__(self, config: Config, device: torch.device) -> None:
        """Initialize ModelBuilder with configuration and target device.

        No model loading or computation happens here. All work is deferred
        to the load_base_model() and build_* methods.

        Args:
            config: Experiment configuration. Key fields used across methods:
                - config.model_name: HuggingFace model identifier
                - config.task: 'math', 'commonsense', or 'glue'
                - config.rank: LoRA rank r
                - config.scaling: Scaling factor s (1.0 for LoRA-SB)
                - config.target_modules: Layer name patterns to replace
                - config.dropout: LoRA dropout probability
                - config.alpha: Alpha for LoRA-XS and LoRA baselines
                - config.gradient_checkpointing: Whether to enable grad ckpt
                - config.precision: Model dtype ('bfloat16')
                - config.dataset_name: Used for GLUE num_labels determination
                - config.svd_niter: Power iterations for svd_lowrank
            device: Compute device for tensor operations. Sourced from
                utils.seed_utils.get_device(). Used when assigning init_data
                tensors to LoRASBLinear parameters.
        """
        self.config: Config = config
        self.device: torch.device = device

    # -----------------------------------------------------------------------
    # Model and tokenizer loading
    # -----------------------------------------------------------------------

    def load_base_model(self) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Load the pre-trained model and tokenizer in bfloat16 precision.

        Selects the appropriate HuggingFace model class based on config.task:
        - 'glue': AutoModelForSequenceClassification (RoBERTa-large)
        - 'math', 'commonsense': AutoModelForCausalLM (Mistral, Gemma, Llama)

        All models are loaded in torch.bfloat16 as specified in Appendix H.
        Gradient checkpointing is enabled for non-GLUE models to fit 7B/9B
        models on a single A6000 GPU (config.yaml: hardware.gradient_checkpointing).

        Returns:
            A tuple (model, tokenizer) where:
                - model: Pre-trained model in bfloat16, on CPU initially.
                  Move to device after layer replacement to avoid double-loading.
                - tokenizer: Corresponding tokenizer with pad_token set.

        Note:
            The model is returned on CPU. The caller (ExperimentRunner) should
            move it to the target device after build_* completes, to avoid
            holding two copies of the model in GPU memory simultaneously.
        """
        model_name: str = self.config.model_name
        precision_map: Dict[str, torch.dtype] = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype: torch.dtype = precision_map.get(
            self.config.precision, torch.bfloat16
        )

        logger.info(
            "Loading base model '%s' in %s precision for task '%s'.",
            model_name, self.config.precision, self.config.task,
        )

        # -----------------------------------------------------------------------
        # Load tokenizer
        # -----------------------------------------------------------------------
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        # For causal LM models (Llama, Mistral, Gemma), pad_token is often
        # not set. Set it to eos_token to enable batched training.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logger.info(
                "Set pad_token = eos_token ('%s') for model '%s'.",
                tokenizer.eos_token, model_name,
            )

        # Right-padding for causal LM training (labels align with input_ids).
        # Left-padding for generation (common practice), but we use right for training.
        if self.config.task in ("math", "commonsense"):
            tokenizer.padding_side = "right"

        # -----------------------------------------------------------------------
        # Load model
        # -----------------------------------------------------------------------
        if self.config.task == "glue":
            # Determine number of labels for the specific GLUE sub-task.
            # config.dataset_name for GLUE is the sub-task name (e.g., 'cola').
            num_labels: int = _GLUE_NUM_LABELS.get(
                self.config.dataset_name.lower(),
                _DEFAULT_GLUE_NUM_LABELS,
            )
            logger.info(
                "Loading sequence classification model for GLUE task '%s' "
                "with num_labels=%d.",
                self.config.dataset_name, num_labels,
            )
            model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
        else:
            # Causal LM for math and commonsense reasoning tasks.
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )

        # -----------------------------------------------------------------------
        # Enable gradient checkpointing for large models (7B/9B).
        # Required to fit on a single A6000 GPU (config.yaml:
        # hardware.gradient_checkpointing: true).
        # Not applied to RoBERTa-large (355M) which fits without it.
        # -----------------------------------------------------------------------
        if self.config.gradient_checkpointing and self.config.task != "glue":
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                logger.info(
                    "Enabled gradient checkpointing for model '%s'.", model_name
                )
            else:
                logger.warning(
                    "Model '%s' does not support gradient_checkpointing_enable(). "
                    "Skipping gradient checkpointing.",
                    model_name,
                )

        # Freeze all parameters initially. build_* methods selectively unfreeze.
        for param in model.parameters():
            param.requires_grad_(False)

        trainable_before: int = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        total_params: int = sum(p.numel() for p in model.parameters())
        logger.info(
            "Loaded model '%s': total_params=%d, trainable_params=%d (all frozen).",
            model_name, total_params, trainable_before,
        )

        return model, tokenizer

    # -----------------------------------------------------------------------
    # LoRA-SB model construction
    # -----------------------------------------------------------------------

    def build_lora_sb(
        self,
        base_model: nn.Module,
        init_data: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> nn.Module:
        """Build a LoRA-SB model by replacing target layers and applying SVD init.

        Replaces target nn.Linear layers with LoRASBLinear instances, then
        populates each layer's lora_B, lora_A, lora_R from the pre-computed
        init_data dictionary produced by LoRASBInitializer.

        After this call:
        - Only lora_R parameters have requires_grad=True
        - lora_B and lora_A are frozen (requires_grad=False)
        - All base model weights are frozen (requires_grad=False)
        - Trainable parameter count = rank² × num_replaced_layers

        Args:
            base_model: Pre-trained model with all parameters frozen.
                Must be the same model used by GradientEstimator to produce
                init_data (same architecture, same layer names).
            init_data: Dict mapping full dotted layer name to (B_init, R_init,
                A_init) tuple. Keys must match the output of
                _get_target_layer_names(). Values are tensors from
                LoRASBInitializer.initialize():
                - B_init: shape (out_features, rank), orthonormal columns
                - R_init: shape (rank, rank), diagonal singular values
                - A_init: shape (rank, in_features), orthonormal rows

        Returns:
            The modified base_model with LoRASBLinear layers replacing target
            nn.Linear layers. Only lora_R parameters are trainable.

        Raises:
            ValueError: If init_data is empty or contains no matching layers.
            RuntimeError: If a layer name in init_data cannot be found in the
                replaced model (indicates name mismatch between GradientEstimator
                and ModelBuilder).
        """
        if not init_data:
            raise ValueError(
                "init_data is empty. LoRA-SB requires pre-computed initialization "
                "from GradientEstimator and LoRASBInitializer."
            )

        # -----------------------------------------------------------------------
        # Step 1: Replace target nn.Linear layers with LoRASBLinear instances.
        # At this point, lora_B, lora_A, lora_R are all zeros (placeholder).
        # -----------------------------------------------------------------------
        model: nn.Module = self._replace_linear(
            base_model,
            LoRASBLinear,
            self.config.target_modules,
            rank=self.config.rank,
            scaling=self.config.scaling,
        )

        # -----------------------------------------------------------------------
        # Step 2: Validate that init_data keys match replaced layer names.
        # -----------------------------------------------------------------------
        replaced_names: List[str] = self._get_target_layer_names(
            model, self.config.target_modules
        )
        replaced_name_set = set(replaced_names)
        init_data_keys = set(init_data.keys())

        missing_in_init = replaced_name_set - init_data_keys
        extra_in_init = init_data_keys - replaced_name_set

        if missing_in_init:
            logger.warning(
                "The following replaced layers have no init_data entry "
                "(will keep zero initialization): %s",
                sorted(missing_in_init),
            )
        if extra_in_init:
            logger.warning(
                "The following init_data keys do not match any replaced layer "
                "(will be ignored): %s",
                sorted(extra_in_init),
            )

        # -----------------------------------------------------------------------
        # Step 3: Populate lora_B, lora_A, lora_R from init_data.
        # -----------------------------------------------------------------------
        layers_initialized: int = 0
        for layer_name, (B_init, R_init, A_init) in init_data.items():
            # Navigate to the LoRASBLinear module using the dotted path.
            target_module: Optional[nn.Module] = self._get_module_by_name(
                model, layer_name
            )

            if target_module is None:
                logger.warning(
                    "Layer '%s' from init_data not found in model after replacement. "
                    "Skipping initialization for this layer.",
                    layer_name,
                )
                continue

            if not isinstance(target_module, LoRASBLinear):
                logger.warning(
                    "Layer '%s' is not a LoRASBLinear instance (got %s). "
                    "Skipping initialization.",
                    layer_name, type(target_module).__name__,
                )
                continue

            # Determine the model's dtype for consistent tensor placement.
            model_dtype: torch.dtype = self._get_model_dtype(model)

            # Assign B_init: shape (out_features, rank)
            # Cast to model dtype (bfloat16) and move to device.
            target_module.lora_B.data.copy_(
                B_init.to(dtype=model_dtype, device=self.device)
            )

            # Assign A_init: shape (rank, in_features)
            target_module.lora_A.data.copy_(
                A_init.to(dtype=model_dtype, device=self.device)
            )

            # Assign R_init: shape (rank, rank)
            target_module.lora_R.data.copy_(
                R_init.to(dtype=model_dtype, device=self.device)
            )

            layers_initialized += 1

            logger.debug(
                "Initialized LoRA-SB layer '%s': "
                "B_init %s, R_init %s, A_init %s",
                layer_name,
                tuple(B_init.shape),
                tuple(R_init.shape),
                tuple(A_init.shape),
            )

        logger.info(
            "LoRA-SB initialization complete: %d/%d layers initialized from init_data.",
            layers_initialized, len(replaced_names),
        )

        # -----------------------------------------------------------------------
        # Step 4: Enforce parameter freezing.
        # Freeze everything, then unfreeze only lora_R in LoRASBLinear layers.
        # -----------------------------------------------------------------------
        for param in model.parameters():
            param.requires_grad_(False)

        num_trainable_layers: int = 0
        for module in model.modules():
            if isinstance(module, LoRASBLinear):
                module.lora_B.requires_grad_(False)
                module.lora_A.requires_grad_(False)
                module.lora_R.requires_grad_(True)
                num_trainable_layers += 1

        trainable_params: int = self.count_trainable_params(model)
        if trainable_params == 0:
            raise RuntimeError(
                "LoRA-SB model has 0 trainable parameters after build_lora_sb(). "
                "This indicates a bug in the freezing logic. "
                f"Expected {self.config.rank ** 2 * num_trainable_layers} "
                f"(rank²={self.config.rank ** 2} × {num_trainable_layers} layers)."
            )

        logger.info(
            "LoRA-SB model built: %d trainable params (rank²=%d × %d layers). "
            "Expected ~%.2fM params.",
            trainable_params,
            self.config.rank ** 2,
            num_trainable_layers,
            trainable_params / 1e6,
        )

        return model

    # -----------------------------------------------------------------------
    # LoRA-XS model construction
    # -----------------------------------------------------------------------

    def build_lora_xs(self, base_model: nn.Module) -> nn.Module:
        """Build a LoRA-XS model with PiSSA-style initialization on W₀.

        Replaces target nn.Linear layers with LoRAXSLinear instances, then
        initializes B and A via truncated SVD of the pre-trained weight W₀
        (not ΔW_avg like LoRA-SB). R is initialized to zeros.

        Key difference from LoRA-SB:
        - LoRA-XS: B, A from SVD(W₀) — pre-trained weight subspace
        - LoRA-SB: B, A from SVD(ΔW_avg) — task-relevant subspace

        Scaling: s = alpha / rank.
        - LLM experiments: alpha = rank → s = 1.0
        - RoBERTa/GLUE: alpha = 16 (fixed, config.yaml: roberta_glue.lora_xs.alpha: 16)

        Args:
            base_model: Pre-trained model with all parameters frozen.

        Returns:
            The modified base_model with LoRAXSLinear layers replacing target
            nn.Linear layers. Only lora_R parameters are trainable.
        """
        # -----------------------------------------------------------------------
        # Determine alpha for LoRA-XS scaling.
        # For GLUE/RoBERTa: alpha=16 (fixed, from config.yaml roberta_glue.lora_xs.alpha: 16)
        # For LLM tasks: alpha=rank (alpha_equals_rank: true)
        # -----------------------------------------------------------------------
        if self.config.task == "glue":
            # RoBERTa GLUE: alpha=16 fixed (Appendix H: "fixed α at 16 for their baseline")
            alpha: float = 16.0
        else:
            # LLM tasks: alpha=rank (Appendix H: "set α = r for their baseline configuration")
            alpha = float(self.config.rank)

        logger.info(
            "Building LoRA-XS model: rank=%d, alpha=%.1f, scaling=%.4f, "
            "target_modules=%s",
            self.config.rank, alpha, alpha / self.config.rank,
            self.config.target_modules,
        )

        # -----------------------------------------------------------------------
        # Step 1: Replace target nn.Linear layers with LoRAXSLinear instances.
        # -----------------------------------------------------------------------
        model: nn.Module = self._replace_linear(
            base_model,
            LoRAXSLinear,
            self.config.target_modules,
            rank=self.config.rank,
            alpha=alpha,
            svd_niter=self.config.svd_niter,
        )

        # -----------------------------------------------------------------------
        # Step 2: Initialize B and A via SVD of W₀ for each replaced layer.
        # -----------------------------------------------------------------------
        layers_initialized: int = 0
        for full_name, module in model.named_modules():
            if not isinstance(module, LoRAXSLinear):
                continue

            # _init_pissa_style computes SVD of W₀ and sets lora_B, lora_A.
            # lora_R is set to zeros (starts with no adaptation).
            module._init_pissa_style(module.weight.data)
            layers_initialized += 1

            logger.debug(
                "LoRA-XS PiSSA init for layer '%s': "
                "B %s, A %s, R zeros %s",
                full_name,
                tuple(module.lora_B.shape),
                tuple(module.lora_A.shape),
                tuple(module.lora_R.shape),
            )

        logger.info(
            "LoRA-XS PiSSA initialization complete: %d layers initialized.",
            layers_initialized,
        )

        # -----------------------------------------------------------------------
        # Step 3: Enforce parameter freezing.
        # Freeze everything, then unfreeze only lora_R in LoRAXSLinear layers.
        # -----------------------------------------------------------------------
        for param in model.parameters():
            param.requires_grad_(False)

        for module in model.modules():
            if isinstance(module, LoRAXSLinear):
                module.lora_B.requires_grad_(False)
                module.lora_A.requires_grad_(False)
                module.lora_R.requires_grad_(True)

        trainable_params: int = self.count_trainable_params(model)
        logger.info(
            "LoRA-XS model built: %d trainable params (~%.2fM).",
            trainable_params, trainable_params / 1e6,
        )

        return model

    # -----------------------------------------------------------------------
    # PEFT-based baseline model construction
    # -----------------------------------------------------------------------

    def build_lora(self, base_model: nn.Module) -> nn.Module:
        """Build a standard LoRA model using the PEFT library.

        Standard LoRA parameterizes the weight update as W = W₀ + s·B·A
        where B is Kaiming-uniform initialized and A is zero-initialized.
        Both B and A are trainable. Scaling s = alpha/rank = 1.0 (alpha=rank).

        Trainable parameters per layer: rank * (in_features + out_features).
        Total (Mistral-7B, rank=32): ~83.88M (Table 1).

        Args:
            base_model: Pre-trained model. PEFT will handle parameter freezing.

        Returns:
            A PEFT-wrapped model with LoRA adapters on target_modules.
            Only lora_A and lora_B parameters are trainable.
        """
        task_type: TaskType = _TASK_TO_PEFT_TASK_TYPE.get(
            self.config.task, TaskType.CAUSAL_LM
        )

        lora_config: LoraConfig = LoraConfig(
            r=self.config.rank,
            lora_alpha=self.config.rank,   # alpha = rank → s = 1.0
            target_modules=self.config.target_modules,
            lora_dropout=self.config.dropout,
            bias="none",
            task_type=task_type,
            init_lora_weights=True,        # Gaussian B, zeros A
            use_rslora=False,
            use_dora=False,
        )

        model: nn.Module = get_peft_model(base_model, lora_config)
        trainable_params: int = self.count_trainable_params(model)

        logger.info(
            "Standard LoRA model built: rank=%d, alpha=%d, "
            "trainable_params=%d (~%.2fM).",
            self.config.rank, self.config.rank,
            trainable_params, trainable_params / 1e6,
        )

        return model

    def build_pissa(self, base_model: nn.Module) -> nn.Module:
        """Build a PiSSA model using the PEFT library.

        PiSSA initializes B and A with principal singular vectors of W₀,
        capturing the most significant pre-trained weight subspaces.
        Both B and A are trainable (same parameter count as standard LoRA).

        Args:
            base_model: Pre-trained model. PEFT handles parameter freezing.

        Returns:
            A PEFT-wrapped model with PiSSA-initialized LoRA adapters.
        """
        task_type: TaskType = _TASK_TO_PEFT_TASK_TYPE.get(
            self.config.task, TaskType.CAUSAL_LM
        )

        lora_config: LoraConfig = LoraConfig(
            r=self.config.rank,
            lora_alpha=self.config.rank,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.dropout,
            bias="none",
            task_type=task_type,
            init_lora_weights="pissa",     # SVD of W₀ initialization
            use_rslora=False,
            use_dora=False,
        )

        model: nn.Module = get_peft_model(base_model, lora_config)
        trainable_params: int = self.count_trainable_params(model)

        logger.info(
            "PiSSA model built: rank=%d, alpha=%d, "
            "trainable_params=%d (~%.2fM).",
            self.config.rank, self.config.rank,
            trainable_params, trainable_params / 1e6,
        )

        return model

    def build_rslora(self, base_model: nn.Module) -> nn.Module:
        """Build an rsLoRA model using the PEFT library.

        rsLoRA uses rank-stabilized scaling s = alpha/sqrt(rank) instead of
        alpha/rank, providing more stable training at higher ranks.
        With alpha=rank: s = rank/sqrt(rank) = sqrt(rank).

        Args:
            base_model: Pre-trained model. PEFT handles parameter freezing.

        Returns:
            A PEFT-wrapped model with rsLoRA scaling.
        """
        task_type: TaskType = _TASK_TO_PEFT_TASK_TYPE.get(
            self.config.task, TaskType.CAUSAL_LM
        )

        lora_config: LoraConfig = LoraConfig(
            r=self.config.rank,
            lora_alpha=self.config.rank,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.dropout,
            bias="none",
            task_type=task_type,
            init_lora_weights=True,
            use_rslora=True,               # s = alpha/sqrt(rank)
            use_dora=False,
        )

        model: nn.Module = get_peft_model(base_model, lora_config)
        trainable_params: int = self.count_trainable_params(model)

        import math
        effective_scale: float = self.config.rank / math.sqrt(self.config.rank)
        logger.info(
            "rsLoRA model built: rank=%d, alpha=%d, effective_scale=%.4f, "
            "trainable_params=%d (~%.2fM).",
            self.config.rank, self.config.rank, effective_scale,
            trainable_params, trainable_params / 1e6,
        )

        return model

    def build_dora(self, base_model: nn.Module) -> nn.Module:
        """Build a DoRA model using the PEFT library.

        DoRA decomposes weights into magnitude and direction components,
        applying LoRA-style adaptation to the direction while learning a
        separate magnitude scaling vector. Slightly more parameters than LoRA.

        Args:
            base_model: Pre-trained model. PEFT handles parameter freezing.

        Returns:
            A PEFT-wrapped model with DoRA weight decomposition.
        """
        task_type: TaskType = _TASK_TO_PEFT_TASK_TYPE.get(
            self.config.task, TaskType.CAUSAL_LM
        )

        lora_config: LoraConfig = LoraConfig(
            r=self.config.rank,
            lora_alpha=self.config.rank,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.dropout,
            bias="none",
            task_type=task_type,
            init_lora_weights=True,
            use_rslora=False,
            use_dora=True,                 # magnitude + direction decomposition
        )

        model: nn.Module = get_peft_model(base_model, lora_config)
        trainable_params: int = self.count_trainable_params(model)

        logger.info(
            "DoRA model built: rank=%d, alpha=%d, "
            "trainable_params=%d (~%.2fM).",
            self.config.rank, self.config.rank,
            trainable_params, trainable_params / 1e6,
        )

        return model

    # -----------------------------------------------------------------------
    # LoRA-Pro model construction
    # -----------------------------------------------------------------------

    def build_lora_pro(self, base_model: nn.Module) -> nn.Module:
        """Build a LoRA-Pro model with optimal gradient correction hooks.

        LoRA-Pro uses standard LoRA architecture (both B and A trainable) but
        applies closed-form gradient corrections at each backward pass to
        minimize the discrepancy between the low-rank equivalent gradient and
        the full fine-tuning gradient.

        The gradient correction is applied via backward hooks registered on
        lora_A and lora_B in each LoRAProLinear layer. The AdamW optimizer
        sees corrected gradients automatically.

        Scaling: s = alpha / rank = rank / rank = 1.0 (alpha = rank per
        config.yaml: baselines.lora_pro.alpha_equals_rank: true).

        Args:
            base_model: Pre-trained model with all parameters frozen.

        Returns:
            The modified base_model with LoRAProLinear layers replacing target
            nn.Linear layers. lora_A and lora_B are trainable
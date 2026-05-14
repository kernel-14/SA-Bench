## lora_sb/initializer_gradient.py
"""Memory-efficient gradient estimator for LoRA-SB initialization.

This module implements GradientEstimator, which computes the averaged gradient
approximation ΔW_avg used to initialize LoRA-SB's frozen matrices B and A.

The core formula from Section 2.4 and Appendix C of the paper is:

    ΔW_avg = -η · sign( Σᵢ ∇_W L(W₀, xᵢ) )

This approximates AdamW's first update step, where both moment estimates are
zero and the update direction reduces to sign(g₁). The sign is applied to the
**sum** of gradients across all n samples (not per-sample), which is critical
for correctness.

Memory efficiency is achieved via backward hooks that accumulate gradients
layer-by-layer and immediately discard them from the computation graph,
ensuring O(1) memory usage independent of model depth (Section 2.6, ref 29, 45).

References:
    Paper Section 2.4: Initialization using update approximation
    Paper Section 2.6: Initialization memory (O(1) via layerwise hooks)
    Paper Appendix C: Simulating first step of full FT under AdamW
    config.yaml: initialization.num_init_samples: 50 (0.1% of 50K MetaMathQA)
    config.yaml: lora_sb.layerwise_grad_hooks: true
    config.yaml: lora_sb.use_sign_gradient: true
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from config import Config

logger = logging.getLogger(__name__)


class GradientEstimator:
    """Estimates ΔW_avg = -sign(Σᵢ ∇_W L(W₀, xᵢ)) for LoRA-SB initialization.

    Uses backward hooks on target weight parameters to accumulate raw gradient
    sums across n initialization samples. After accumulation, applies the sign
    operation once to produce the averaged gradient approximation that mimics
    AdamW's first update step.

    The hook-based approach ensures O(1) memory usage: each hook captures the
    current layer's gradient, adds it to the running sum, and returns None to
    discard the gradient from the computation graph. At any point during the
    backward pass, only the current layer's gradient is in memory beyond the
    accumulated sums (which are the same size as the weight matrices already
    in GPU memory).

    Attributes:
        model: The base pre-trained model. Weights are temporarily enabled for
            gradient computation during estimation, then re-frozen afterward.
        config: Experiment configuration. Key fields: target_modules,
            num_init_samples, rank.
        device: Compute device for tensor placement.
        _hooks: List of RemovableHook handles from register_hook() calls.
            Populated by _register_hooks(), cleared by _remove_hooks().
        _grad_accum: Dict mapping full dotted layer name to accumulated raw
            gradient sum tensor (float32 for numerical stability). Populated
            by _accumulate_grad() during backward passes.

    Example:
        >>> estimator = GradientEstimator(model, config, device)
        >>> delta_w_dict = estimator.estimate(init_dataloader)
        >>> # delta_w_dict maps layer names to ΔW_avg tensors
        >>> # e.g., {"model.layers.0.self_attn.q_proj": Tensor(4096, 4096)}
    """

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        device: torch.device,
    ) -> None:
        """Initialize GradientEstimator.

        No computation happens here. All gradient estimation is deferred to
        the estimate() call.

        Args:
            model: The base pre-trained model with frozen weights. Must be
                loaded in the target precision (bfloat16 per config.yaml:
                hardware.precision). Weights will be temporarily unfrozen
                during estimate() and re-frozen afterward.
            config: Experiment configuration. Key fields used:
                - config.target_modules: List of layer name patterns to hook
                  (e.g., ['q_proj', 'k_proj', 'v_proj', ...]).
                - config.num_init_samples: Number of samples for gradient
                  estimation (50 for MetaMathQA, from config.yaml:
                  initialization.num_init_samples).
                - config.rank: LoRA rank (used for logging context).
            device: Compute device. Should match the device of model parameters.
                Sourced from utils.seed_utils.get_device().
        """
        self.model: nn.Module = model
        self.config: Config = config
        self.device: torch.device = device

        # Hook handles returned by register_hook(); populated in _register_hooks(),
        # cleared in _remove_hooks(). Each handle corresponds to one weight parameter.
        self._hooks: List[torch.utils.hooks.RemovableHook] = []

        # Accumulated raw gradient sums, keyed by full dotted module path.
        # Values are float32 tensors (regardless of model precision) for
        # numerical stability when summing many bfloat16 gradients.
        # Populated by _accumulate_grad() during backward passes.
        self._grad_accum: Dict[str, Tensor] = {}

    def estimate(self, dataloader: DataLoader) -> Dict[str, Tensor]:
        """Compute ΔW_avg = -sign(Σᵢ ∇_W L(W₀, xᵢ)) for all target layers.

        This is the main entry point for LoRA-SB initialization. Runs n forward
        and backward passes over the initialization subset, accumulates raw
        gradient sums via hooks, then applies the sign operation once.

        The sequence follows Algorithm 1 (Appendix D) and Appendix C:
        1. Temporarily enable gradients on target weight parameters.
        2. Register backward hooks on target weight parameters.
        3. Run n forward+backward passes (no optimizer step).
        4. Apply sign to accumulated gradient sums → ΔW_avg per layer.
        5. Remove hooks and re-freeze all parameters.
        6. Free intermediate gradient memory.

        Args:
            dataloader: DataLoader over the initialization subset. Should have
                batch_size=1 and contain exactly config.num_init_samples batches
                (or more — we stop after num_init_samples). Each batch must
                contain 'input_ids', 'attention_mask', and 'labels' keys for
                causal LM loss computation. Produced by
                DatasetLoader.load_init_subset(config.num_init_samples).

        Returns:
            Dict mapping full dotted layer name (e.g.,
            "model.layers.0.self_attn.q_proj") to ΔW_avg tensor of shape
            (out_features, in_features). Tensors are in the model's original
            dtype (bfloat16) and on self.device. These are passed to
            LoRASBInitializer.initialize() per layer, then to
            ModelBuilder.build_lora_sb().

        Note:
            The model is left in eval mode with all parameters frozen after
            this call, identical to its state before the call.

        Note:
            Gradient checkpointing (if enabled on the model) is temporarily
            disabled during estimation to prevent hooks from firing multiple
            times per layer per backward call.
        """
        logger.info(
            "Starting gradient estimation for LoRA-SB initialization. "
            "Target modules: %s | num_init_samples: %d | rank: %d",
            self.config.target_modules,
            self.config.num_init_samples,
            self.config.rank,
        )

        # -----------------------------------------------------------------------
        # Track which parameters we temporarily enable for gradient computation.
        # We must re-freeze exactly these parameters after estimation.
        # -----------------------------------------------------------------------
        temporarily_enabled: List[str] = []

        # -----------------------------------------------------------------------
        # Track gradient checkpointing state to disable during estimation.
        # Gradient checkpointing causes activations to be recomputed during
        # backward, which can cause hooks to fire multiple times per layer.
        # -----------------------------------------------------------------------
        grad_ckpt_disabled_modules: List[nn.Module] = []

        try:
            # -------------------------------------------------------------------
            # Step 1: Temporarily enable gradients on target weight parameters.
            # Hooks only fire during backward if the parameter has requires_grad=True
            # and participates in the computation graph.
            # -------------------------------------------------------------------
            target_module_names: Set[str] = set(self.config.target_modules)
            for full_name, module in self.model.named_modules():
                if not isinstance(module, nn.Linear):
                    continue
                # Check if the module's short name matches any target pattern.
                # e.g., "model.layers.0.self_attn.q_proj" → short name "q_proj"
                short_name: str = full_name.split(".")[-1]
                if short_name not in target_module_names:
                    continue
                if not module.weight.requires_grad:
                    module.weight.requires_grad_(True)
                    temporarily_enabled.append(full_name)

            logger.info(
                "Temporarily enabled gradients on %d target weight parameters.",
                len(temporarily_enabled),
            )

            # -------------------------------------------------------------------
            # Step 2: Disable gradient checkpointing if active.
            # We check for the use_gradient_checkpointing attribute used by
            # HuggingFace models. This prevents hooks from firing multiple times.
            # -------------------------------------------------------------------
            for module in self.model.modules():
                if hasattr(module, "gradient_checkpointing") and module.gradient_checkpointing:
                    module.gradient_checkpointing = False
                    grad_ckpt_disabled_modules.append(module)

            if grad_ckpt_disabled_modules:
                logger.info(
                    "Temporarily disabled gradient checkpointing on %d modules "
                    "to prevent duplicate hook firings during initialization.",
                    len(grad_ckpt_disabled_modules),
                )

            # -------------------------------------------------------------------
            # Step 3: Register backward hooks on target weight parameters.
            # Must happen after requires_grad=True is set.
            # -------------------------------------------------------------------
            self._register_hooks(self.config.target_modules)
            logger.info("Registered %d gradient hooks.", len(self._hooks))

            # -------------------------------------------------------------------
            # Step 4: Set model to eval mode.
            # Disables dropout and batch norm training behavior, ensuring the
            # gradient signal reflects the pre-trained model's forward pass
            # without stochastic perturbations.
            # -------------------------------------------------------------------
            self.model.eval()

            # -------------------------------------------------------------------
            # Step 5: Gradient accumulation loop.
            # Run n forward+backward passes. No optimizer step — we only want
            # the gradient signal, not weight updates.
            # -------------------------------------------------------------------
            samples_processed: int = 0
            total_loss: float = 0.0

            for batch_idx, batch in enumerate(dataloader):
                if samples_processed >= self.config.num_init_samples:
                    break

                # Move batch to device
                batch = self._move_batch_to_device(batch)

                # Zero out any stale .grad tensors from previous iterations.
                # This prevents .grad buildup (memory waste) while our hook-based
                # accumulation in _grad_accum remains unaffected.
                self.model.zero_grad()

                # Forward + backward pass with gradient computation enabled.
                with torch.enable_grad():
                    # Extract model inputs from batch dict.
                    # The dataloader produces batches with 'input_ids', 'attention_mask',
                    # 'labels' for causal LM loss computation.
                    model_inputs = self._extract_model_inputs(batch)

                    # Forward pass: compute loss.
                    # For causal LM models (Mistral, Gemma, Llama), passing 'labels'
                    # triggers cross-entropy loss computation internally.
                    # For classification models (RoBERTa), 'labels' triggers
                    # cross-entropy over logits.
                    outputs = self.model(**model_inputs)

                    # Extract scalar loss.
                    loss: Tensor = self._extract_loss(outputs, batch)

                    # Backward pass: triggers hooks for each target weight parameter.
                    # Each hook call fires _accumulate_grad(name, grad), which adds
                    # grad.detach().float() to _grad_accum[name].
                    loss.backward()

                total_loss += loss.detach().item()
                samples_processed += 1

                if samples_processed % max(1, self.config.num_init_samples // 5) == 0:
                    logger.info(
                        "Gradient estimation progress: %d/%d samples | "
                        "avg loss: %.4f",
                        samples_processed,
                        self.config.num_init_samples,
                        total_loss / samples_processed,
                    )

            logger.info(
                "Gradient estimation complete: %d samples processed | "
                "avg loss: %.4f | layers accumulated: %d",
                samples_processed,
                total_loss / max(1, samples_processed),
                len(self._grad_accum),
            )

            if samples_processed == 0:
                raise RuntimeError(
                    "No samples were processed during gradient estimation. "
                    "Check that the initialization dataloader is non-empty."
                )

            if len(self._grad_accum) == 0:
                raise RuntimeError(
                    "No gradients were accumulated. Check that target_modules "
                    f"({self.config.target_modules}) match layer names in the model, "
                    "and that requires_grad was correctly enabled."
                )

            # -------------------------------------------------------------------
            # Step 6: Compute ΔW_avg = -sign(Σᵢ gradᵢ) for each layer.
            # -------------------------------------------------------------------
            delta_w_dict: Dict[str, Tensor] = self._compute_delta_w()

            logger.info(
                "Computed ΔW_avg for %d layers. "
                "Sample layer norms: %s",
                len(delta_w_dict),
                {
                    k: f"{v.float().norm().item():.2f}"
                    for k, v in list(delta_w_dict.items())[:3]
                },
            )

        finally:
            # -------------------------------------------------------------------
            # Cleanup: always execute regardless of success or exception.
            # -------------------------------------------------------------------

            # Remove all backward hooks.
            self._remove_hooks()

            # Re-freeze parameters that were temporarily enabled.
            for full_name in temporarily_enabled:
                # Navigate to the module using the dotted path.
                module = self._get_module_by_name(full_name)
                if module is not None and hasattr(module, "weight"):
                    module.weight.requires_grad_(False)

            logger.info(
                "Re-froze %d temporarily enabled weight parameters.",
                len(temporarily_enabled),
            )

            # Re-enable gradient checkpointing where it was disabled.
            for module in grad_ckpt_disabled_modules:
                module.gradient_checkpointing = True

            # Zero out .grad tensors to free memory.
            self.model.zero_grad()

            # Clear accumulated gradients to free GPU memory.
            self._grad_accum.clear()

            # Free GPU memory cache.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return delta_w_dict

    def _register_hooks(self, target_modules: List[str]) -> None:
        """Register backward hooks on target weight parameters.

        Iterates over all named modules in the model, identifies nn.Linear
        layers whose short name matches any pattern in target_modules, and
        registers a gradient hook on their weight parameter.

        The hook captures the gradient during backward, accumulates it into
        self._grad_accum, and returns None to avoid modifying the gradient
        or interfering with the computation graph.

        Args:
            target_modules: List of layer name patterns to match against the
                short name (last component) of each module's full dotted path.
                E.g., ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj',
                'up_proj', 'down_proj'] for LLMs, or ['query', 'key', 'value',
                'dense'] for RoBERTa (config.yaml: target_modules).

        Note:
            Uses `name=full_name` as a default argument in the lambda to
            create a new binding per iteration, avoiding Python closure
            capture-by-reference issues.

        Note:
            Hooks are only registered on parameters with requires_grad=True.
            This method must be called after _temporarily_enable_grads().
        """
        target_module_set: Set[str] = set(target_modules)
        hooks_registered: int = 0

        for full_name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            # Match against the short name (last component of dotted path).
            # E.g., "model.layers.0.self_attn.q_proj" → short name "q_proj".
            short_name: str = full_name.split(".")[-1]
            if short_name not in target_module_set:
                continue

            # Only register hooks on parameters that participate in the graph.
            if not module.weight.requires_grad:
                logger.warning(
                    "Skipping hook registration for '%s': weight.requires_grad=False. "
                    "Ensure _temporarily_enable_grads() was called first.",
                    full_name,
                )
                continue

            # Register hook with name captured as default argument to avoid
            # Python closure capture-by-reference issue in loops.
            # The hook returns None to avoid modifying the gradient.
            handle = module.weight.register_hook(
                lambda grad, name=full_name: self._accumulate_grad(name, grad)
            )
            self._hooks.append(handle)
            hooks_registered += 1

            logger.debug(
                "Registered gradient hook on '%s' (weight shape: %s).",
                full_name,
                tuple(module.weight.shape),
            )

        if hooks_registered == 0:
            logger.warning(
                "No gradient hooks were registered. "
                "target_modules=%s may not match any nn.Linear layer names. "
                "Check model architecture with model.named_modules().",
                target_modules,
            )

    def _remove_hooks(self) -> None:
        """Remove all registered backward hooks.

        Calls handle.remove() on each hook handle stored in self._hooks,
        then clears the list. This is called in the finally block of estimate()
        to ensure cleanup even if an exception occurs during gradient estimation.

        After this call, self._hooks is an empty list and no hooks remain
        registered on any model parameters.
        """
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        logger.debug("Removed all gradient hooks.")

    def _accumulate_grad(self, name: str, grad: Tensor) -> None:
        """Accumulate a gradient tensor into the running sum for a layer.

        Called by the backward hook registered in _register_hooks(). Adds the
        current gradient to the running sum in self._grad_accum[name].

        Accumulation is done in float32 regardless of the model's precision
        (bfloat16) to avoid numerical precision loss when summing many gradients.
        bfloat16 has only 7 mantissa bits, which can cause significant rounding
        errors when accumulating 50+ gradient tensors.

        The sign operation is NOT applied here — it is applied once to the
        final sum in _compute_delta_w(). This implements:
            ΔW_avg = -sign(Σᵢ gradᵢ)
        not:
            ΔW_avg = -Σᵢ sign(gradᵢ)
        which would be mathematically incorrect per Appendix C.

        Args:
            name: Full dotted module path (e.g., "model.layers.0.self_attn.q_proj").
                Used as the key in self._grad_accum.
            grad: Gradient tensor for the weight parameter, shape
                (out_features, in_features). May be in bfloat16 or float32.
                This is the raw gradient ∇_W L(W₀, xᵢ) for the current sample.

        Note:
            This method returns None implicitly. Per PyTorch hook semantics,
            returning None means the gradient is not modified and continues
            to flow normally through the computation graph. This is the
            "purely observational" hook pattern.

        Note:
            Uses grad.detach().float() to:
            1. .detach(): Remove from computation graph (prevents memory leak)
            2. .float(): Cast to float32 for numerical stability in accumulation
        """
        # Detach from computation graph and cast to float32 for stable accumulation.
        grad_f32: Tensor = grad.detach().float()

        if name not in self._grad_accum:
            # First sample for this layer: initialize accumulator with zeros.
            # Using zeros_like ensures correct shape and device placement.
            self._grad_accum[name] = torch.zeros_like(grad_f32)

        # Accumulate raw gradient sum (sign applied later in _compute_delta_w).
        self._grad_accum[name] += grad_f32

        # Return None implicitly — hook does not modify the gradient.

    def _compute_delta_w(self) -> Dict[str, Tensor]:
        """Apply sign to accumulated gradient sums to produce ΔW_avg.

        Implements the final step of the gradient approximation from Appendix C:

            ΔW_avg = -η · sign( Σᵢ ∇_W L(W₀, xᵢ) )

        The learning rate η is absorbed into the singular values S during SVD
        in LoRASBInitializer (it only scales S, not the direction captured by
        U and V). So we set η=1 here without loss of generality.

        The sign operation is applied to the accumulated sum (not per-sample),
        which is the correct implementation of AdamW's first-step behavior
        as derived in Appendix C.

        Returns:
            Dict mapping full dotted layer name to ΔW_avg tensor of shape
            (out_features, in_features). Tensors are cast to the model's
            original dtype (bfloat16 per config.yaml: hardware.precision)
            and placed on self.device.

        Note:
            torch.sign(0) = 0 for zero elements. This is acceptable — zero
            entries in ΔW_avg result in zero singular values in the SVD,
            which are naturally excluded from the rank-r approximation.
        """
        # Determine the target dtype from the model's parameters.
        # This is bfloat16 per config.yaml: hardware.precision: bfloat16.
        model_dtype: torch.dtype = self._get_model_dtype()

        delta_w_dict: Dict[str, Tensor] = {}

        for name, grad_sum in self._grad_accum.items():
            # Apply sign to the accumulated sum (not per-sample sign).
            # Multiply by -1.0 for gradient descent direction.
            # grad_sum is in float32 (accumulated in _accumulate_grad).
            delta_w_f32: Tensor = -1.0 * torch.sign(grad_sum)

            # Cast back to model dtype (bfloat16) and ensure correct device.
            delta_w_dict[name] = delta_w_f32.to(
                dtype=model_dtype,
                device=self.device,
            )

            logger.debug(
                "ΔW_avg for '%s': shape=%s, dtype=%s, "
                "nonzero_fraction=%.3f, norm=%.4f",
                name,
                tuple(delta_w_dict[name].shape),
                delta_w_dict[name].dtype,
                (delta_w_dict[name] != 0).float().mean().item(),
                delta_w_dict[name].float().norm().item(),
            )

        return delta_w_dict

    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move all tensor values in a batch dict to self.device.

        Args:
            batch: Dictionary of batch tensors from the DataLoader.
                May contain non-tensor values (e.g., strings for metadata)
                which are left unchanged.

        Returns:
            A new dict with all tensor values moved to self.device.
            Non-tensor values are copied unchanged.
        """
        moved: Dict = {}
        for key, value in batch.items():
            if isinstance(value, Tensor):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    def _extract_model_inputs(self, batch: Dict) -> Dict:
        """Extract model-compatible inputs from a batch dict.

        Filters the batch to only include keys that are valid model inputs,
        removing metadata keys (e.g., 'dataset_name', 'answer_text') that
        would cause errors if passed to model.forward().

        For causal LM models (Mistral, Gemma, Llama): passes 'input_ids',
        'attention_mask', 'labels'.
        For classification models (RoBERTa): passes 'input_ids',
        'attention_mask', 'token_type_ids' (if present), 'labels'.

        Args:
            batch: Full batch dict from the DataLoader, potentially containing
                extra metadata keys.

        Returns:
            Filtered dict containing only valid model input keys.
        """
        # Standard HuggingFace model input keys.
        # 'labels' is included to trigger loss computation in the model's
        # forward method (both causal LM and sequence classification).
        valid_keys: Set[str] = {
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "labels",
            "position_ids",
            "head_mask",
            "inputs_embeds",
        }
        return {k: v for k, v in batch.items() if k in valid_keys}

    def _extract_loss(self, outputs: object, batch: Dict) -> Tensor:
        """Extract the scalar loss from model outputs.

        HuggingFace models return a ModelOutput object with a .loss attribute
        when labels are provided. This method handles both this case and the
        fallback case where loss must be computed manually.

        Args:
            outputs: Model output object. Expected to have a .loss attribute
                (CausalLMOutputWithPast, SequenceClassifierOutput, etc.).
            batch: Original batch dict (used as fallback for manual loss
                computation if outputs.loss is None).

        Returns:
            Scalar loss tensor with gradient tracking enabled.

        Raises:
            RuntimeError: If no loss can be extracted from outputs and batch
                does not contain labels for manual computation.
        """
        # Primary path: HuggingFace models return loss when labels are provided.
        if hasattr(outputs, "loss") and outputs.loss is not None:
            return outputs.loss

        # Fallback: compute cross-entropy loss from logits manually.
        # This handles edge cases where the model doesn't compute loss internally.
        if hasattr(outputs, "logits"):
            logits: Tensor = outputs.logits
            if "labels" in batch:
                labels: Tensor = batch["labels"]
                # Flatten for cross-entropy: (batch * seq_len, vocab_size)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
                return loss

        raise RuntimeError(
            "Cannot extract loss from model outputs. "
            "Ensure the dataloader provides 'labels' in each batch, "
            "which triggers automatic loss computation in HuggingFace models."
        )

    def _get_model_dtype(self) -> torch.dtype:
        """Infer the model's parameter dtype from its first parameter.

        Used to cast ΔW_avg tensors back to the model's precision after
        float32 accumulation.

        Returns:
            The dtype of the first model parameter found. Defaults to
            torch.bfloat16 if no parameters are found (per config.yaml:
            hardware.precision: bfloat16).
        """
        for param in self.model.parameters():
            return param.dtype
        # Fallback: bfloat16 per config.yaml hardware.precision
        return torch.bfloat16

    def _get_module_by_name(self, full_name: str) -> Optional[nn.Module]:
        """Navigate to a submodule using its full dotted path.

        Used in the finally block of estimate() to re-freeze parameters
        that were temporarily enabled for gradient computation.

        Args:
            full_name: Full dotted module path, e.g.,
                "model.layers.0.self_attn.q_proj".

        Returns:
            The nn.Module at the specified path, or None if not found.
        """
        parts: List[str] = full_name.split(".")
        current: nn.Module = self.model

        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                # Try integer indexing for Sequential/ModuleList containers.
                try:
                    idx: int = int(part)
                    current = current[idx]  # type: ignore[index]
                except (ValueError, TypeError, KeyError, IndexError):
                    logger.debug(
                        "Could not navigate to module '%s': "
                        "part '%s' not found in %s.",
                        full_name, part, type(current).__name__,
                    )
                    return None

        return current if isinstance(current, nn.Module) else None

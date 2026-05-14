## Code: training/wise.py

```python
## training/wise.py
"""Weight-Space Ensembles (WiSE) for PEFT robustness experiments.

This module implements WiSE (Weight-Space Ensembles) for PEFT methods as
described in Section 7 of the paper:

    "Lessons and Insights from a Unifying Study of Parameter-Efficient
    Fine-Tuning (PEFT) in Visual Recognition"

Paper: "WiSE [96], which linearly interpolates the weights of a fully FT model
with those of the original backbone, is a popular approach for boosting
robustness. We explore whether WiSE can also enhance the robustness of PEFT."

The interpolation strategy differs per PEFT category:
- Adapter-based: scale adapter weights by α (feature ensemble interpretation)
- LoRA/FacT: scale additive residuals ΔW by α
- Selective (BitFit/LN/DiffFit/SSF): linear interpolation of modified params
- Full FT: standard WiSE (interpolate all parameters)
- Head: interpolate between fine-tuned head and zero-shot CLIP head

Config references (config.yaml):
    robustness.wise.alphas: [0.0, 0.1, 0.2, ..., 1.0]
    robustness.shift_datasets: [imagenet, imagenet_v2, imagenet_r, imagenet_s, imagenet_a]
    robustness.methods_evaluated: [full, bitfit, layernorm, houlsby_adapter, ...]

Typical usage (called by main.py for the robustness experiment):
    wise = WiSE(
        pretrained_state_dict=pretrained_state,
        finetuned_state_dict=finetuned_state,
        method='lora',
    )
    results = wise.sweep(
        model=peft_model,
        alphas=[0.0, 0.1, 0.2, ..., 1.0],
        target_loader=imagenet_test_loader,
        shift_loaders={'imagenet_v2': ..., 'imagenet_r': ..., ...},
        device='cuda',
    )
    # results is a List[dict] with keys: alpha, target_acc, imagenet_v2_acc, ...
"""

import copy
import logging
from typing import Any, Dict, List, Optional, Set

import torch
import torch.nn as nn

from evaluation.metrics import Metrics

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
_logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PEFT method category sets for interpolation dispatch.
# These must match the method name strings used in SUPPORTED_METHODS
# (models/peft_factory.py).
# ---------------------------------------------------------------------------

# Adapter-based methods: WiSE scales adapter weights by α.
# Paper: "For Adapter-based methods, WiSE can be considered feature ensembles,
# where α controls how strong the domain-specific features from the adapter
# module blend with the domain-agnostic ones from original backbones."
ADAPTER_METHODS: Set[str] = {
    "houlsby_adapter",
    "pfeiffer_adapter",
    "adaptformer",
    "convpass",
    "repadapter",
}

# Efficient selective methods: WiSE scales additive residuals ΔW by α.
# Paper: "For efficient selective methods (e.g. LoRA) as they learn additive
# residuals to the original parameters."
LORA_METHODS: Set[str] = {
    "lora",
    "fact_tt",
    "fact_tk",
}

# Direct selective tuning methods: WiSE linearly interpolates modified params.
# Paper: "For direct selective tuning methods (e.g. BitFit), this involves
# merging the PEFT-tuned parameters and the original model."
SELECTIVE_METHODS: Set[str] = {
    "bitfit",
    "layernorm",
    "difffit",
    "ssf",
}

# VPT methods: treated as selective (interpolate prompt tokens).
VPT_METHODS: Set[str] = {
    "vpt_shallow",
    "vpt_deep",
}

# Full fine-tuning: standard WiSE (interpolate all parameters).
FULL_FT_METHODS: Set[str] = {
    "full",
}

# Linear probing: only head differs; backbone is identical to pretrained.
LINEAR_METHODS: Set[str] = {
    "linear",
}

# ---------------------------------------------------------------------------
# Known adapter-related key substrings for parameter identification.
# Keys in finetuned_state containing these substrings belong to adapter modules.
# ---------------------------------------------------------------------------
_ADAPTER_KEY_SUBSTRINGS: List[str] = [
    "adapter",
    "W_down",
    "W_up",
    "convpass",
    "repadapter",
    "pfeiffer",
    "houlsby",
    "adaptformer",
    "down_proj",
    "up_proj",
]

# ---------------------------------------------------------------------------
# Known LoRA/FacT-related key substrings for parameter identification.
# ---------------------------------------------------------------------------
_LORA_KEY_SUBSTRINGS: List[str] = [
    "W_down_Q",
    "W_up_Q",
    "W_down_V",
    "W_up_V",
    "lora",
    "LoRA",
]

_FACT_KEY_SUBSTRINGS: List[str] = [
    ".U",
    ".V",
    "Sigma",
    "fact_B",
    "fact_A",
    "FacT",
    "fact",
]

# ---------------------------------------------------------------------------
# Known SSF-related key substrings.
# ---------------------------------------------------------------------------
_SSF_KEY_SUBSTRINGS: List[str] = [
    "ssf_w",
    "ssf_b",
    "selective_module._extra_params",
]

# ---------------------------------------------------------------------------
# Known DiffFit gamma key substrings.
# ---------------------------------------------------------------------------
_DIFFFIT_KEY_SUBSTRINGS: List[str] = [
    "gamma1_",
    "gamma2_",
]

# ---------------------------------------------------------------------------
# Head parameter key substrings.
# ---------------------------------------------------------------------------
_HEAD_KEY_SUBSTRINGS: List[str] = [
    "head.weight",
    "head.bias",
]

# ---------------------------------------------------------------------------
# Default α values from config.yaml: robustness.wise.alphas
# ---------------------------------------------------------------------------
_DEFAULT_ALPHAS: List[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# ---------------------------------------------------------------------------
# Default shift dataset keys (config.yaml: robustness.shift_datasets minus 'imagenet')
# ---------------------------------------------------------------------------
_DEFAULT_SHIFT_KEYS: List[str] = [
    "imagenet_v2",
    "imagenet_r",
    "imagenet_s",
    "imagenet_a",
]


class WiSE:
    """Weight-Space Ensemble (WiSE) for PEFT robustness experiments.

    Linearly interpolates between a fine-tuned PEFT model's parameters and
    the original pre-trained model's parameters, controlled by a mixing
    coefficient α ∈ [0, 1].

    α = 0.0 → fully pre-trained model (zero-shot CLIP)
    α = 1.0 → fully fine-tuned PEFT model

    The interpolation strategy is method-specific:
    - Adapter methods: scale adapter weights by α (residual connection semantics)
    - LoRA/FacT: scale additive residuals ΔW by α
    - Selective (BitFit/LN/DiffFit/SSF): linear interpolation of modified params
    - Full FT: standard WiSE (interpolate all parameters)
    - Head: interpolate between fine-tuned head and zero-shot CLIP head

    Paper: "To the best of our knowledge, we are the first to study WiSE for
    PEFT." (Section 7)

    Attributes:
        pretrained_state: Deep copy of the original pre-trained model state dict.
            For CLIP experiments, this is the CLIP ViT-B/16 state dict before
            any fine-tuning, including the zero-shot head weights.
        finetuned_state: Deep copy of the PEFT fine-tuned model state dict.
            Contains both backbone parameters (frozen, identical to pretrained)
            and PEFT-specific parameters (updated during fine-tuning).
        method: PEFT method name string. Determines the interpolation strategy.
            Must be one of the method names in SUPPORTED_METHODS.
    """

    def __init__(
        self,
        pretrained_state_dict: Dict[str, torch.Tensor],
        finetuned_state_dict: Dict[str, torch.Tensor],
        method: str,
    ) -> None:
        """Initialises WiSE with pretrained and fine-tuned state dicts.

        Both state dicts are deep-copied to prevent mutation during the sweep.
        The method string determines which interpolation strategy is used.

        Args:
            pretrained_state_dict: State dict of the original pre-trained model
                (before any fine-tuning). For CLIP experiments, this should
                include the zero-shot head weights under 'head.weight' and
                optionally 'head.bias'. All backbone parameters should be at
                their pretrained values.
                Produced by: copy.deepcopy(model.state_dict()) before fine-tuning.
            finetuned_state_dict: State dict of the PEFT fine-tuned model.
                Contains frozen backbone parameters (identical to pretrained)
                and updated PEFT-specific parameters (adapters, LoRA matrices,
                bias terms, SSF scale/shift, etc.).
                Produced by: checkpoint.load(path, model) after fine-tuning.
            method: PEFT method name string. One of:
                - Adapter: 'houlsby_adapter', 'pfeiffer_adapter', 'adaptformer',
                  'convpass', 'repadapter'
                - LoRA/FacT: 'lora', 'fact_tt', 'fact_tk'
                - Selective: 'bitfit', 'layernorm', 'difffit', 'ssf'
                - VPT: 'vpt_shallow', 'vpt_deep'
                - Baselines: 'full', 'linear'
                Unknown methods fall back to selective-style interpolation.
        """
        # Deep-copy both state dicts to prevent mutation during sweep.
        self.pretrained_state: Dict[str, torch.Tensor] = copy.deepcopy(
            pretrained_state_dict
        )
        self.finetuned_state: Dict[str, torch.Tensor] = copy.deepcopy(
            finetuned_state_dict
        )
        self.method: str = method

        # Log key statistics for debugging.
        pretrained_keys: Set[str] = set(self.pretrained_state.keys())
        finetuned_keys: Set[str] = set(self.finetuned_state.keys())
        new_keys: Set[str] = finetuned_keys - pretrained_keys
        removed_keys: Set[str] = pretrained_keys - finetuned_keys
        shared_keys: Set[str] = pretrained_keys & finetuned_keys

        _logger.info(
            "WiSE initialised: method='%s', pretrained_keys=%d, "
            "finetuned_keys=%d, new_keys=%d, removed_keys=%d, shared_keys=%d",
            method,
            len(pretrained_keys),
            len(finetuned_keys),
            len(new_keys),
            len(removed_keys),
            len(shared_keys),
        )

        if new_keys:
            _logger.debug(
                "New keys in finetuned (PEFT-specific): %s",
                sorted(new_keys)[:10],  # Log first 10 for brevity
            )
        if removed_keys:
            _logger.warning(
                "Keys in pretrained but not in finetuned: %s. "
                "These will be taken from pretrained_state in interpolation.",
                sorted(removed_keys)[:10],
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def interpolate(self, alpha: float) -> Dict[str, torch.Tensor]:
        """Computes the interpolated state dict for a given mixing coefficient.

        Dispatches to the appropriate method-specific interpolation strategy
        based on self.method. The head is always interpolated via
        _interpolate_head() and merged into the returned state dict.

        α = 0.0 → fully pre-trained model
        α = 1.0 → fully fine-tuned PEFT model

        Args:
            alpha: Mixing coefficient in [0.0, 1.0].
                config.yaml: robustness.wise.alphas: [0.0, 0.1, ..., 1.0]
                Values outside [0, 1] are accepted but may produce
                extrapolated (out-of-distribution) parameter values.

        Returns:
            A new state dict (deep copy, not in-place) representing the
            interpolated model. All tensor values are on CPU. The returned
            dict can be loaded into a model via:
                model.load_state_dict(interpolated_state, strict=False)

        Note:
            The returned state dict may not contain all keys from the original
            model (e.g., frozen backbone keys not present in finetuned_state
            are taken from pretrained_state). Use strict=False when loading.
        """
        _logger.debug(
            "Interpolating: method='%s', alpha=%.3f", self.method, alpha
        )

        # ------------------------------------------------------------------
        # Dispatch to method-specific interpolation strategy.
        # ------------------------------------------------------------------
        if self.method in ADAPTER_METHODS:
            interpolated: Dict[str, torch.Tensor] = self._interpolate_adapter(alpha)
        elif self.method in LORA_METHODS:
            interpolated = self._interpolate_lora(alpha)
        elif self.method in SELECTIVE_METHODS:
            interpolated = self._interpolate_selective(alpha)
        elif self.method in FULL_FT_METHODS:
            interpolated = self._interpolate_full(alpha)
        elif self.method in VPT_METHODS:
            # VPT prompts are treated like selective parameters.
            interpolated = self._interpolate_selective(alpha)
        elif self.method in LINEAR_METHODS:
            # Linear probing: backbone is identical to pretrained; only head differs.
            interpolated = self._interpolate_selective(alpha)
        else:
            # Unknown method: fall back to selective-style interpolation.
            _logger.warning(
                "Unknown method '%s' for WiSE interpolation. "
                "Falling back to selective-style (linear interpolation of all params).",
                self.method,
            )
            interpolated = self._interpolate_selective(alpha)

        # ------------------------------------------------------------------
        # Always interpolate the head and merge into the result.
        # The head interpolation may update keys already set by the backbone
        # interpolation above (e.g., 'head.weight', 'head.bias').
        # ------------------------------------------------------------------
        head_state: Dict[str, torch.Tensor] = self._interpolate_head(alpha)
        interpolated.update(head_state)

        return interpolated

    def sweep(
        self,
        model: nn.Module,
        alphas: Optional[List[float]] = None,
        target_loader: Any = None,
        shift_loaders: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
    ) -> List[Dict[str, Any]]:
        """Sweeps over α values and evaluates the interpolated model.

        For each α in alphas:
        1. Computes the interpolated state dict via interpolate(α)
        2. Loads it into the model (non-destructive — restores after each α)
        3. Evaluates on the target distribution (ImageNet-1K test)
        4. Evaluates on each distribution shift dataset
        5. Records results

        Generates the accuracy-robustness tradeoff curve (Figure 1c in the
        paper). The X-axis is target_acc (ImageNet-1K), the Y-axis is
        avg_shift_acc (mean of V2/R/S/A).

        Paper: "Each curve corresponds to the WiSE + PEFT method, with dots
        indicating different mixing coefficients α." (Figure 1c details)

        Config: robustness.wise.alphas: [0.0, 0.1, 0.2, ..., 1.0]

        Args:
            model: The PEFTModel (or any nn.Module) to evaluate. The model's
                state dict is saved before the sweep and restored after each
                α evaluation. The model should already be on the correct device.
            alphas: List of mixing coefficients to sweep over. Default: the
                11 values from config.yaml (robustness.wise.alphas):
                [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0].
            target_loader: DataLoader for the target distribution (ImageNet-1K
                test set). If None, target accuracy is recorded as None.
            shift_loaders: Dict mapping split name to DataLoader for each
                distribution shift dataset. Expected keys (from config.yaml:
                robustness.shift_datasets minus 'imagenet'):
                    {'imagenet_v2': DataLoader, 'imagenet_r': DataLoader,
                     'imagenet_s': DataLoader, 'imagenet_a': DataLoader}
                If None or empty, shift accuracies are recorded as None.
            device: Target device for evaluation. Default: 'cuda'.
                Falls back to 'cpu' if CUDA is unavailable.

        Returns:
            List of dicts, one per α value, with keys:
            - 'alpha': float — the mixing coefficient
            - 'target_acc': float or None — Top-1 accuracy on ImageNet-1K test
            - 'imagenet_v2_acc': float or None — Top-1 accuracy on ImageNet-V2
            - 'imagenet_r_acc': float or None — Top-1 accuracy on ImageNet-R
            - 'imagenet_s_acc': float or None — Top-1 accuracy on ImageNet-S
            - 'imagenet_a_acc': float or None — Top-1 accuracy on ImageNet-A
            - 'avg_shift_acc': float or None — mean of available shift accs
            All accuracy values are in [0.0, 1.0] (fractions, not percentages).

        Note:
            The model's state dict is fully restored after the sweep completes
            (or if an error occurs). The model is left in eval mode after the
            sweep.
        """
        # ------------------------------------------------------------------
        # Resolve defaults.
        # ------------------------------------------------------------------
        if alphas is None:
            alphas = list(_DEFAULT_ALPHAS)

        if shift_loaders is None:
            shift_loaders = {}

        # Resolve device.
        resolved_device: str = device
        if device == "cuda" and not torch.cuda.is_available():
            _logger.warning(
                "CUDA requested but not available. Falling back to CPU for WiSE sweep."
            )
            resolved_device = "cpu"

        _logger.info(
            "Starting WiSE sweep: method='%s', %d alpha values, "
            "target_loader=%s, shift_loaders=%s, device='%s'",
            self.method,
            len(alphas),
            "provided" if target_loader is not None else "None",
            list(shift_loaders.keys()),
            resolved_device,
        )

        # ------------------------------------------------------------------
        # Save the original model state dict for restoration after each α.
        # ------------------------------------------------------------------
        original_state: Dict[str, torch.Tensor] = copy.deepcopy(
            model.state_dict()
        )

        # ------------------------------------------------------------------
        # Initialise Metrics instance for accuracy computation.
        # ------------------------------------------------------------------
        metrics: Metrics = Metrics()

        # ------------------------------------------------------------------
        # Sweep over all α values.
        # ------------------------------------------------------------------
        results: List[Dict[str, Any]] = []

        for alpha_idx, alpha in enumerate(alphas):
            _logger.info(
                "WiSE sweep: alpha=%.3f (%d/%d)",
                alpha,
                alpha_idx + 1,
                len(alphas),
            )

            try:
                # ----------------------------------------------------------
                # Step 1: Compute interpolated state dict.
                # ----------------------------------------------------------
                interpolated_state: Dict[str, torch.Tensor] = self.interpolate(alpha)

                # ----------------------------------------------------------
                # Step 2: Load interpolated state into model.
                # strict=False: interpolated_state may not contain all keys
                # (e.g., frozen backbone keys not in finetuned_state).
                # ----------------------------------------------------------
                missing_keys: List[str]
                unexpected_keys: List[str]
                load_result = model.load_state_dict(
                    interpolated_state, strict=False
                )
                missing_keys = load_result.missing_keys
                unexpected_keys = load_result.unexpected_keys

                if unexpected_keys:
                    _logger.warning(
                        "WiSE load_state_dict: unexpected keys (ignored): %s",
                        unexpected_keys[:5],
                    )
                if missing_keys:
                    _logger.debug(
                        "WiSE load_state_dict: missing keys (retained from model): %d",
                        len(missing_keys),
                    )

                # Move model to device and set to eval mode.
                model.to(resolved_device)
                model.eval()

                # ----------------------------------------------------------
                # Step 3: Evaluate on target distribution (ImageNet-1K test).
                # ----------------------------------------------------------
                target_acc: Optional[float] = None

                if target_loader is not None:
                    try:
                        target_preds: torch.Tensor
                        target_confs: torch.Tensor
                        target_labels: torch.Tensor
                        target_preds, target_confs, target_labels = (
                            metrics.compute_predictions(
                                model=model,
                                loader=target_loader,
                                device=resolved_device,
                            )
                        )
                        target_acc = metrics.top1_accuracy(
                            target_preds, target_labels
                        )
                        _logger.info(
                            "alpha=%.3f, target_acc=%.4f (%.2f%%)",
                            alpha,
                            target_acc,
                            target_acc * 100.0,
                        )
                    except Exception as exc:  # pylint: disable=broad-except
                        _logger.error(
                            "Failed to evaluate target distribution at alpha=%.3f: %s",
                            alpha,
                            exc,
                        )
                        target_acc = None

                # ----------------------------------------------------------
                # Step 4: Evaluate on each distribution shift dataset.
                # config.yaml: robustness.shift_datasets (minus 'imagenet')
                # ----------------------------------------------------------
                shift_accs: Dict[str, Optional[float]] = {}

                for split_name in _DEFAULT_SHIFT_KEYS:
                    if split_name not in shift_loaders:
                        shift_accs[split_name] = None
                        continue

                    shift_loader = shift_loaders[split_name]

                    try:
                        shift_preds: torch.Tensor
                        shift_confs: torch.Tensor
                        shift_labels: torch.Tensor
                        shift_preds, shift_confs, shift_labels = (
                            metrics.compute_predictions(
                                model=model,
                                loader=shift_loader,
                                device=resolved_device,
                            )
                        )
                        shift_acc: float = metrics.top1_accuracy(
                            shift_preds, shift_labels
                        )
                        shift_accs[split_name] = shift_acc

                        _logger.info(
                            "alpha=%.3f, %s_acc=%.4f (%.2f%%)",
                            alpha,
                            split_name,
                            shift_acc,
                            shift_acc * 100.0,
                        )
                    except Exception as exc:  # pylint: disable=broad-except
                        _logger.error(
                            "Failed to evaluate shift dataset '%s' at alpha=%.3f: %s",
                            split_name,
                            alpha,
                            exc,
                        )
                        shift_accs[split_name] = None

                # ----------------------------------------------------------
                # Step 5: Compute average shift accuracy.
                # Average over all available (non-None) shift accuracies.
                # ----------------------------------------------------------
                available_shift_accs: List[float] = [
                    v for v in shift_accs.values() if v is not None
                ]
                avg_shift_acc: Optional[float] = (
                    float(sum(available_shift_accs) / len(available_shift_accs))
                    if available_shift_accs
                    else None
                )

                if avg_shift_acc is not None:
                    _logger.info(
                        "alpha=%.3f, avg_shift_acc=%.4f (%.2f%%) "
                        "over %d shift datasets",
                        alpha,
                        avg_shift_acc,
                        avg_shift_acc * 100.0,
                        len(available_shift_accs),
                    )

                # ----------------------------------------------------------
                # Step 6: Record result for this α.
                # ----------------------------------------------------------
                result: Dict[str, Any] = {
                    "alpha": alpha,
                    "target_acc": target_acc,
                    "imagenet_v2_acc": shift_accs.get("imagenet_v2", None),
                    "imagenet_r_acc": shift_accs.get("imagenet_r", None),
                    "imagenet_s_acc": shift_accs.get("imagenet_s", None),
                    "imagenet_a_acc": shift_accs.get("imagenet_a", None),
                    "avg_shift_acc": avg_shift_acc,
                }
                results.append(result)

            except Exception as exc:  # pylint: disable=broad-except
                _logger.error(
                    "WiSE sweep failed at alpha=%.3f: %s. "
                    "Recording None values and continuing.",
                    alpha,
                    exc,
                    exc_info=True,
                )
                results.append({
                    "alpha": alpha,
                    "target_acc": None,
                    "imagenet_v2_acc": None,
                    "imagenet_r_acc": None,
                    "imagenet_s_acc": None,
                    "imagenet_a_acc": None,
                    "avg_shift_acc": None,
                })

            finally:
                # ----------------------------------------------------------
                # Step 7: Restore original model state after each α.
                # This ensures each α evaluation starts from the same base.
                # ----------------------------------------------------------
                try:
                    model.load_state_dict(original_state, strict=False)
                except Exception as restore_exc:  # pylint: disable=broad-except
                    _logger.error(
                        "Failed to restore model state after alpha=%.3f: %s",
                        alpha,
                        restore_exc,
                    )

        _logger.info(
            "WiSE sweep complete: %d alpha values evaluated, "
            "%d results recorded.",
            len(alphas),
            len(results),
        )

        return results

    # ------------------------------------------------------------------
    # Private interpolation methods
    # ------------------------------------------------------------------

    def _interpolate_adapter(self, alpha: float) -> Dict[str, torch.Tensor]:
        """Interpolates adapter-based PEFT models.

        Paper: "For Adapter-based methods, WiSE can be considered feature
        ensembles, where α controls how strong the domain-specific features
        from the adapter module blend with the domain-agnostic ones from
        original backbones."

        Strategy:
        - Backbone parameters (present in both state dicts, frozen during PEFT):
          taken from pretrained_state (they are identical in both).
        - Adapter parameters (new keys only in finetuned_state):
          scaled by α: adapter_param_wise = α * adapter_param_finetuned
          At α=0: adapters contribute nothing (pure pre-trained behavior).
          At α=1: full adapter contribution.
        - Parameters only in pretrained_state: taken from pretrained_state.

        This scaling works because adapters use residual connections:
            Adapter(h) = s · W_up(σ(W_down(h))) + h
        Scaling W_down and W_up by α scales the non-identity branch by α²,
        but scaling the scale factor s by α gives a cleaner linear blend.
        We scale all adapter parameters by α for simplicity.

        Args:
            alpha: Mixing coefficient in [0.0, 1.0].

        Returns:
            Interpolated state dict with backbone params from pretrained and
            adapter params scaled by α.
        """
        result: Dict[str, torch.Tensor] = {}

        pretrained_keys: Set[str] = set(self.pretrained_state.keys())
        finetuned_keys: Set[str] = set(self.finetuned_state.keys())

        # ------------------------------------------------------------------
        # Process all keys from both state dicts.
        # ------------------------------------------------------------------
        all_keys: Set[str] = pretrained_keys | finetuned_keys

        for key in all_keys:
            # Skip head keys — handled by _interpolate_head().
            if self._is_head_key(key):
                continue

            if key in pretrained_keys and key in finetuned_keys:
                # Key present in both: backbone parameter (frozen during PEFT).
                # Values should be identical; use pretrained for clarity.
                result[key] = self.pretrained_state[key].clone()

            elif key in finetuned_keys and key not in pretrained_keys:
                # New key: PEFT-specific parameter (adapter weights, etc.).
                # Scale by α: at α=0, adapter contributes nothing.
                finetuned_tensor: torch.Tensor = self.finetuned_state[key]
                result[key] = (alpha * finetuned_tensor).clone()

            elif key in pretrained_keys and key not in finetuned_keys:
                # Key only in pretrained: take from pretrained.
                result[key] = self.pretrained_state[key].clone()

        return result

    def _interpolate_lora(self, alpha: float) -> Dict[str, torch.Tensor]:
        """Interpolates LoRA and FacT PEFT models.

        Paper: "For efficient selective methods (e.g. LoRA) as they learn
        additive residuals to the original parameters."

        Strategy:
        - Backbone parameters (frozen, identical in both): from pretrained_state.
        - LoRA parameters (W_down_Q, W_up_Q, W_down_V, W_up_V):
          Scale W_up by α: W_up_wise = α * W_up_finetuned
          This scales ΔW = W_up @ W_down by α, giving:
              W_wise = W_pretrained + α * ΔW
          W_down is kept unchanged (scaling W_up alone scales the residual).
        - FacT parameters (U, V, Sigma/B/A):
          Scale Sigma (TT) or A (TK) by α, which scales ΔW_FacT by α.
          U, V, B are kept unchanged.
        - New keys not matching LoRA/FacT patterns: scale by α (safe default).

        Args:
            alpha: Mixing coefficient in [0.0, 1.0].

        Returns:
            Interpolated state dict with backbone params from pretrained and
            LoRA/FacT residuals scaled by α.
        """
        result: Dict[str, torch.Tensor] = {}

        pretrained_keys: Set[str] = set(self.pretrained_state.keys())
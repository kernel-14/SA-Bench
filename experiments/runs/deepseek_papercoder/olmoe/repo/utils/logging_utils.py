# utils/logging_utils.py
"""
Centralised logging and metrics tracking for OLMoE reproduction.

Uses Weights & Biases (wandb) for experiment tracking and metric visualisation.

Provides a single ``LoggingManager`` class that:
* initialises wandb with the full project configuration,
* logs model architecture information,
* streams training and evaluation metrics,
* records expert‑specific statistics (expert load, dead experts),
* and cleanly finishes the run.

All paths and hyperparameter names are taken from the global configuration
dictionary (``config.yaml``).  The class is stateless apart from an ``initialized``
flag and a handle to the active wandb run.

Typical usage:
    from utils.logging_utils import LoggingManager
    logger = LoggingManager(config)
    logger.init_wandb()
    logger.log_model_info(model)
    ...
    logger.log_train_metrics({'train/loss': 2.3}, step=10)
    ...
    logger.finish()
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Union

import torch
from torch import nn

# Weights & Biases may not be installed in offline environments;
# wrap the import so that the module can still be loaded.
try:
    import wandb
except ImportError:
    wandb = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Helper: flatten a nested dict into a single‑level dict with dot‑keys.
# ---------------------------------------------------------------------------
def _flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Recursively flatten a nested dictionary.

    Example:  {'a': {'b': 1}}  →  {'a.b': 1}

    Args:
        d:          Nested dictionary.
        parent_key: Prefix for keys (used internally during recursion).
        sep:        Separator between keys.

    Returns:
        Flat dictionary.
    """
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# ---------------------------------------------------------------------------
#  Main logging class
# ---------------------------------------------------------------------------
class LoggingManager:
    """
    W&B logging facade for pretraining, adaptation, and evaluation.

    All configuration values are drawn from the ``config`` dictionary that
    matches the structure of ``config.yaml``.

    Args:
        cfg: Full project configuration (as loaded from config.yaml).
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.initialized: bool = False

        # Extract frequently used settings (none of them are mandatory)
        log_cfg = cfg.get("logging", {})
        self.log_interval: int = log_cfg.get("log_interval", 10)

        # wandb is optional; if not installed, all methods become no‑ops.
        self.wandb_available: bool = wandb is not None

    # ------------------------------------------------------------------
    #  Initialisation
    # ------------------------------------------------------------------
    def init_wandb(self, mode: str = "online") -> None:
        """
        Start a wandb run and upload the project configuration.

        Should be called once, typically right after the trainer is created.

        Args:
            mode:     wandb run mode – "online", "offline", or "disabled".
                      If wandb is not installed, mode will be ignored.
        """
        if not self.wandb_available:
            logger.warning(
                "wandb is not installed. Proceeding without experiment tracking."
            )
            return

        try:
            log_cfg = self.cfg.get("logging", {})
            project = log_cfg.get("project", "OLMoE")
            entity = log_cfg.get("entity", None)

            # Flatten the full configuration so it appears as hyperparameters
            # in the wandb dashboard.
            flat_config = _flatten_dict(self.cfg)

            wandb.init(
                project=project,
                entity=entity,
                config=flat_config,
                mode=mode,
                # resume="allow" could be added if checkpoint recovery is needed
            )

            # Log hardware information from config (optional)
            hw = self.cfg.get("hardware", {})
            if hw:
                wandb.config.update({"hardware/train_nodes": hw.get("train_nodes", 0)})
                wandb.config.update({"hardware/gpus_per_node": hw.get("gpus_per_node", 0)})

            self.initialized = True
            logger.info(
                "wandb run started: project=%s, entity=%s, mode=%s",
                project, entity, mode,
            )

        except Exception as e:
            logger.error("wandb initialisation failed: %s", e)
            self.wandb_available = False   # disable further functionality

    # ------------------------------------------------------------------
    #  Model architecture logging
    # ------------------------------------------------------------------
    def log_model_info(self, model: nn.Module) -> None:
        """
        Log model size, parameter counts, and (optionally) watch the model
        for gradient/parameter histograms.

        Args:
            model: The model instance (e.g., MoETransformer).
        """
        if not self._should_log():
            return

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Write summary metrics (visible on the wandb run page)
        wandb.run.summary["model/total_params"] = total_params
        wandb.run.summary["model/trainable_params"] = trainable_params

        # Optionally log the graph of the model.
        # wandb.watch can be heavy; enable it only if wanted.
        try:
            wandb.watch(model, log="parameters", log_freq=100)
        except Exception as e:
            logger.debug("wandb.watch failed (non‑critical): %s", e)

        logger.info(
            "Model info logged: total=%d, trainable=%d",
            total_params, trainable_params,
        )

    # ------------------------------------------------------------------
    #  Training metrics (scalars)
    # ------------------------------------------------------------------
    def log_train_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """
        Log a dictionary of scalar training metrics at the given step.

        The dictionary may contain:
            - ``train/loss``, ``train/load_balancing_loss``,
              ``train/router_z_loss``, ``train/total_loss``
            - ``train/learning_rate``
            - ``train/grad_norm``
            - ``train/tokens_per_second``

        Args:
            metrics: Flat dict of metric name → value.
            step:    Global training step (used as x‑axis).
        """
        if not self._should_log():
            return

        # All values must be numeric scalars; if not, convert to float.
        sanitized = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if isinstance(v, (int, float)):
                sanitized[k] = float(v)
            else:
                logger.warning("Metric '%s' has non‑numeric type %s, skipping.", k, type(v))

        if sanitized:
            wandb.log(sanitized, step=step)

    # ------------------------------------------------------------------
    #  Evaluation metrics (accuracy, perplexity, etc.)
    # ------------------------------------------------------------------
    def log_eval_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """
        Log downstream evaluation results.

        Expected keys follow the naming convention used in the evaluators:
            - ``eval/validation_loss`` or ``eval/perplexity``
            - Per‑task accuracy: ``eval/hellaswag_acc``, ``eval/mmlu_var_acc``, ...
            - OLMES or instruct benchmark results.

        Args:
            metrics: Flat dict of metric name → value.
            step:    Global training step (or evaluation step).
        """
        if not self._should_log():
            return

        sanitized = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if isinstance(v, (int, float)):
                sanitized[k] = float(v)
            else:
                logger.warning("Evaluation metric '%s' has non‑numeric type %s, skipping.", k, type(v))

        if sanitized:
            wandb.log(sanitized, step=step)

    # ------------------------------------------------------------------
    #  Expert‑specific statistics (histograms, dead‑expert fractions)
    # ------------------------------------------------------------------
    def log_expert_stats(
        self,
        expert_assignments: Optional[torch.Tensor] = None,
        dead_expert_frac: Optional[float] = None,
        entropy: Optional[float] = None,
        step: int = 0,
    ) -> None:
        """
        Log expert usage statistics at the end of a training interval.

        Args:
            expert_assignments: 2D LongTensor of shape (num_layers, num_experts)
                                with the count (or fraction) of tokens assigned
                                to each expert.  None if not available.
            dead_expert_frac:   Fraction of experts that were never selected
                                in the recent interval.
            entropy:            Shannon entropy of the expert selection distribution
                                (optional).
            step:               Global training step.
        """
        if not self._should_log():
            return

        metrics: Dict[str, Any] = {}
        if expert_assignments is not None:
            # Log a histogram for the entire model (flattened counts)
            assignments_np = expert_assignments.detach().cpu().numpy()
            metrics["expert/assignment_distribution"] = wandb.Histogram(
                assignments_np.flatten()
            )
            # Also log per‑layer histograms? Could be bulky; skip for simplicity.

        if dead_expert_frac is not None:
            metrics["expert/dead_expert_fraction"] = dead_expert_frac

        if entropy is not None:
            metrics["expert/entropy"] = entropy

        if metrics:
            wandb.log(metrics, step=step)

    # ------------------------------------------------------------------
    #  Cleanup
    # ------------------------------------------------------------------
    def finish(self) -> None:
        """Complete the wandb run and release resources."""
        if self._should_log():
            try:
                wandb.finish()
                self.initialized = False
                logger.info("wandb run finished.")
            except Exception as e:
                logger.error("Error during wandb.finish(): %s", e)

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------
    def _should_log(self) -> bool:
        """Return True if wandb is initialised and the logger is active."""
        return self.wandb_available and self.initialized


# ---------------------------------------------------------------------------
#  Convenience module‑level functions (optional, for backward compatibility)
# ---------------------------------------------------------------------------
def init_wandb(cfg: Dict[str, Any], mode: str = "online") -> LoggingManager:
    """
    Create and initialise a LoggingManager from the configuration.

    Provided as a convenience function for simple scripts.
    Returns the manager instance.
    """
    mgr = LoggingManager(cfg)
    mgr.init_wandb(mode)
    return mgr


def log_metrics(metrics: Dict[str, Any], step: int = 0) -> None:
    """
    Log metrics using the currently active wandb run (if any).

    This is a lightweight global helper intended for use inside modules
    that do not hold a reference to a LoggingManager (e.g., evaluation
    callbacks). It does **not** create a new run.

    Args:
        metrics: Dict of metric name → value.
        step:    Global step.
    """
    if wandb is not None and wandb.run is not None:
        sanitized = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if isinstance(v, (int, float)):
                sanitized[k] = float(v)
        if sanitized:
            wandb.log(sanitized, step=step)
    else:
        logger.debug("wandb not active; metrics not logged: %s", metrics)


# ---------------------------------------------------------------------------
#  Self‑test (executed only when the module is run directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # A trivial sanity check
    import yaml
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        config = {"logging": {"project": "test_project", "entity": "test_entity"}}

    mgr = LoggingManager(config)
    # If wandb is installed, this will try to connect. Disable for test.
    mgr.init_wandb(mode="disabled")
    # Build a dummy model
    model = nn.Linear(10, 5)
    mgr.log_model_info(model)
    mgr.log_train_metrics({"train/loss": 0.5, "train/learning_rate": 1e-4}, step=0)
    mgr.log_eval_metrics({"eval/accuracy": 0.9}, step=100)
    mgr.log_expert_stats(
        expert_assignments=torch.randint(0, 100, (16, 64)),
        dead_expert_frac=0.02,
        entropy=4.1,
        step=100,
    )
    mgr.finish()
    print("Logging test completed (wandb mode=disabled).")

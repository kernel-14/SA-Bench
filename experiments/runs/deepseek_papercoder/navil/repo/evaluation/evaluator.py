# evaluation/evaluator.py

"""
Evaluator for the NaViL model on standard multimodal benchmarks.

This module implements the ``Evaluator`` class, which orchestrates end‑to‑end
evaluation of a trained NaViL model across the benchmarks listed in the paper
(Tables 1 & 2).  It leverages the `OpenCompass` toolkit for dataset loading
and metric computation wherever possible, falling back to custom logic for
benchmarks not yet covered by that library.

The public interface is minimal:

* ``Evaluator(model, config)`` – initialise with a trained ``NaViLModel``
  and an ``EvalConfig``.
* ``evaluate()`` – run all benchmarks and return a dictionary of normalised
  scores plus an overall average.
* ``run_benchmark(name)`` – evaluate a single benchmark, returning raw
  metrics.
* ``aggregate_metrics(results)`` – normalise per‑benchmark scores into the
  0‑100 range and compute the paper’s “Average” column.

All logging uses the project‑wide ``utils.logging`` infrastructure.
"""

from __future__ import annotations

import copy
import importlib
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Project imports – circular import is avoided by importing at runtime
# ---------------------------------------------------------------------------
from config import EvalConfig
from models.navil_model import NaViLModel
from utils.logging import get_logger

# ---------------------------------------------------------------------------
# Logger (will inherit root configuration from main.py)
# ---------------------------------------------------------------------------
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal helper: load an OpenCompass dataset / evaluator pair
# ---------------------------------------------------------------------------
def _try_import_opencompass_component(module_path: str, class_name: str) -> Any:
    """
    Attempt to dynamically import a class from an OpenCompass module.

    Args:
        module_path: Dotted module path, e.g. ``"opencompass.datasets"``.
        class_name: Desired class name.

    Returns:
        The class if found, otherwise ``None``.
    """
    try:
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name, None)
    except (ImportError, AttributeError):
        logger.warning(
            "Could not import %s.%s. Falling back to custom logic.", module_path, class_name
        )
        return None


# ---------------------------------------------------------------------------
# Benchmark‑specific normalisation constants
# ---------------------------------------------------------------------------
# These values mirror the scaling used in the NaViL paper:
#   - For accuracy‑based metrics (0‑100 scale) we normalise by 100.
#   - For MME, the sum of perception and cognition max is 2×2000 = 4000.
#   - For OCRBench, max is 1000.
#   - For DocVQA, the official metric ANLS is in [0,1]; we multiply by 100.
#   - For ScienceQA‑IMG, accuracy is 0‑100 already.
_NORM_FACTOR: Dict[str, float] = {
    "MMVet": 100.0,           # accuracy (0‑100)
    "MMMU_val": 100.0,        # accuracy
    "MMBench_EN_test": 100.0, # accuracy
    "MME": 4000.0,            # sum of perception + cognition
    "MathVista_MINI": 100.0,  # accuracy
    "OCRBench": 1000.0,       # overall score
    "CCBench": 100.0,         # accuracy
    "TextVQA_val": 100.0,     # accuracy
    "ScienceQA_IMG_test": 100.0, # accuracy
    "GQA_testdev": 100.0,     # accuracy
    "DocVQA_test": 100.0,     # ANLS × 100 ⇒ mapped to 0‑100
    "AI2D_test": 100.0,       # accuracy
    "ChartQA_test": 100.0,    # relaxed accuracy
    "InfographicVQA_test": 100.0, # accuracy
}

# Primary metric key for each benchmark as returned by OpenCompass evaluator
_PRIMARY_METRIC_KEY: Dict[str, str] = {
    "MMVet": "accuracy",
    "MMMU_val": "accuracy",
    "MMBench_EN_test": "accuracy",
    "MME": "score",                 # sum already computed by MMEEvaluator
    "MathVista_MINI": "accuracy",
    "OCRBench": "score",
    "CCBench": "accuracy",
    "TextVQA_val": "accuracy",
    "ScienceQA_IMG_test": "accuracy",
    "GQA_testdev": "accuracy",
    "DocVQA_test": "anls",
    "AI2D_test": "accuracy",
    "ChartQA_test": "relaxed_accuracy",
    "InfographicVQA_test": "accuracy",
}


# ---------------------------------------------------------------------------
# Lightweight model wrapper for OpenCompass
# ---------------------------------------------------------------------------
class _NaViLModelAdapter:
    """
    Adapter that conforms to the minimal generative interface expected by
    OpenCompass (``opencompass.models.BaseModel`` – only the ``generate``
    method is strictly required for evaluation).

    Each input dictionary is expected to contain:
      - ``"prompt"``: text string (already including the conversation template).
      - ``"image"``: optional PIL Image or path to an image.

    The adapter batches inputs sequentially (batch_size = 1) because the
    NaViL model does not support batched generation and because multi‑scale
    packing may produce variable token lengths.
    """

    def __init__(self, navil_model: NaViLModel, max_new_tokens: int = 256) -> None:
        self.model = navil_model
        self.max_new_tokens = max_new_tokens
        self.model.eval()

    def generate(self, inputs: List[Dict[str, Any]], max_out_len: int = None) -> List[str]:
        """
        Generate responses for a list of prompts.

        Args:
            inputs: List of dictionaries, each with at least ``"prompt"``
                and optionally ``"image"`` (PIL Image).
            max_out_len: Maximum number of generated tokens. Overrides the
                default if provided.

        Returns:
            List of generated text strings.
        """
        max_len = max_out_len if max_out_len is not None else self.max_new_tokens
        responses: List[str] = []

        with torch.inference_mode():
            for inp in inputs:
                prompt = inp["prompt"]
                image = inp.get("image", None)
                # The NaViL model's generate expects a list of images (multi‑scale pyramid).
                # If image is provided, we pre‑process it into the multi‑scale format.
                pixel_values = []  # list of per‑scale tensors (each [1,3,H,W])
                if image is not None:
                    if isinstance(image, (str, os.PathLike)):
                        image = Image.open(image).convert("RGB")
                    # Use the model's image tokenizer (contained in the model's
                    # preprocessing module) to build the pyramid. We access it
                    # via model.image_tokenizer.
                    try:
                        pixel_values = self.model.image_tokenizer.process_image(image)
                    except AttributeError:
                        # Fallback: create a simple single‑scale processing
                        # using the model's patch size and multi‑scale config.
                        pixel_values = [image]  # Bypass if not available
                # The model's generate method will take pixel_values as a list of
                # scale tensors. We need to stack them into a batch of 1.
                if pixel_values:
                    # Stack each scale to have batch dim = 1
                    pixel_values = [p.unsqueeze(0) for p in pixel_values]
                else:
                    pixel_values = None

                # Tokenize the prompt (ignoring special tokens because the
                # model's generate expects raw token IDs without image placeholders)
                # We just pass prompt to the model's generate, which internally
                # tokenizes and adds image tokens.
                # Direct generation call:
                generated_text = self.model.generate(
                    pixel_values=pixel_values,
                    input_ids=self.model.tokenizer.encode(prompt, return_tensors="pt").to(
                        self.model.vocab_device()
                    ),
                    max_new_tokens=max_len,
                )
                responses.append(generated_text)

        return responses

    def get_tokenizer(self):
        """Return the model's tokenizer (required by some OpenCompass evaluators)."""
        return self.model.tokenizer

    def vocab_device(self) -> torch.device:
        """Return the device on which to place tokenised tensors."""
        # We can get device from any parameter
        return next(self.model.parameters()).device


# ---------------------------------------------------------------------------
# Evaluator class
# ---------------------------------------------------------------------------
class Evaluator:
    """
    Runs benchmark evaluation for a trained NaViL model.

    Args:
        model: A fully initialised ``NaViLModel`` (should be in eval mode).
        config: Evaluation configuration (which benchmarks to run, batch size, etc.).
    """

    def __init__(self, model: NaViLModel, config: EvalConfig) -> None:
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        self.tokenizer = model.tokenizer
        self.image_tokenizer = getattr(model, "image_tokenizer", None)
        self.use_multi_scale = self.model.config.multi_scale.enabled

        # Ensure model is in evaluation mode
        self.model.eval()

        # Store generated predictions for reproducibility
        os.makedirs(self.config.output_dir, exist_ok=True)

        self.logger = logger

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self) -> Dict[str, float]:
        """
        Run evaluation on all benchmarks listed in the config.

        Returns:
            Dictionary containing:
            - ``"avg"``: Overall normalised average (0‑100).
            - Per‑benchmark normalised scores (keyed by benchmark name).
            - ``"raw_metrics"``: raw metric values for debugging.
        """
        raw_results: Dict[str, Dict[str, Any]] = {}
        for benchmark in self.config.benchmarks:
            self.logger.info("Starting evaluation on %s", benchmark)
            try:
                result = self.run_benchmark(benchmark)
                raw_results[benchmark] = result
                self.logger.info("Benchmark %s completed. Metrics: %s", benchmark, result)
            except Exception:
                self.logger.exception("Benchmark %s failed – skipping.", benchmark)
                # Create an empty result so aggregation can still proceed.
                raw_results[benchmark] = {}

        return self.aggregate_metrics(raw_results)

    def run_benchmark(self, name: str) -> Dict[str, Any]:
        """
        Evaluate the model on a single benchmark.

        Args:
            name: Benchmark identifier (e.g., ``"MMVet"``).

        Returns:
            Dictionary of metric names -> values as returned by the evaluator.
        """
        # ----- Try OpenCompass first -----
        cdataset, cevaluator = self._get_opencompass_components(name)
        if cdataset is not None and cevaluator is not None:
            return self._evaluate_with_opencompass(name, cdataset, cevaluator)
        # ----- Fallback to custom pipelines -----
        else:
            self.logger.warning(
                "No OpenCompass support for %s – using custom evaluation.", name
            )
            return self._evaluate_custom(name)

    def aggregate_metrics(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Normalise each benchmark's primary metric to a 0‑100 scale and compute
        the overall average (as described in the NaViL paper, Section 5.1).

        Args:
            results: Dictionary mapping benchmark name to its raw metrics dict.

        Returns:
            Dictionary with keys:
                - ``"avg"``: average of normalised scores.
                - Normalised score for each benchmark, e.g. ``"MMVet": 78.3``.
                - ``"raw_metrics"``: a copy of the input for inspection.
        """
        normalised: Dict[str, float] = {}
        for bm, metrics in results.items():
            if not metrics:
                # Benchmark failed or was not evaluated; set score to 0.0
                normalised[bm] = 0.0
                continue
            # Retrieve the primary metric key for this benchmark
            key = _PRIMARY_METRIC_KEY.get(bm, "accuracy")
            if key not in metrics:
                # try "score" as fallback
                if "score" in metrics:
                    key = "score"
                else:
                    self.logger.warning(
                        "Cannot find metric '%s' in results for %s. Available: %s",
                        key,
                        bm,
                        list(metrics.keys()),
                    )
                    normalised[bm] = 0.0
                    continue
            raw_val = metrics[key]
            # Compute normalised value
            if bm == "MME":
                # MME metric "score" in opencompass is already the sum of perception+cognition
                norm_val = (raw_val / 4000.0) * 100.0
            elif bm == "OCRBench":
                norm_val = (raw_val / 1000.0) * 100.0
            elif bm == "DocVQA_test":
                # ANLS in [0,1] → scale to 0‑100
                norm_val = raw_val * 100.0
            else:
                # For accuracy‑based metrics, raw is already 0‑100 or fraction; we assume 0‑100
                norm_val = float(raw_val)
            normalised[bm] = norm_val

        avg = sum(normalised.values()) / max(1, len(normalised))

        output = {
            "avg": avg,
            "raw_metrics": copy.deepcopy(results),
        }
        output.update(normalised)
        return output

    # ------------------------------------------------------------------
    # Internal – OpenCompass integration
    # ------------------------------------------------------------------

    def _get_opencompass_components(self, name: str) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Map a benchmark name to OpenCompass dataset and evaluator classes.

        Returns:
            Tuple (dataset_class, evaluator_class) or (None, None) if not available.
        """
        mapping = {
            "MMVet": ("opencompass.datasets", "MMVetDataset", "MMVetEvaluator"),
            "MMMU_val": ("opencompass.datasets", "MMMUDataset", "MMMUEvaluator"),
            "MMBench_EN_test": ("opencompass.datasets", "MMBenchDataset", "MMBenchEvaluator"),
            "MME": ("opencompass.datasets", "MMEDataset", "MMEEvaluator"),
            "MathVista_MINI": ("opencompass.datasets", "MathVistaDataset", "MathVistaEvaluator"),
            "OCRBench": ("opencompass.datasets", "OCRBenchDataset", "OCRBenchEvaluator"),
            "CCBench": ("opencompass.datasets", "CCBenchDataset", "CCBenchEvaluator"),
            "TextVQA_val": ("opencompass.datasets", "TextVQADataset", "VQAEvaluator"),
            "ScienceQA_IMG_test": ("opencompass.datasets", "ScienceQADataset", "ScienceQAEvaluator"),
            "GQA_testdev": ("opencompass.datasets", "GQADataset", "GQAEvaluator"),
            "DocVQA_test": ("opencompass.datasets", "DocVQADataset", "DocVQAEvaluator"),
            "AI2D_test": ("opencompass.datasets", "AI2DDataset", "AI2DEvaluator"),
            "ChartQA_test": ("opencompass.datasets", "ChartQADataset", "ChartQAEvaluator"),
            "InfographicVQA_test": ("opencompass.datasets", "InfoVQADataset", "InfoVQAEvaluator"),
        }
        entry = mapping.get(name)
        if entry is None:
            self.logger.warning("No OpenCompass mapping for %s", name)
            return None, None
        module, ds_cls_name, ev_cls_name = entry
        ds_cls = _try_import_opencompass_component(module, ds_cls_name)
        ev_cls = _try_import_opencompass_component(module, ev_cls_name)
        return ds_cls, ev_cls

    def _evaluate_with_opencompass(
        self, name: str, dataset_cls: Any, evaluator_cls: Any
    ) -> Dict[str, Any]:
        """
        Run evaluation using OpenCompass dataset and evaluator.

        This method:
        1. Instantiates the dataset.
        2. Instantiates the model adapter.
        3. Iterates over samples, calls model.generate, collects predictions.
        4. Passes predictions to the evaluator.
        """
        self.logger.info("Using OpenCompass for %s", name)

        # Instantiate dataset (some datasets require tokenizer, image_root, etc.)
        # We'll try to create with common parameters.
        dataset_kwargs: Dict[str, Any] = {}
        # If the dataset expects an image root, we can try to infer from environment
        # or use a default path. For simplicity, we rely on OpenCompass's defaults.
        # If they fail, we catch and fallback.
        try:
            dataset = dataset_cls(**dataset_kwargs)
        except Exception:
            # Some datasets need explicit split identifiers; try with split if needed
            if name == "MMBench_EN_test":
                dataset = dataset_cls(split="test")
            else:
                raise

        # Create the model adapter
        model_adapter = _NaViLModelAdapter(self.model, max_new_tokens=256)

        # Collect prompts and references
        predictions: List[str] = []
        references: List[str] = []
        for idx, sample in enumerate(dataset):
            # Each sample is expected to be a dict with 'prompt' and 'answer' or 'gt'
            prompt = sample["prompt"]
            image = sample.get("image", None)  # may be a PIL image or path
            gt = sample.get("answer", sample.get("gt", ""))

            # Generate
            self.logger.debug("Generating sample %d for %s", idx, name)
            output = model_adapter.generate([{"prompt": prompt, "image": image}])[0]
            predictions.append(output)
            references.append(gt)

        # Instantiate evaluator and compute metrics
        evaluator = evaluator_cls()
        metrics = evaluator.evaluate(predictions, references, output_dir=self.config.output_dir)
        # Ensure we return a flat dict
        if isinstance(metrics, dict):
            return metrics
        # Some evaluators return a single number; wrap it
        return {"score": metrics}

    # ------------------------------------------------------------------
    # Internal – custom fallback evaluation for unsupported benchmarks
    # ------------------------------------------------------------------

    def _evaluate_custom(self, name: str) -> Dict[str, Any]:
        """
        Custom evaluation pipeline for benchmarks not covered by OpenCompass.

        This is a placeholder; in practice you would manually load the dataset
        and use official evaluation scripts (e.g., from the respective authors).

        Raises:
            NotImplementedError: if no custom evaluator is registered.
        """
        raise NotImplementedError(
            f"Custom evaluation for {name} is not implemented. "
            "Please install OpenCompass or provide your own dataset loading/evaluation code."
        )


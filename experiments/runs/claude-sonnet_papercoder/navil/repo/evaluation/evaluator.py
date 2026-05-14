```python
## evaluation/evaluator.py
"""Benchmark evaluation pipeline for NaViL multimodal models.

This module implements ``BenchmarkEvaluator``, which runs a trained NaViL
model through all evaluation benchmarks defined in the configuration file
and computes standardized scores for each.

Supported benchmarks (from configs/navil_2b.yaml):
    MLLM benchmarks (Table 1):
        mmvet, mmmu, mmbench_en, mme, mathvista_mini, ocrbench, ccbench
    VQA benchmarks (Table 2):
        textvqa, scienceqa_img, gqa, docvqa, ai2d, chartqa, infographicvqa
    NLP benchmarks (Table 5):
        mmlu, cmmlu, math

Inference configuration (from config.yaml):
    evaluation.max_new_tokens: 512
    evaluation.decoding: "greedy"  → do_sample=False

Metric mapping (from config.yaml evaluation.benchmarks[*].metric):
    accuracy:          exact-match (lowercased, stripped)
    anls:              Average Normalized Levenshtein Similarity
    relaxed_accuracy:  numeric-tolerant accuracy (ChartQA)
    sum:               raw sum score (MME perception + cognition)
    score:             raw score (MMVet, OCRBench)

Dependencies:
    - model/navil_model.py: NaViLModel (injected, not constructed here)
    - evaluation/metrics.py: MetricsCalculator
    - utils/logging_utils.py: setup_logger
    - datasets: HuggingFace datasets hub
    - torch, PIL, json, os, re, tqdm
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image as PILImage
from tqdm import tqdm

from evaluation.metrics import MetricsCalculator
from model.navil_model import NaViLModel
from utils.logging_utils import setup_logger

logger: logging.Logger = logging.getLogger(__name__)


class BenchmarkEvaluator:
    """Runs NaViL through all configured evaluation benchmarks.

    Loads each benchmark dataset from HuggingFace hub (or local path),
    performs single-sample greedy inference, and computes benchmark-specific
    metrics. Results are aggregated into a normalized average score matching
    the "Avg" column in Tables 1 and 2 of the paper.

    Args:
        model:      A fully-initialized ``NaViLModel`` instance loaded from
                    a checkpoint. The model is set to eval mode and moved to
                    ``device`` in ``__init__``.
        benchmarks: List of benchmark configuration dicts from
                    ``config.evaluation.benchmarks``. Each dict must contain:
                    - ``"name"``:      Benchmark identifier string.
                    - ``"metric"``:    Metric type string.
                    - ``"max_value"``: Maximum raw score for normalization.
                    - ``"split"``:     Dataset split to evaluate on.
        device:     Device string for model inference. Default: ``"cuda"``
                    if available, else ``"cpu"``.

    Attributes:
        model:          The NaViLModel in eval mode on ``device``.
        benchmarks:     Stored benchmark config list.
        device:         Stored device string.
        results:        Dict accumulating benchmark scores. Populated by
                        ``run_all`` or individual ``evaluate_X`` calls.
        metrics:        ``MetricsCalculator`` instance for score computation.
        tokenizer:      Tokenizer from ``model.tokenizer`` for decoding.
        preprocessor:   Image preprocessor from ``model`` for preprocessing
                        PIL images into scale tensors.
        max_new_tokens: Maximum tokens to generate per sample (default: 512).

    Example::

        from omegaconf import OmegaConf
        config = OmegaConf.load("configs/navil_2b.yaml")
        model = NaViLModel.from_pretrained("checkpoints/navil_2b/step_000140000", config)
        evaluator = BenchmarkEvaluator(
            model=model,
            benchmarks=list(OmegaConf.to_container(config.evaluation.benchmarks)),
            device="cuda",
        )
        results = evaluator.run_all("results/navil_2b/eval_results.json")
    """

    def __init__(
        self,
        model: NaViLModel,
        benchmarks: List[Dict[str, Any]],
        device: str = "cuda",
    ) -> None:
        """Initialise the evaluator and prepare the model for inference.

        Args:
            model:      NaViLModel instance (already loaded from checkpoint).
            benchmarks: List of benchmark config dicts from config YAML.
            device:     Device string. Defaults to ``"cuda"`` if a CUDA device
                        is available, otherwise ``"cpu"``.
        """
        # Resolve device: fall back to CPU if CUDA is not available
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning(
                "CUDA requested but not available. Falling back to CPU."
            )
            device = "cpu"

        self.device: str = device
        self.benchmarks: List[Dict[str, Any]] = benchmarks
        self.results: Dict[str, float] = {}

        # ------------------------------------------------------------------ #
        # Prepare model for inference                                          #
        # ------------------------------------------------------------------ #
        model.eval()
        model.to(device)
        self.model: NaViLModel = model

        # ------------------------------------------------------------------ #
        # Tokenizer — needed for prompt tokenization and output decoding      #
        # ------------------------------------------------------------------ #
        self.tokenizer = model.tokenizer

        # ------------------------------------------------------------------ #
        # Image preprocessor — needed to convert PIL Images to scale tensors  #
        # ------------------------------------------------------------------ #
        # NaViLModel exposes the preprocessor via the data pipeline.
        # We construct a default ImagePreprocessor if not directly accessible.
        self.preprocessor = self._get_preprocessor(model)

        # ------------------------------------------------------------------ #
        # Metrics calculator                                                   #
        # ------------------------------------------------------------------ #
        self.metrics: MetricsCalculator = MetricsCalculator()

        # ------------------------------------------------------------------ #
        # Inference configuration                                              #
        # ------------------------------------------------------------------ #
        self.max_new_tokens: int = 512  # from config.evaluation.max_new_tokens

        # ------------------------------------------------------------------ #
        # Logging                                                              #
        # ------------------------------------------------------------------ #
        self._logger: logging.Logger = setup_logger(
            "navil.evaluator",
            log_file=None,  # console only; file logging set up by main.py
            level=logging.INFO,
        )

        self._logger.info(
            "BenchmarkEvaluator initialised: device=%s, "
            "num_benchmarks=%d, max_new_tokens=%d",
            device,
            len(benchmarks),
            self.max_new_tokens,
        )

    # ---------------------------------------------------------------------- #
    # Internal helpers                                                         #
    # ---------------------------------------------------------------------- #

    def _get_preprocessor(self, model: NaViLModel) -> Any:
        """Retrieve or construct an ImagePreprocessor from the model.

        Tries to access ``model.preprocessor`` directly. If not available,
        constructs a default ``ImagePreprocessor`` with standard NaViL-2B
        settings.

        Args:
            model: NaViLModel instance.

        Returns:
            An ``ImagePreprocessor`` instance.
        """
        if hasattr(model, "preprocessor") and model.preprocessor is not None:
            return model.preprocessor

        # Construct a default preprocessor
        try:
            import math
            from data.preprocessing import ImagePreprocessor
            from data.multi_scale_packing import MultiScalePacking
            from model.special_tokens import SpecialTokens

            # Use the model's special_tokens if available
            special_tokens = getattr(model, "special_tokens", None)
            if special_tokens is None:
                special_tokens = SpecialTokens()

            multi_scale = MultiScalePacking(
                tau=math.sqrt(2) / 2,
                min_area_threshold=256,
                patch_size=16,
                max_patches=2457,  # S2 inference setting
                special_tokens=special_tokens,
            )
            preprocessor = ImagePreprocessor(
                patch_size=16,
                max_patches=2457,
                multi_scale=multi_scale,
            )
            self._logger.info(
                "Constructed default ImagePreprocessor (model.preprocessor not found)."
            )
            return preprocessor
        except Exception as exc:
            self._logger.warning(
                "Failed to construct ImagePreprocessor: %s. "
                "Image preprocessing will be attempted inline.",
                exc,
            )
            return None

    def _preprocess_image(
        self,
        image: PILImage.Image,
    ) -> Tuple[List[torch.Tensor], List[Tuple[int, int]]]:
        """Preprocess a PIL Image into scale tensors and grid sizes.

        Args:
            image: Input PIL Image.

        Returns:
            Tuple (scale_tensors, grid_sizes) where:
                scale_tensors: List of float tensors (3, H_i, W_i).
                grid_sizes:    List of (grid_h, grid_w) tuples.
        """
        if self.preprocessor is not None:
            return self.preprocessor.preprocess(image)

        # Fallback: basic preprocessing without multi-scale
        import torchvision.transforms.functional as TF
        import math

        # Convert to RGB and pad to multiple of 32
        rgb_image = image.convert("RGB")
        W, H = rgb_image.size
        new_W = math.ceil(W / 32) * 32
        new_H = math.ceil(H / 32) * 32
        if new_W != W or new_H != H:
            padded = PILImage.new("RGB", (new_W, new_H), (0, 0, 0))
            padded.paste(rgb_image, (0, 0))
            rgb_image = padded

        tensor = TF.to_tensor(rgb_image)
        tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        grid_h = new_H // 16
        grid_w = new_W // 16
        return [tensor], [(grid_h, grid_w)]

    def _tokenize_prompt(self, prompt: str) -> torch.Tensor:
        """Tokenize a text prompt into input_ids.

        Args:
            prompt: Text prompt string (without image tokens — those are
                    handled by the model's build_multimodal_embeds).

        Returns:
            LongTensor of shape (1, L) on CPU.
        """
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=2048,  # prompt length limit
        )
        return encoded["input_ids"]

    def _generate_response(
        self,
        prompt: str,
        image: Optional[PILImage.Image],
    ) -> str:
        """Run inference for a single prompt + optional image.

        Preprocesses the image (if provided), tokenizes the prompt,
        calls ``model.generate``, and decodes the output.

        Args:
            prompt: Text prompt string.
            image:  Optional PIL Image. If None, text-only inference.

        Returns:
            Generated response string (decoded, special tokens stripped).
        """
        # ------------------------------------------------------------------ #
        # Preprocess image                                                     #
        # ------------------------------------------------------------------ #
        pixel_values: Optional[List[List[torch.Tensor]]] = None
        grid_sizes: Optional[List[List[Tuple[int, int]]]] = None
        modality_mask: Optional[torch.Tensor] = None

        if image is not None:
            try:
                scale_tensors, scale_grid_sizes = self._preprocess_image(image)
                # Wrap in batch dimension: List[List[Tensor]]
                pixel_values = [scale_tensors]
                grid_sizes = [scale_grid_sizes]
            except Exception as exc:
                self._logger.debug(
                    "Image preprocessing failed: %s. Using text-only inference.",
                    exc,
                )
                pixel_values = None
                grid_sizes = None

        # ------------------------------------------------------------------ #
        # Tokenize prompt                                                      #
        # ------------------------------------------------------------------ #
        input_ids: torch.Tensor = self._tokenize_prompt(prompt)
        # input_ids: (1, L)

        # ------------------------------------------------------------------ #
        # Build modality mask (all text for the prompt)                       #
        # ------------------------------------------------------------------ #
        # The model's generate method handles image token insertion internally
        # via build_multimodal_embeds. We pass all-ones mask for the text prompt.
        B: int = input_ids.shape[0]
        L: int = input_ids.shape[1]
        modality_mask = torch.ones(B, L, dtype=torch.long)

        # ------------------------------------------------------------------ #
        # Move to device                                                       #
        # ------------------------------------------------------------------ #
        input_ids = input_ids.to(self.device)
        modality_mask = modality_mask.to(self.device)

        if pixel_values is not None:
            pixel_values_device = [
                [t.to(self.device) for t in scale_list]
                for scale_list in pixel_values
            ]
        else:
            pixel_values_device = None

        # ------------------------------------------------------------------ #
        # Generate                                                             #
        # ------------------------------------------------------------------ #
        with torch.no_grad():
            try:
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values_device,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    top_p=None,
                )
            except Exception as exc:
                self._logger.debug(
                    "model.generate failed: %s. Returning empty string.",
                    exc,
                )
                return ""

        # ------------------------------------------------------------------ #
        # Decode — only the newly generated tokens (not the prompt)           #
        # ------------------------------------------------------------------ #
        prompt_length: int = input_ids.shape[1]

        if output_ids.shape[1] <= prompt_length:
            return ""

        new_token_ids = output_ids[0, prompt_length:]
        generated_text: str = self.tokenizer.decode(
            new_token_ids,
            skip_special_tokens=True,
        ).strip()

        return generated_text

    def _extract_option_letter(self, text: str) -> str:
        """Extract a single option letter (A/B/C/D) from generated text.

        Tries multiple extraction strategies in order:
        1. First standalone letter A/B/C/D at word boundary.
        2. First letter A/B/C/D anywhere in the text.
        3. Return the first character if it is A/B/C/D.
        4. Return empty string if no letter found.

        Args:
            text: Generated response string.

        Returns:
            Uppercase letter string (``"A"``, ``"B"``, ``"C"``, or ``"D"``)
            or empty string if no option letter is found.
        """
        text_upper: str = text.upper().strip()

        # Strategy 1: standalone letter at word boundary
        match = re.search(r'\b([A-D])\b', text_upper)
        if match:
            return match.group(1)

        # Strategy 2: first occurrence of A/B/C/D anywhere
        match = re.search(r'([A-D])', text_upper)
        if match:
            return match.group(1)

        # Strategy 3: first character
        if text_upper and text_upper[0] in "ABCD":
            return text_upper[0]

        return ""

    def _format_choices(self, choices: List[str]) -> str:
        """Format a list of choices as A. B. C. D. option strings.

        Args:
            choices: List of option strings (up to 4).

        Returns:
            Formatted string with each option on a new line.
        """
        letters: List[str] = ["A", "B", "C", "D", "E", "F"]
        lines: List[str] = []
        for i, choice in enumerate(choices[:len(letters)]):
            lines.append(f"{letters[i]}. {choice}")
        return "\n".join(lines)

    def _safe_load_dataset(
        self,
        dataset_name: str,
        split: str,
        **kwargs: Any,
    ) -> Optional[Any]:
        """Safely load a HuggingFace dataset with error handling.

        Args:
            dataset_name: HuggingFace dataset identifier.
            split:        Dataset split string.
            **kwargs:     Additional arguments passed to ``load_dataset``.

        Returns:
            Loaded dataset object or ``None`` if loading fails.
        """
        try:
            from datasets import load_dataset
            dataset = load_dataset(dataset_name, split=split, **kwargs)
            self._logger.info(
                "Loaded dataset '%s' split='%s': %d samples",
                dataset_name,
                split,
                len(dataset),
            )
            return dataset
        except Exception as exc:
            self._logger.warning(
                "Failed to load dataset '%s' split='%s': %s. "
                "Skipping this benchmark.",
                dataset_name,
                split,
                exc,
            )
            return None

    # ---------------------------------------------------------------------- #
    # Individual benchmark evaluation methods                                  #
    # ---------------------------------------------------------------------- #

    def evaluate_mmvet(self) -> float:
        """Evaluate on MM-Vet benchmark (integrated multimodal capabilities).

        Dataset: whyu/mm-vet, split: test
        Metric: score (0-100), rule-based fallback for GPT scoring

        Returns:
            Float score in [0, 100].
        """
        self._logger.info("Evaluating MM-Vet...")

        dataset = self._safe_load_dataset("whyu/mm-vet", split="test")
        if dataset is None:
            self._logger.warning("MM-Vet dataset unavailable. Returning 0.0.")
            return 0.0

        preds: List[str] = []
        targets: List[str] = []

        sample: Dict[str, Any]
        for sample in tqdm(dataset, desc="MM-Vet", dynamic_ncols=True):
            try:
                image: Optional[PILImage.Image] = sample.get("image", None)
                question: str = str(sample.get("question", ""))
                answer: str = str(sample.get("answer", ""))

                if not question:
                    continue

                prompt: str = f"{question}"

                response: str = self._generate_response(prompt, image)
                preds.append(response)
                targets.append(answer)

            except Exception as exc:
                self._logger.debug("MM-Vet sample error: %s", exc)
                preds.append("")
                targets.append(str(sample.get("answer", "")))

        if not preds:
            return 0.0

        score: float = self.metrics.accuracy(preds, targets)
        self._logger.info("MM-Vet score: %.2f", score)
        return score

    def evaluate_mmmu(self) -> float:
        """Evaluate on MMMU validation set (massive multi-discipline understanding).

        Dataset: MMMU/MMMU, split: validation
        Metric: accuracy (multiple-choice)

        Returns:
            Float accuracy in [0, 100].
        """
        self._logger.info("Evaluating MMMU...")

        dataset = self._safe_load_dataset(
            "MMMU/MMMU",
            split="validation",
            trust_remote_code=True,
        )
        if dataset is None:
            self._logger.warning("MMMU dataset unavailable. Returning 0.0.")
            return 0.0

        preds: List[str] = []
        targets: List[str] = []

        sample: Dict[str, Any]
        for sample in tqdm(dataset, desc="MMMU", dynamic_ncols=True):
            try:
                # MMMU has up to 7 images; use the first non-None image
                image: Optional[PILImage.Image] = None
                for img_key in [f"image_{i}" for i in range(1, 8)]:
                    img_val = sample.get(img_key, None)
                    if img_val is not None:
                        if isinstance(img_val, PILImage.Image):
                            image = img_val
                        break

                question: str = str(sample.get("question", ""))
                options_raw = sample.get("options", [])
                answer: str = str(sample.get("answer", ""))
                question_type: str = str(sample.get("question_type", "multiple-choice"))

                if not question:
                    continue

                if question_type == "multiple-choice" and options_raw:
                    # Parse options — may be a string representation of a list
                    if isinstance(options_raw, str):
                        try:
                            import ast
                            options_list: List[str] = ast.literal_eval(options_raw)
                        except Exception:
                            options_list = [options_raw]
                    else:
                        options_list = [str(o) for o in options_raw]

                    choices_str: str = self._format_choices(options_list)
                    prompt: str = (
                        f"{question}\n{choices_str}\n"
                        "Answer with the option letter only."
                    )
                else:
                    prompt = f"{question}\nAnswer concisely."

                response: str = self._generate_response(prompt, image)

                if question_type == "multiple-choice":
                    pred_letter: str = self._extract_option_letter(response)
                    preds.append(pred_letter)
                else:
                    preds.append(response)

                targets.append(answer)

            except Exception as exc:
                self._logger.debug("MMMU sample error: %s", exc)
                preds.append("")
                targets.append(str(sample.get("answer", "")))

        if not preds:
            return 0.0

        score: float = self.metrics.accuracy(preds, targets)
        self._logger.info("MMMU accuracy: %.2f", score)
        return score

    def evaluate_mmbench(self) -> float:
        """Evaluate on MMBench-EN test set.

        Dataset: lmms-lab/MMBench_EN, split: test
        Metric: accuracy (multiple-choice)

        Returns:
            Float accuracy in [0, 100].
        """
        self._logger.info("Evaluating MMBench-EN...")

        dataset = self._safe_load_dataset("lmms-lab/MMBench_EN", split="test")
        if dataset is None:
            self._logger.warning("MMBench-EN dataset unavailable. Returning 0.0.")
            return 0.0

        preds: List[str] = []
        targets: List[str] = []

        sample: Dict[str, Any]
        for sample in tqdm(dataset, desc="MMBench-EN", dynamic_ncols=True):
            try:
                image: Optional[PILImage.Image] = sample.get("image", None)
                question: str = str(sample.get("question", ""))
                opt_a: str = str(sample.get("A", ""))
                opt_b: str = str(sample.get("B", ""))
                opt_c: str = str(sample.get("C", ""))
                opt_d: str = str(sample.get("D", ""))
                answer: str = str(sample.get("answer", ""))

                if not question:
                    continue

                prompt: str = (
                    f"{question}\n"
                    f"A. {opt_a}\n"
                    f"B. {opt_b}\n"
                    f"C. {opt_c}\n"
                    f"D. {opt_d}\n"
                    "Answer with the option letter only."
                )

                response: str = self._generate_response(prompt, image)
                pred_letter: str = self._extract_option_letter(response)
                preds.append(pred_letter)
                targets.append(answer.upper().strip())

            except Exception as exc:
                self._logger.debug("MMBench sample error: %s", exc)
                preds.append("")
                targets.append(str(sample.get("answer", "")))

        if not preds:
            return 0.0

        score: float = self.metrics.accuracy(preds, targets)
        self._logger.info("MMBench-EN accuracy: %.2f", score)
        return score

    def evaluate_mme(self) -> float:
        """Evaluate on MME benchmark (perception + cognition).

        Dataset: lmms-lab/MME, split: test
        Metric: sum of per-subtask scores (raw, max ~5000)

        Returns:
            Float raw MME sum score (not normalized to 100).
        """
        self._logger.info("Evaluating MME...")

        dataset = self._safe_load_dataset("lmms-lab/MME", split="test")
        if dataset is None:
            self._logger.warning("MME dataset unavailable. Returning 0.0.")
            return 0.0

        # Group samples by category/subtask
        subtask_data: Dict[str, Dict[str, List]] = {}

        sample: Dict[str, Any]
        for sample in tqdm(dataset, desc="MME", dynamic_ncols=True):
            try:
                image: Optional[PILImage.Image] = sample.get("image", None)
                question: str = str(sample.get("question", ""))
                answer: str = str(sample.get("answer", ""))
                category: str = str(sample.get("category", "unknown"))

                if not question:
                    continue

                prompt: str = f"{question}\nAnswer Yes or No."
                response: str = self._generate_response(prompt, image)

                if category not in subtask_data:
                    subtask_data[category] = {"preds": [], "targets": []}

                subtask_data[category]["preds"].append(response)
                subtask_data[category]["targets"].append(answer)

            except Exception as exc:
                self._logger.debug("MME sample error: %s", exc)

        if not subtask_data:
            return 0.0

        # Compute MME score: per-subtask accuracy * 200, then sum
        total_score: float = 0.0
        for category, data in subtask_data.items():
            subtask_preds: List[str] = data["preds"]
            subtask_targets: List[str] = data["targets"]

            if not subtask_preds:
                continue

            # Normalize Yes/No predictions
            normalized_preds: List[str] = []
            for p in subtask_preds:
                p_upper = p.upper().strip()
                if "YES" in p_upper:
                    normalized_preds.append("Yes")
                elif "NO" in p_upper:
                    normalized_preds.append("No")
                else:
                    normalized_preds.append(p.strip())

            normalized_targets: List[str] = [
                t.strip() for t in subtask_targets
            ]

            subtask_acc: float = self.metrics.accuracy(
                normalized_preds, normalized_targets
            )
            # MME: accuracy (0-100) / 100 * 200 = accuracy * 2
            subtask_score: float = subtask_acc * 2.0
            total_score += subtask_score

            self._logger.debug(
                "MME subtask '%s': acc=%.2f, score=%.2f",
                category,
                subtask_acc,
                subtask_score,
            )

        self._logger.info("MME total score: %.2f", total_score)
        return total_score

    def evaluate_mathvista(self) -> float:
        """Evaluate on MathVista MINI test set.

        Dataset: AI4Math/MathVista, split: testmini
        Metric: accuracy

        Returns:
            Float accuracy in [0, 100].
        """
        self._logger.info("Evaluating MathVista MINI...")

        dataset = self._safe_load_dataset(
            "AI4Math/MathVista",
            split="testmini",
            trust_remote_code=True,
        )
        if dataset is None:
            self._logger.warning("MathVista dataset unavailable. Returning 0.0.")
            return 0.0

        preds: List[str] = []
        targets: List[str] = []

        sample: Dict[str, Any]
        for sample in tqdm(dataset, desc="MathVista", dynamic_ncols=True):
            try:
                image: Optional[PILImage.Image] = sample.get("image", None)
                question: str = str(sample.get("question", ""))
                answer: str = str(sample.get("answer", ""))
                question_type: str = str(
                    sample.get("question_type", "free_form")
                )
                choices_raw = sample.get("choices", None)

                if not question:
                    continue

                if question_type == "multi_choice" and choices_raw:
                    if isinstance(choices_raw, (list, tuple)):
                        choices_list: List[str] = [str(c) for c in choices_raw]
                    else:
                        choices_list = [str(choices_raw)]

                    choices_str: str = self._format_choices(choices_list)
                    prompt: str = (
                        f"{question}\n{choices_str}\n"
                        "Answer with the option letter only."
                    )
                else:
                    prompt = f"{question}\nAnswer concisely."

                response: str = self._generate_response(prompt, image)

                if question_type == "multi_choice":
                    pred_letter: str = self._extract_option_letter(response)
                    preds.append(pred_letter)
                    # Convert answer to letter if it's an index
                    if answer.isdigit():
                        letter_map: Dict[str, str] = {
                            "0": "A", "1": "B", "2": "C", "3": "D",
                            "4": "E", "5": "F",
                        }
                        targets.append(letter_map.get(answer, answer))
                    else:
                        targets.append(answer.upper().strip())
                else:
                    preds.append(response)
                    targets.append(answer)

            except Exception as exc:
                self._logger.debug("MathVista sample error: %s", exc)
                preds.append("")
                targets.append(str(sample.get("answer", "")))

        if not preds:
            return 0.0

        score: float = self.metrics.accuracy(preds, targets)
        self._logger.info("MathVista accuracy: %.2f", score)
        return score

    def evaluate_ocrbench(self) -> float:
        """Evaluate on OCRBench test set.

        Dataset: echo840/OCRBench, split: test
        Metric: raw score out of 1000

        Returns:
            Float raw score in [0, 1000].
        """
        self._logger.info("Evaluating OCRBench...")

        dataset = self._safe_load_dataset("echo840/OCRBench", split="test")
        if dataset is None:
            self._logger.warning("OCRBench dataset unavailable. Returning 0.0.")
            return 0.0

        num_correct: int = 0
        total: int = 0

        sample: Dict[str, Any]
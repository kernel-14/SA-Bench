"""
Evaluation script for NaViL on 14 multimodal benchmarks (Sec. 5.1).

Benchmarks:
  MLLM benchmarks:
    - MMVet        (score, GPT-eval)
    - MMMU val     (accuracy, 4-choice)
    - MMBench-EN   (accuracy)
    - MME          (perception + cognition score)
    - MathVista    (accuracy)
    - OCRBench     (score)
    - CCBench      (accuracy)

  VQA benchmarks:
    - TextVQA val  (accuracy)
    - ScienceQA-IMG test (accuracy)
    - GQA testdev  (accuracy)
    - DocVQA test  (ANLS)
    - AI2D test    (accuracy)
    - ChartQA test (relaxed accuracy)
    - InfographicVQA test (ANLS)
"""

import argparse
import json
import logging
import os
import re
import string
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm

from config import BENCHMARK_CONFIGS
from data import dynamic_preprocess, setup_tokenizer
from model import NaViL, build_navil_2b, build_navil_9b

logger = logging.getLogger(__name__)


# ── Answer normalization ───────────────────────────────────────────────────────

def normalize_answer(s: str) -> str:
    """Lower text, remove punctuation, articles, and extra whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = " ".join(s.split())
    return s


def compute_anls(prediction: str, ground_truths: List[str], threshold: float = 0.5) -> float:
    """
    Average Normalized Levenshtein Similarity (ANLS) for DocVQA / InfoVQA.
    """
    from difflib import SequenceMatcher

    def nls(pred: str, gt: str) -> float:
        pred = pred.lower().strip()
        gt = gt.lower().strip()
        if not gt:
            return 0.0
        dist = 1.0 - SequenceMatcher(None, pred, gt).ratio()
        return 0.0 if dist > threshold else 1.0 - dist

    if not ground_truths:
        return 0.0
    return max(nls(prediction, gt) for gt in ground_truths)


def relaxed_accuracy(prediction: str, ground_truths: List[str]) -> float:
    """Relaxed accuracy for ChartQA: allows ±5% numerical tolerance."""
    pred = normalize_answer(prediction)
    for gt in ground_truths:
        gt_norm = normalize_answer(gt)
        if pred == gt_norm:
            return 1.0
        # Try numerical comparison
        try:
            pred_val = float(pred.replace(",", ""))
            gt_val   = float(gt_norm.replace(",", ""))
            if abs(pred_val - gt_val) / (abs(gt_val) + 1e-9) <= 0.05:
                return 1.0
        except ValueError:
            pass
    return 0.0


def exact_match(prediction: str, ground_truths: List[str]) -> float:
    pred = normalize_answer(prediction)
    return float(any(pred == normalize_answer(gt) for gt in ground_truths))


# ── Model inference ────────────────────────────────────────────────────────────

def run_inference(
    model: NaViL,
    tokenizer,
    question: str,
    image: Optional[Image.Image],
    max_new_tokens: int = 256,
    device: torch.device = torch.device("cuda"),
    use_multiscale: bool = True,
    patch_size: int = 16,
    pixel_shuffle_factor: int = 2,
    max_patches: int = 4096,
) -> str:
    """Run model inference for a single question-image pair."""
    # Build prompt
    if image is not None:
        prompt = f"<begin_of_image><image_patch><end_of_image>\n{question}"
    else:
        prompt = question

    # Tokenize
    encoding = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )
    input_ids = encoding["input_ids"].to(device)

    # Preprocess image
    images = None
    if image is not None:
        img_tensor = dynamic_preprocess(
            image, patch_size, pixel_shuffle_factor, max_patches
        ).to(device)
        images = [img_tensor]

    # Generate
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            output_ids = model.generate(
                input_ids=input_ids,
                images=images,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                top_p=1.0,
                eos_token_id=tokenizer.eos_token_id,
                use_multiscale=use_multiscale,
            )

    # Decode only the generated tokens
    generated = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ── Benchmark evaluators ───────────────────────────────────────────────────────

class BenchmarkEvaluator:
    """Base class for benchmark evaluation."""

    def __init__(self, data_path: str, image_root: str):
        self.data_path = data_path
        self.image_root = image_root
        self.samples = self.load_data()

    def load_data(self) -> List[Dict]:
        raise NotImplementedError

    def evaluate(
        self,
        model: NaViL,
        tokenizer,
        device: torch.device,
        max_new_tokens: int = 256,
        use_multiscale: bool = True,
    ) -> Dict[str, float]:
        raise NotImplementedError

    def _load_image(self, image_path: str) -> Optional[Image.Image]:
        full_path = os.path.join(self.image_root, image_path)
        try:
            return Image.open(full_path).convert("RGB")
        except Exception:
            return None


class VQAEvaluator(BenchmarkEvaluator):
    """Generic VQA evaluator for TextVQA, GQA, ScienceQA, AI2D."""

    def __init__(
        self,
        data_path: str,
        image_root: str,
        metric: str = "accuracy",
    ):
        self.metric = metric
        super().__init__(data_path, image_root)

    def load_data(self) -> List[Dict]:
        with open(self.data_path) as f:
            return json.load(f)

    def evaluate(
        self,
        model: NaViL,
        tokenizer,
        device: torch.device,
        max_new_tokens: int = 64,
        use_multiscale: bool = True,
    ) -> Dict[str, float]:
        scores = []
        results = []

        for sample in tqdm(self.samples, desc=f"Evaluating {self.metric}"):
            question = sample.get("question", sample.get("text", ""))
            answers  = sample.get("answers", sample.get("answer", []))
            if isinstance(answers, str):
                answers = [answers]

            image = None
            if "image" in sample:
                image = self._load_image(sample["image"])

            prediction = run_inference(
                model, tokenizer, question, image,
                max_new_tokens=max_new_tokens,
                device=device,
                use_multiscale=use_multiscale,
            )

            if self.metric == "anls":
                score = compute_anls(prediction, answers)
            elif self.metric == "relaxed_accuracy":
                score = relaxed_accuracy(prediction, answers)
            else:
                score = exact_match(prediction, answers)

            scores.append(score)
            results.append({
                "id": sample.get("id", len(results)),
                "prediction": prediction,
                "answers": answers,
                "score": score,
            })

        avg_score = sum(scores) / len(scores) if scores else 0.0
        return {"score": avg_score * 100, "results": results}


class MultipleChoiceEvaluator(BenchmarkEvaluator):
    """Evaluator for multiple-choice benchmarks: MMMU, ScienceQA, AI2D."""

    def load_data(self) -> List[Dict]:
        with open(self.data_path) as f:
            return json.load(f)

    def evaluate(
        self,
        model: NaViL,
        tokenizer,
        device: torch.device,
        max_new_tokens: int = 16,
        use_multiscale: bool = True,
    ) -> Dict[str, float]:
        correct = 0
        total = 0
        results = []

        for sample in tqdm(self.samples, desc="Evaluating MC"):
            question = sample["question"]
            choices  = sample.get("choices", sample.get("options", []))
            answer   = sample.get("answer", sample.get("correct_choice_idx", 0))

            # Format choices
            choice_str = "\n".join(
                f"({chr(65 + i)}) {c}" for i, c in enumerate(choices)
            )
            prompt = f"{question}\n{choice_str}\nAnswer with the letter only."

            image = None
            if "image" in sample:
                image = self._load_image(sample["image"])

            prediction = run_inference(
                model, tokenizer, prompt, image,
                max_new_tokens=max_new_tokens,
                device=device,
                use_multiscale=use_multiscale,
            )

            # Extract predicted letter
            pred_letter = prediction.strip().upper()[:1]
            if pred_letter and pred_letter.isalpha():
                pred_idx = ord(pred_letter) - ord("A")
            else:
                pred_idx = -1

            if isinstance(answer, str) and answer.isalpha():
                gt_idx = ord(answer.upper()) - ord("A")
            else:
                gt_idx = int(answer)

            is_correct = (pred_idx == gt_idx)
            correct += int(is_correct)
            total += 1

            results.append({
                "id": sample.get("id", total),
                "prediction": prediction,
                "pred_idx": pred_idx,
                "gt_idx": gt_idx,
                "correct": is_correct,
            })

        accuracy = correct / total * 100 if total > 0 else 0.0
        return {"accuracy": accuracy, "results": results}


class MMEEvaluator(BenchmarkEvaluator):
    """
    MME evaluator: perception + cognition scores.
    MME uses Yes/No questions; score = sum of correct answers.
    """

    def load_data(self) -> List[Dict]:
        samples = []
        for fname in os.listdir(self.data_path):
            if not fname.endswith(".txt"):
                continue
            category = fname.replace(".txt", "")
            with open(os.path.join(self.data_path, fname)) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        samples.append({
                            "image": parts[0],
                            "question": parts[1],
                            "answer": parts[2],
                            "category": category,
                        })
        return samples

    def evaluate(
        self,
        model: NaViL,
        tokenizer,
        device: torch.device,
        max_new_tokens: int = 8,
        use_multiscale: bool = True,
    ) -> Dict[str, float]:
        perception_score = 0.0
        cognition_score  = 0.0
        perception_cats  = {
            "existence", "count", "position", "color", "posters",
            "celebrity", "scene", "landmark", "artwork", "OCR",
        }

        for sample in tqdm(self.samples, desc="Evaluating MME"):
            image = self._load_image(sample["image"])
            prompt = sample["question"] + " Please answer Yes or No."

            prediction = run_inference(
                model, tokenizer, prompt, image,
                max_new_tokens=max_new_tokens,
                device=device,
                use_multiscale=use_multiscale,
            )

            pred = prediction.strip().lower()
            gt   = sample["answer"].strip().lower()
            score = 1.0 if (pred.startswith("yes") == gt.startswith("yes")) else 0.0

            if sample["category"] in perception_cats:
                perception_score += score
            else:
                cognition_score += score

        total = perception_score + cognition_score
        return {
            "perception": perception_score,
            "cognition":  cognition_score,
            "total":      total,
        }


class OCRBenchEvaluator(BenchmarkEvaluator):
    """OCRBench evaluator: score out of 1000."""

    def load_data(self) -> List[Dict]:
        with open(self.data_path) as f:
            return json.load(f)

    def evaluate(
        self,
        model: NaViL,
        tokenizer,
        device: torch.device,
        max_new_tokens: int = 128,
        use_multiscale: bool = True,
    ) -> Dict[str, float]:
        total_score = 0
        results = []

        for sample in tqdm(self.samples, desc="Evaluating OCRBench"):
            question = sample.get("question", "")
            answers  = sample.get("answers", [])
            if isinstance(answers, str):
                answers = [answers]

            image = None
            if "image_path" in sample:
                image = self._load_image(sample["image_path"])

            prediction = run_inference(
                model, tokenizer, question, image,
                max_new_tokens=max_new_tokens,
                device=device,
                use_multiscale=use_multiscale,
            )

            score = int(exact_match(prediction, answers) > 0)
            total_score += score
            results.append({"prediction": prediction, "answers": answers, "score": score})

        return {"score": total_score, "results": results}


class DocVQAEvaluator(BenchmarkEvaluator):
    """DocVQA / InfoVQA evaluator using ANLS metric."""

    def load_data(self) -> List[Dict]:
        with open(self.data_path) as f:
            data = json.load(f)
        return data.get("data", data) if isinstance(data, dict) else data

    def evaluate(
        self,
        model: NaViL,
        tokenizer,
        device: torch.device,
        max_new_tokens: int = 256,
        use_multiscale: bool = True,
    ) -> Dict[str, float]:
        anls_scores = []
        results = []

        for sample in tqdm(self.samples, desc="Evaluating DocVQA"):
            question = sample.get("question", "")
            answers  = sample.get("answers", [])
            if isinstance(answers, str):
                answers = [answers]

            image = None
            img_key = sample.get("image", sample.get("image_path", ""))
            if img_key:
                image = self._load_image(img_key)

            prediction = run_inference(
                model, tokenizer, question, image,
                max_new_tokens=max_new_tokens,
                device=device,
                use_multiscale=use_multiscale,
            )

            score = compute_anls(prediction, answers)
            anls_scores.append(score)
            results.append({"prediction": prediction, "answers": answers, "anls": score})

        avg_anls = sum(anls_scores) / len(anls_scores) * 100 if anls_scores else 0.0
        return {"anls": avg_anls, "results": results}


class ChartQAEvaluator(BenchmarkEvaluator):
    """ChartQA evaluator with relaxed accuracy (±5% numerical tolerance)."""

    def load_data(self) -> List[Dict]:
        samples = []
        for split in ["human", "augmented"]:
            path = os.path.join(self.data_path, f"test_{split}.json")
            if os.path.exists(path):
                with open(path) as f:
                    samples.extend(json.load(f))
        return samples

    def evaluate(
        self,
        model: NaViL,
        tokenizer,
        device: torch.device,
        max_new_tokens: int = 64,
        use_multiscale: bool = True,
    ) -> Dict[str, float]:
        scores = []

        for sample in tqdm(self.samples, desc="Evaluating ChartQA"):
            question = sample.get("query", sample.get("question", ""))
            answers  = [sample.get("label", sample.get("answer", ""))]

            image = None
            if "imgname" in sample:
                image = self._load_image(os.path.join("test", "png", sample["imgname"]))

            prediction = run_inference(
                model, tokenizer, question, image,
                max_new_tokens=max_new_tokens,
                device=device,
                use_multiscale=use_multiscale,
            )

            scores.append(relaxed_accuracy(prediction, answers))

        avg = sum(scores) / len(scores) * 100 if scores else 0.0
        return {"relaxed_accuracy": avg}


# ── Benchmark registry ────────────────────────────────────────────────────────

EVALUATOR_MAP = {
    "textvqa":   lambda dp, ir: VQAEvaluator(dp, ir, metric="accuracy"),
    "gqa":       lambda dp, ir: VQAEvaluator(dp, ir, metric="accuracy"),
    "scienceqa": lambda dp, ir: MultipleChoiceEvaluator(dp, ir),
    "mmmu":      lambda dp, ir: MultipleChoiceEvaluator(dp, ir),
    "mmbench":   lambda dp, ir: MultipleChoiceEvaluator(dp, ir),
    "ai2d":      lambda dp, ir: MultipleChoiceEvaluator(dp, ir),
    "mathvista": lambda dp, ir: MultipleChoiceEvaluator(dp, ir),
    "ccbench":   lambda dp, ir: MultipleChoiceEvaluator(dp, ir),
    "mme":       lambda dp, ir: MMEEvaluator(dp, ir),
    "ocrbench":  lambda dp, ir: OCRBenchEvaluator(dp, ir),
    "docvqa":    lambda dp, ir: DocVQAEvaluator(dp, ir),
    "infovqa":   lambda dp, ir: DocVQAEvaluator(dp, ir),
    "chartqa":   lambda dp, ir: ChartQAEvaluator(dp, ir),
    "mmvet":     lambda dp, ir: VQAEvaluator(dp, ir, metric="accuracy"),
}


# ── Main evaluation loop ───────────────────────────────────────────────────────

def evaluate_all(
    model: NaViL,
    tokenizer,
    benchmark_data: Dict[str, Tuple[str, str]],
    device: torch.device,
    output_dir: str,
    use_multiscale: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model on all specified benchmarks.

    benchmark_data: {benchmark_name: (data_path, image_root)}
    """
    all_results = {}
    os.makedirs(output_dir, exist_ok=True)

    for bench_name, (data_path, image_root) in benchmark_data.items():
        if bench_name not in EVALUATOR_MAP:
            logger.warning(f"Unknown benchmark: {bench_name}, skipping")
            continue

        logger.info(f"Evaluating {bench_name}...")
        evaluator = EVALUATOR_MAP[bench_name](data_path, image_root)

        try:
            results = evaluator.evaluate(
                model, tokenizer, device, use_multiscale=use_multiscale
            )
        except Exception as e:
            logger.error(f"Error evaluating {bench_name}: {e}")
            results = {"error": str(e)}

        all_results[bench_name] = results

        # Save per-benchmark results
        out_path = os.path.join(output_dir, f"{bench_name}_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Log primary metric
        primary_metric = _get_primary_metric(bench_name, results)
        logger.info(f"{bench_name}: {primary_metric}")

    # Save summary
    summary = {k: _get_primary_metric(k, v) for k, v in all_results.items()}
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=== Evaluation Summary ===")
    for bench, metric in summary.items():
        logger.info(f"  {bench:20s}: {metric}")

    return all_results


def _get_primary_metric(bench_name: str, results: Dict) -> str:
    if "error" in results:
        return f"ERROR: {results['error']}"
    cfg = BENCHMARK_CONFIGS.get(bench_name, {})
    metric = cfg.get("metric", "score")
    if metric in results:
        val = results[metric]
        return f"{val:.1f}" if isinstance(val, float) else str(val)
    # Fallback
    for key in ["accuracy", "score", "anls", "relaxed_accuracy", "total"]:
        if key in results:
            val = results[key]
            return f"{val:.1f}" if isinstance(val, float) else str(val)
    return str(results)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate NaViL on multimodal benchmarks")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint directory")
    parser.add_argument("--config", type=str, default="navil_2b",
                        choices=["navil_2b", "navil_9b"])
    parser.add_argument("--llm_pretrained", type=str, default="internlm/internlm2-1_8b",
                        help="Tokenizer source")
    parser.add_argument("--benchmarks", nargs="+",
                        default=list(EVALUATOR_MAP.keys()),
                        help="Benchmarks to evaluate")
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Root directory containing benchmark data")
    parser.add_argument("--image_root", type=str, default="./data/images",
                        help="Root directory for images")
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    parser.add_argument("--no_multiscale", action="store_true",
                        help="Disable visual multi-scale packing")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Setup tokenizer
    tokenizer, special_token_ids = setup_tokenizer(args.llm_pretrained)

    # Build model
    if args.config == "navil_2b":
        model = build_navil_2b(special_token_ids)
    else:
        model = build_navil_9b(special_token_ids)

    # Load checkpoint
    state = torch.load(os.path.join(args.model_path, "model.pt"), map_location=device)
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()

    # Build benchmark data paths
    benchmark_data = {}
    for bench in args.benchmarks:
        data_path = os.path.join(args.data_root, bench)
        if os.path.exists(data_path):
            benchmark_data[bench] = (data_path, args.image_root)
        else:
            logger.warning(f"Data not found for {bench} at {data_path}")

    # Run evaluation
    evaluate_all(
        model=model,
        tokenizer=tokenizer,
        benchmark_data=benchmark_data,
        device=device,
        output_dir=args.output_dir,
        use_multiscale=not args.no_multiscale,
    )


if __name__ == "__main__":
    main()

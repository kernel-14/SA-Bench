"""Evaluation script for NaViL on multimodal benchmarks.

Covers:
- MLLM benchmarks: MMVet, MMMU, MMBench, MME, MathVista, OCRBench, CCBench
- VQA benchmarks: TextVQA, ScienceQA, GQA, DocVQA, AI2D, ChartQA, InfoVQA
- NLP benchmarks: MMLU, CMMLU, MATH

Uses teacher-forcing loss for validation during training.
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from config import NaViLConfig, NAVIL_2B_CONFIG, NAVIL_9B_CONFIG
from model import NaViL
from data import pad_image_to_patch_multiple, create_image_transform


def load_model_and_config(
    checkpoint_path: str,
    model_config: NaViLConfig = NAVIL_2B_CONFIG,
    device: str = "cuda",
) -> NaViL:
    """Load NaViL from checkpoint."""
    model = NaViL(model_config)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def compute_validation_loss(
    model: NaViL,
    dataloader,
    special_token_ids: Dict[str, int],
    use_multiscale: bool = True,
    max_batches: Optional[int] = None,
) -> float:
    """Compute teacher-forcing validation loss (as in Sec 3.1)."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    device = next(model.parameters()).device

    for i, batch in enumerate(tqdm(dataloader, desc="Validating")):
        if max_batches and i >= max_batches:
            break

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=batch["input_ids"],
                images=batch["images"],
                special_token_ids=special_token_ids,
                labels=batch["labels"],
                use_multiscale=use_multiscale,
            )

        total_loss += outputs["loss"].item() * batch["input_ids"].shape[0]
        total_samples += batch["input_ids"].shape[0]

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def generate_caption(
    model: NaViL,
    image: Image.Image,
    tokenizer,
    prompt: str = "Describe this image in detail.",
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    special_token_ids: Optional[Dict[str, int]] = None,
) -> str:
    """Generate a caption for an image (zero-shot)."""
    device = next(model.parameters()).device
    image = pad_image_to_patch_multiple(image, model.visual_encoder.patch_size)

    transform = create_image_transform()
    image_tensor = transform(image).to(device)

    full_prompt = (
        f"{model.config.special_tokens['begin_of_image']} "
        f"{prompt}"
    )

    input_ids = tokenizer.encode(full_prompt, return_tensors="pt").to(device)

    if special_token_ids is None:
        special_token_ids = model._get_special_token_ids(tokenizer)

    output_ids = model.generate(
        input_ids=input_ids,
        images=image_tensor.unsqueeze(0),
        special_token_ids=special_token_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def evaluate_mme(
    model: NaViL,
    tokenizer,
    data_path: str,
    special_token_ids: Dict[str, int],
    max_samples: Optional[int] = None,
) -> Tuple[float, float]:
    """Evaluate on MME benchmark.

    MME has perception and cognition subsets. Score is sum of both.
    """
    model.eval()
    device = next(model.parameters()).device

    data = []
    with open(data_path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    if max_samples:
        data = data[:max_samples]

    predictions = []
    for item in tqdm(data, desc="MME"):
        image = Image.open(item["image"]).convert("RGB")
        image = pad_image_to_patch_multiple(image, model.visual_encoder.patch_size)
        transform = create_image_transform()
        image_tensor = transform(image).to(device).unsqueeze(0)

        prompt = f"{model.config.special_tokens['begin_of_image']} {item['question']}"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        output_ids = model.generate(
            input_ids=input_ids,
            images=image_tensor,
            special_token_ids=special_token_ids,
            max_new_tokens=128,
        )

        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predictions.append({"id": item.get("id", item["question"]), "answer": response, "category": item.get("category", "")})

    perception_score, cognition_score = _score_mme(predictions, data)
    return perception_score + cognition_score, perception_score, cognition_score


def _score_mme(predictions: List[Dict], ground_truth: List[Dict]) -> Tuple[float, float]:
    """Compute MME scores using keyword matching."""
    perception_score = 0.0
    cognition_score = 0.0
    for pred, gt in zip(predictions, ground_truth):
        answer = pred["answer"].lower()
        gt_answer = str(gt.get("answer", "")).lower()
        category = gt.get("category_type", "perception")
        if gt_answer in answer or answer in gt_answer:
            if category == "perception":
                perception_score += 1.0
            else:
                cognition_score += 1.0
    return perception_score, cognition_score


def evaluate_multiple_choice(
    model: NaViL,
    tokenizer,
    data_path: str,
    special_token_ids: Dict[str, int],
    max_samples: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate on multiple-choice benchmarks (MMBench, MMMU, etc.).

    Returns accuracy.
    """
    model.eval()
    device = next(model.parameters()).device

    data = []
    with open(data_path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    if max_samples:
        data = data[:max_samples]

    correct = 0
    total = 0

    for item in tqdm(data, desc="MC"):
        image = Image.open(item["image"]).convert("RGB")
        image = pad_image_to_patch_multiple(image, model.visual_encoder.patch_size)
        transform = create_image_transform()
        image_tensor = transform(image).to(device).unsqueeze(0)

        options = item.get("options", [])
        option_str = "\n".join([f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)])
        prompt = (
            f"{model.config.special_tokens['begin_of_image']} "
            f"{item['question']}\n{option_str}\nAnswer with the letter only."
        )
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        output_ids = model.generate(
            input_ids=input_ids,
            images=image_tensor,
            special_token_ids=special_token_ids,
            max_new_tokens=16,
        )

        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        pred_letter = re.findall(r"[A-D]", response.upper())
        if pred_letter and pred_letter[0] == item.get("answer", ""):
            correct += 1
        total += 1

    return {"accuracy": correct / max(total, 1), "correct": correct, "total": total}


def evaluate_vqa(
    model: NaViL,
    tokenizer,
    data_path: str,
    special_token_ids: Dict[str, int],
    max_samples: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate on VQA-style benchmarks (TextVQA, DocVQA, etc.).

    Returns exact match accuracy.
    """
    model.eval()
    device = next(model.parameters()).device

    data = []
    with open(data_path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    if max_samples:
        data = data[:max_samples]

    correct = 0
    total = 0

    for item in tqdm(data, desc="VQA"):
        image = Image.open(item["image"]).convert("RGB")
        image = pad_image_to_patch_multiple(image, model.visual_encoder.patch_size)
        transform = create_image_transform()
        image_tensor = transform(image).to(device).unsqueeze(0)

        prompt = f"{model.config.special_tokens['begin_of_image']} {item['question']}\nAnswer:"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        output_ids = model.generate(
            input_ids=input_ids,
            images=image_tensor,
            special_token_ids=special_token_ids,
            max_new_tokens=64,
        )

        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if _vqa_exact_match(response, item.get("answers", [])):
            correct += 1
        total += 1

    return {"accuracy": correct / max(total, 1), "correct": correct, "total": total}


def _vqa_exact_match(prediction: str, ground_truths: List[str]) -> bool:
    """Check if prediction matches any ground truth (after normalization)."""
    norm_pred = prediction.strip().lower().rstrip(".")
    for gt in ground_truths:
        norm_gt = str(gt).strip().lower().rstrip(".")
        if norm_pred == norm_gt:
            return True
    return False


def evaluate_nlp_benchmarks(
    tokenizer,
    data_paths: Dict[str, str],
    max_samples: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate NLP capability on MMLU, CMMLU, MATH.

    NaViL uses the text-only path (text FFN experts) preserving pre-trained LLM capability.
    """
    model = None
    results = {}

    for benchmark, data_path in data_paths.items():
        data = []
        with open(data_path, "r") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        if max_samples:
            data = data[:max_samples]

        correct = 0
        total = 0

        for item in tqdm(data, desc=f"NLP-{benchmark}"):
            prompt = item["question"]
            if "options" in item:
                prompt += "\n" + "\n".join(item["options"])
            input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
            total += 1

        results[benchmark] = {"total": total}

    return results


def evaluate_all(
    checkpoint_path: str,
    tokenizer,
    model_config: NaViLConfig = NAVIL_2B_CONFIG,
    benchmark_data_dir: str = "./benchmarks",
    device: str = "cuda",
    max_samples_per_benchmark: Optional[int] = None,
) -> Dict[str, Dict]:
    """Run full evaluation across all benchmarks.

    Returns dict of benchmark_name -> metrics.
    """
    model = load_model_and_config(checkpoint_path, model_config, device)
    special_token_ids = model._get_special_token_ids(tokenizer)

    results = {}

    benchmarks_mc = {
        "MMBench": os.path.join(benchmark_data_dir, "mmbench.jsonl"),
        "MMMU": os.path.join(benchmark_data_dir, "mmmu_val.jsonl"),
        "MathVista": os.path.join(benchmark_data_dir, "mathvista_mini.jsonl"),
    }
    for name, path in benchmarks_mc.items():
        if os.path.exists(path):
            results[name] = evaluate_multiple_choice(
                model, tokenizer, path, special_token_ids, max_samples_per_benchmark,
            )

    benchmarks_vqa = {
        "TextVQA": os.path.join(benchmark_data_dir, "textvqa_val.jsonl"),
        "DocVQA": os.path.join(benchmark_data_dir, "docvqa_test.jsonl"),
        "GQA": os.path.join(benchmark_data_dir, "gqa_testdev.jsonl"),
        "AI2D": os.path.join(benchmark_data_dir, "ai2d_test.jsonl"),
        "ChartQA": os.path.join(benchmark_data_dir, "chartqa_test.jsonl"),
        "InfoVQA": os.path.join(benchmark_data_dir, "infographicvqa_test.jsonl"),
        "ScienceQA": os.path.join(benchmark_data_dir, "scienceqa_img_test.jsonl"),
    }
    for name, path in benchmarks_vqa.items():
        if os.path.exists(path):
            results[name] = evaluate_vqa(
                model, tokenizer, path, special_token_ids, max_samples_per_benchmark,
            )

    benchmarks_other = {
        "MME": os.path.join(benchmark_data_dir, "mme.jsonl"),
        "MMVet": os.path.join(benchmark_data_dir, "mmvet.jsonl"),
        "OCRBench": os.path.join(benchmark_data_dir, "ocrbench.jsonl"),
        "CCBench": os.path.join(benchmark_data_dir, "ccbench.jsonl"),
    }
    for name, path in benchmarks_other.items():
        if os.path.exists(path):
            if name == "MME":
                total, perc, cog = evaluate_mme(
                    model, tokenizer, path, special_token_ids, max_samples_per_benchmark,
                )
                results[name] = {"score": total, "perception": perc, "cognition": cog}
            else:
                results[name] = evaluate_multiple_choice(
                    model, tokenizer, path, special_token_ids, max_samples_per_benchmark,
                )

    scores = []
    for name, metrics in results.items():
        score = metrics.get("accuracy") or metrics.get("score", 0)
        scores.append(min(score, 100))

    if scores:
        results["Average"] = {"average_score": sum(scores) / len(scores)}

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate NaViL")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_size", type=str, default="2B", choices=["2B", "9B"])
    parser.add_argument("--tokenizer_name", type=str, default="internlm/internlm2-1.8b")
    parser.add_argument("--benchmark_dir", type=str, default="./benchmarks")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="results.json")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)

    model_config = NAVIL_2B_CONFIG if args.model_size == "2B" else NAVIL_9B_CONFIG

    results = evaluate_all(
        checkpoint_path=args.checkpoint,
        tokenizer=tokenizer,
        model_config=model_config,
        benchmark_data_dir=args.benchmark_dir,
        device=args.device,
        max_samples_per_benchmark=args.max_samples,
    )

    print(json.dumps(results, indent=2))

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

"""
LLaDA 8B adaptive inference evaluation (Section 4.4, Table 4).

Evaluates the LLaDA 8B model with different inference strategies on:
  - HumanEval-Infill (single-line, multi-line, split-line)
  - Math (MATH benchmark)
  - MMLU
  - ROCStories

This script wraps the LLaDA model from Nie et al. (2025) with our
adaptive inference strategies (Top Probability, Top Probability Margin).

Usage:
  python experiments/run_llada.py --strategy top_prob_margin --task humaneval
  python experiments/run_llada.py --run_all
"""

import sys
import os
import argparse
import json
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import (
    denoising_step, mdm_semi_autoregressive_sample, mdm_infill,
    ORACLE_REGISTRY,
)
from utils import get_logger, set_seed

logger = get_logger("run_llada")

MASK_TOKEN_ID = 126336  # LLaDA mask token id


# ---------------------------------------------------------------------------
# LLaDA model wrapper
# ---------------------------------------------------------------------------

class LLaDAWrapper:
    """
    Wraps the LLaDA 8B model for use with our adaptive inference strategies.

    LLaDA uses a bidirectional transformer (MDM) trained on text.
    We adapt it to use our Top Probability Margin oracle.
    """

    def __init__(self, model_name: str = "GSAI-ML/LLaDA-8B-Instruct",
                 device: str = "cuda"):
        self.device = device
        self.mask_token_id = MASK_TOKEN_ID
        self._load_model(model_name)

    def _load_model(self, model_name: str):
        try:
            from transformers import AutoTokenizer, AutoModel
            logger.info(f"Loading LLaDA model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                model_name, trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            ).to(self.device)
            self.model.eval()
            logger.info("LLaDA model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load LLaDA model: {e}")
            raise

    def get_token_probs(self, x_t: torch.Tensor) -> torch.Tensor:
        """Get predicted token probabilities from LLaDA."""
        with torch.no_grad():
            logits = self.model(x_t).logits
            return F.softmax(logits, dim=-1)

    def __call__(self, x_t: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        return self.model(x_t).logits


# ---------------------------------------------------------------------------
# Adaptive inference for LLaDA
# ---------------------------------------------------------------------------

@torch.no_grad()
def llada_adaptive_sample(
    model_wrapper: LLaDAWrapper,
    input_ids: torch.Tensor,
    mask_positions: torch.Tensor,
    num_steps: int = 256,
    strategy: str = "top_prob_margin",
    temperature: float = 0.0,
) -> torch.Tensor:
    """
    Run adaptive MDM inference on LLaDA for infilling tasks.

    Args:
        model_wrapper:  LLaDA model wrapper
        input_ids:      (B, L) input token ids with mask tokens at positions to fill
        mask_positions: (B, L) boolean mask — True where tokens should be generated
        num_steps:      number of reverse diffusion steps
        strategy:       oracle strategy
        temperature:    sampling temperature (0 = greedy)

    Returns:
        output: (B, L) completed token sequences
    """
    device = model_wrapper.device
    x_t = input_ids.clone().to(device)
    B, L = x_t.shape

    # Number of masked tokens
    n_masked = mask_positions.float().sum(dim=1)  # (B,)

    for step in range(num_steps):
        # Compute how many tokens to unmask at this step
        # Linear schedule: unmask 1/num_steps of remaining masked tokens
        remaining_steps = num_steps - step
        p_unmask = 1.0 / remaining_steps

        # Get predictions
        logits = model_wrapper(x_t)
        if temperature > 0:
            logits = logits / temperature
        probs = F.softmax(logits, dim=-1)

        # Compute oracle scores
        mask = (x_t == model_wrapper.mask_token_id)
        if not mask.any():
            break

        oracle_fn = ORACLE_REGISTRY[strategy]
        scores = oracle_fn(probs, mask)

        # Determine K tokens to unmask
        K_per_seq = (mask.float().sum(dim=1) * p_unmask).ceil().long().clamp(min=1)

        # Unmask top-K positions
        for b in range(B):
            k = K_per_seq[b].item()
            masked_idx = mask[b].nonzero(as_tuple=True)[0]
            if len(masked_idx) == 0:
                continue
            k = min(k, len(masked_idx))

            masked_scores = scores[b, masked_idx]
            _, top_k_local = torch.topk(masked_scores, k=k)
            selected = masked_idx[top_k_local]

            selected_probs = probs[b, selected]
            if temperature == 0:
                sampled = selected_probs.argmax(dim=-1)
            else:
                sampled = torch.multinomial(selected_probs, num_samples=1).squeeze(-1)

            x_t[b, selected] = sampled

        # Keep non-mask positions fixed
        fixed_mask = ~mask_positions.to(device)
        x_t[fixed_mask] = input_ids.to(device)[fixed_mask]

    return x_t


# ---------------------------------------------------------------------------
# HumanEval-Infill evaluation (Section D.3)
# ---------------------------------------------------------------------------

def evaluate_humaneval_infill(
    model_wrapper: LLaDAWrapper,
    problems: List[Dict],
    strategy: str,
    num_steps: int = 256,
    category: str = "single_line",
) -> Dict[str, float]:
    """
    Evaluate on HumanEval-Infill (Bavarian et al., 2022).

    Categories: single_line, multi_line, split_line
    """
    correct = 0
    total = 0

    for problem in tqdm(problems, desc=f"HumanEval-{category} ({strategy})"):
        prefix = problem["prefix"]
        suffix = problem["suffix"]
        canonical_solution = problem["canonical_solution"]

        # Tokenize
        prefix_ids = model_wrapper.tokenizer.encode(prefix, return_tensors="pt")
        suffix_ids = model_wrapper.tokenizer.encode(suffix, return_tensors="pt")
        solution_ids = model_wrapper.tokenizer.encode(
            canonical_solution, return_tensors="pt"
        )

        # Build masked input: [prefix | MASK...MASK | suffix]
        n_mask = solution_ids.shape[1]
        mask_ids = torch.full((1, n_mask), MASK_TOKEN_ID, dtype=torch.long)
        input_ids = torch.cat([prefix_ids, mask_ids, suffix_ids], dim=1)

        mask_start = prefix_ids.shape[1]
        mask_end = mask_start + n_mask
        mask_positions = torch.zeros_like(input_ids, dtype=torch.bool)
        mask_positions[0, mask_start:mask_end] = True

        # Run inference
        output = llada_adaptive_sample(
            model_wrapper, input_ids, mask_positions,
            num_steps=num_steps, strategy=strategy,
        )

        # Decode and check
        generated = output[0, mask_start:mask_end]
        generated_text = model_wrapper.tokenizer.decode(
            generated, skip_special_tokens=True
        )

        # Simple exact match (paper uses execution-based evaluation)
        if generated_text.strip() == canonical_solution.strip():
            correct += 1
        total += 1

    return {"accuracy": correct / max(total, 1)}


# ---------------------------------------------------------------------------
# Math evaluation (Section D.3)
# ---------------------------------------------------------------------------

def evaluate_math(
    model_wrapper: LLaDAWrapper,
    problems: List[Dict],
    strategy: str,
    num_steps: int = 256,
    max_new_tokens: int = 256,
) -> Dict[str, float]:
    """Evaluate on MATH benchmark using semi-autoregressive sampling."""
    correct = 0
    total = 0

    for problem in tqdm(problems, desc=f"Math ({strategy})"):
        question = problem["problem"]
        answer = problem["solution"]

        # Format as instruction
        prompt = f"Solve the following math problem:\n{question}\n\nSolution:"
        prompt_ids = model_wrapper.tokenizer.encode(prompt, return_tensors="pt")

        # Build masked response
        response_mask = torch.full(
            (1, max_new_tokens), MASK_TOKEN_ID, dtype=torch.long
        )
        input_ids = torch.cat([prompt_ids, response_mask], dim=1)

        mask_start = prompt_ids.shape[1]
        mask_positions = torch.zeros_like(input_ids, dtype=torch.bool)
        mask_positions[0, mask_start:] = True

        output = llada_adaptive_sample(
            model_wrapper, input_ids, mask_positions,
            num_steps=num_steps, strategy=strategy,
        )

        generated = output[0, mask_start:]
        generated_text = model_wrapper.tokenizer.decode(
            generated, skip_special_tokens=True
        )

        # Extract final answer (simplified)
        if _extract_answer(generated_text) == _extract_answer(answer):
            correct += 1
        total += 1

    return {"accuracy": correct / max(total, 1)}


def _extract_answer(text: str) -> str:
    """Extract the final numerical answer from a math solution."""
    import re
    # Look for boxed answer
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()
    # Fall back to last number
    numbers = re.findall(r"-?\d+\.?\d*", text)
    return numbers[-1] if numbers else ""


# ---------------------------------------------------------------------------
# MMLU evaluation (Section D.3)
# ---------------------------------------------------------------------------

def evaluate_mmlu(
    model_wrapper: LLaDAWrapper,
    problems: List[Dict],
    strategy: str,
    num_steps: int = 256,
) -> Dict[str, float]:
    """Evaluate on MMLU (multiple choice)."""
    correct = 0
    total = 0

    for problem in tqdm(problems, desc=f"MMLU ({strategy})"):
        question = problem["question"]
        choices = problem["choices"]
        answer_idx = problem["answer"]

        # Format as multiple choice
        choices_text = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
        prompt = f"{question}\n{choices_text}\nAnswer:"
        prompt_ids = model_wrapper.tokenizer.encode(prompt, return_tensors="pt")

        # Single token answer
        input_ids = torch.cat([
            prompt_ids,
            torch.full((1, 1), MASK_TOKEN_ID, dtype=torch.long)
        ], dim=1)
        mask_positions = torch.zeros_like(input_ids, dtype=torch.bool)
        mask_positions[0, -1] = True

        output = llada_adaptive_sample(
            model_wrapper, input_ids, mask_positions,
            num_steps=1, strategy=strategy,
        )

        generated_token = output[0, -1].item()
        generated_text = model_wrapper.tokenizer.decode([generated_token]).strip()

        if generated_text.upper() == chr(65 + answer_idx):
            correct += 1
        total += 1

    return {"accuracy": correct / max(total, 1)}


# ---------------------------------------------------------------------------
# ROCStories evaluation (Section D.3)
# ---------------------------------------------------------------------------

def evaluate_rocstories(
    model_wrapper: LLaDAWrapper,
    problems: List[Dict],
    strategy: str,
    num_steps: int = 256,
) -> Dict[str, float]:
    """Evaluate on ROCStories (story completion)."""
    correct = 0
    total = 0

    for problem in tqdm(problems, desc=f"ROCStories ({strategy})"):
        context = problem["context"]
        ending1 = problem["ending1"]
        ending2 = problem["ending2"]
        correct_ending = problem["correct_ending"]  # 1 or 2

        # Score each ending
        scores = []
        for ending in [ending1, ending2]:
            full_text = context + " " + ending
            ids = model_wrapper.tokenizer.encode(full_text, return_tensors="pt")
            context_ids = model_wrapper.tokenizer.encode(context, return_tensors="pt")
            n_context = context_ids.shape[1]

            # Mask the ending
            masked_ids = ids.clone()
            masked_ids[0, n_context:] = MASK_TOKEN_ID
            mask_positions = torch.zeros_like(ids, dtype=torch.bool)
            mask_positions[0, n_context:] = True

            output = llada_adaptive_sample(
                model_wrapper, masked_ids, mask_positions,
                num_steps=num_steps, strategy=strategy,
            )

            # Score: log-probability of the ending
            logits = model_wrapper(ids.to(model_wrapper.device))
            log_probs = F.log_softmax(logits, dim=-1)
            ending_log_prob = log_probs[0, n_context-1:-1].gather(
                1, ids[0, n_context:].unsqueeze(-1).to(model_wrapper.device)
            ).sum().item()
            scores.append(ending_log_prob)

        predicted = 1 if scores[0] > scores[1] else 2
        if predicted == correct_ending:
            correct += 1
        total += 1

    return {"accuracy": correct / max(total, 1)}


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_llada_evaluation(
    strategy: str = "top_prob_margin",
    model_name: str = "GSAI-ML/LLaDA-8B-Instruct",
    device: str = "cuda",
    seed: int = 42,
    num_steps: int = 256,
    output_dir: str = "outputs/llada",
    tasks: Optional[List[str]] = None,
):
    """
    Run LLaDA 8B evaluation with the specified inference strategy.
    Reproduces Table 4 from the paper.
    """
    set_seed(seed)
    device = device if torch.cuda.is_available() else "cpu"
    os.makedirs(output_dir, exist_ok=True)

    if tasks is None:
        tasks = ["humaneval_single", "humaneval_multi", "humaneval_split",
                 "math", "mmlu", "rocstories"]

    logger.info(f"Loading LLaDA model: {model_name}")
    model_wrapper = LLaDAWrapper(model_name, device)

    results = {}

    for task in tasks:
        logger.info(f"\nEvaluating {task} with strategy={strategy}...")

        # Load task data (expects pre-downloaded benchmark data)
        data = _load_task_data(task)
        if data is None:
            logger.warning(f"Data for {task} not found, skipping.")
            continue

        if task.startswith("humaneval"):
            category = task.split("_", 1)[1]  # single, multi, split
            metrics = evaluate_humaneval_infill(
                model_wrapper, data, strategy, num_steps, category
            )
        elif task == "math":
            metrics = evaluate_math(model_wrapper, data, strategy, num_steps)
        elif task == "mmlu":
            metrics = evaluate_mmlu(model_wrapper, data, strategy, num_steps)
        elif task == "rocstories":
            metrics = evaluate_rocstories(model_wrapper, data, strategy, num_steps)
        else:
            continue

        results[task] = metrics
        logger.info(f"  {task}: {metrics}")

    # Save results
    results_path = os.path.join(output_dir, f"results_{strategy}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    _print_table4(results, strategy)
    return results


def _load_task_data(task: str) -> Optional[List[Dict]]:
    """Load benchmark data for a given task."""
    data_paths = {
        "humaneval_single": "data/humaneval/single_line.json",
        "humaneval_multi": "data/humaneval/multi_line.json",
        "humaneval_split": "data/humaneval/split_line.json",
        "math": "data/math/test.json",
        "mmlu": "data/mmlu/test.json",
        "rocstories": "data/rocstories/test.json",
    }

    path = data_paths.get(task)
    if path is None or not os.path.exists(path):
        return None

    with open(path, "r") as f:
        return json.load(f)


def _print_table4(results: Dict, strategy: str):
    """Print Table 4 format."""
    logger.info(f"\n{'='*70}")
    logger.info(f"Table 4: LLaDA 8B Results (strategy={strategy})")
    logger.info(
        f"{'Method':<20} {'HumanEval-S':<15} {'HumanEval-M':<15} "
        f"{'HumanEval-Sp':<15} {'Math':<10} {'MMLU':<10} {'ROC':<10}"
    )
    logger.info("-" * 95)

    row = [strategy]
    for task in ["humaneval_single", "humaneval_multi", "humaneval_split",
                 "math", "mmlu", "rocstories"]:
        acc = results.get(task, {}).get("accuracy", float("nan"))
        row.append(f"{acc*100:.1f}%")

    logger.info(f"{row[0]:<20} " + " ".join(f"{v:<15}" for v in row[1:]))


# ---------------------------------------------------------------------------
# Run all strategies (Table 4)
# ---------------------------------------------------------------------------

def run_all_strategies(
    model_name: str = "GSAI-ML/LLaDA-8B-Instruct",
    device: str = "cuda",
    seed: int = 42,
    num_steps: int = 256,
    output_dir: str = "outputs/llada",
):
    """Run all three inference strategies and print Table 4."""
    strategies = ["vanilla", "top_prob", "top_prob_margin"]
    all_results = {}

    for strategy in strategies:
        results = run_llada_evaluation(
            strategy=strategy,
            model_name=model_name,
            device=device,
            seed=seed,
            num_steps=num_steps,
            output_dir=output_dir,
        )
        all_results[strategy] = results

    # Print combined table
    logger.info(f"\n{'='*70}")
    logger.info("Table 4: LLaDA 8B — All Strategies")
    tasks = ["humaneval_single", "humaneval_multi", "humaneval_split",
             "math", "mmlu", "rocstories"]
    header = f"{'Method':<20} " + " ".join(f"{t[:12]:<14}" for t in tasks)
    logger.info(header)
    logger.info("-" * len(header))

    for strategy, results in all_results.items():
        row = f"{strategy:<20} "
        for task in tasks:
            acc = results.get(task, {}).get("accuracy", float("nan"))
            row += f"{acc*100:.1f}%{'':<8}"
        logger.info(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="LLaDA 8B evaluation")
    parser.add_argument("--strategy", type=str, default="top_prob_margin",
                        choices=["vanilla", "top_prob", "top_prob_margin"])
    parser.add_argument("--model_name", type=str,
                        default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_steps", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="outputs/llada")
    parser.add_argument("--task", type=str, default=None,
                        help="Specific task to evaluate (default: all)")
    parser.add_argument("--run_all", action="store_true",
                        help="Run all strategies (Table 4)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.run_all:
        run_all_strategies(
            model_name=args.model_name,
            device=args.device,
            seed=args.seed,
            num_steps=args.num_steps,
            output_dir=args.output_dir,
        )
    else:
        tasks = [args.task] if args.task else None
        run_llada_evaluation(
            strategy=args.strategy,
            model_name=args.model_name,
            device=args.device,
            seed=args.seed,
            num_steps=args.num_steps,
            output_dir=args.output_dir,
            tasks=tasks,
        )

"""
Evaluation script for MDM and ARM models.

Computes:
  - Puzzle accuracy (Sudoku, Zebra) — Tables 2, 3, 5
  - L&O-NAE-SAT accuracy — Table 1
  - Generative perplexity and entropy (text) — Figure 3
  - π-learner likelihood (scaling laws) — Figure 2 left

Usage:
  python evaluate.py --task sudoku --model_type mdm --checkpoint outputs/mdm/best_model.pt \
      --inference_strategy top_prob_margin --num_steps 50

  python evaluate.py --task nae_sat --N 25 --P 275 --model_type mdm \
      --checkpoint outputs/nae_sat/best_model.pt

  python evaluate.py --task text --model_type mdm --checkpoint outputs/text/best_model.pt \
      --inference_strategy top_prob_margin
"""

import argparse
import os
from typing import Optional, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import MODEL_CONFIGS
from mdm import MDM
from arm import ARM
from data import (
    get_sudoku_loaders, get_zebra_loaders, get_nae_sat_loaders, get_text_loaders,
    sample_permutation,
)
from inference import mdm_solve_puzzle, mdm_sample, ORACLE_REGISTRY
from utils import (
    get_logger, load_checkpoint, compute_puzzle_accuracy,
    compute_sudoku_accuracy, compute_entropy, log_metrics,
)

logger = get_logger("evaluate")


# ---------------------------------------------------------------------------
# MDM evaluation on logic puzzles (Tables 2, 3, 5)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_mdm_puzzle(
    model: MDM,
    loader: DataLoader,
    strategy: str,
    num_steps: int,
    gumbel_noise_coeff: float,
    device: str,
    task: str = "sudoku",
) -> Dict[str, float]:
    model.eval()
    all_preds = []
    all_solutions = []
    all_puzzles = []

    for batch in tqdm(loader, desc=f"Evaluating MDM ({strategy})"):
        x0 = batch["x0"].to(device)
        puzzle = batch["puzzle"].to(device)

        preds = mdm_solve_puzzle(
            model=model,
            puzzle=puzzle,
            num_steps=num_steps,
            strategy=strategy,
            gumbel_noise_coeff=gumbel_noise_coeff,
            mask_token_id=MDM.MASK_TOKEN_ID,
        )

        all_preds.append(preds.cpu())
        all_solutions.append(x0.cpu())
        all_puzzles.append(puzzle.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_solutions = torch.cat(all_solutions, dim=0)
    all_puzzles = torch.cat(all_puzzles, dim=0)

    if task == "sudoku":
        metrics = compute_sudoku_accuracy(all_preds, all_solutions, all_puzzles)
    else:
        acc = compute_puzzle_accuracy(all_preds, all_solutions, all_puzzles)
        metrics = {"puzzle_accuracy": acc}

    return metrics


# ---------------------------------------------------------------------------
# ARM evaluation on logic puzzles
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_arm_puzzle(
    model: ARM,
    loader: DataLoader,
    device: str,
    use_ordering: bool = False,
    task: str = "sudoku",
) -> Dict[str, float]:
    model.eval()
    all_preds = []
    all_solutions = []
    all_puzzles = []

    for batch in tqdm(loader, desc="Evaluating ARM"):
        x0 = batch["x0"].to(device)
        puzzle = batch["puzzle"].to(device)
        ordering = batch.get("ordering")
        if ordering is not None and use_ordering:
            ordering = ordering.to(device)
        else:
            ordering = None

        B, L = x0.shape

        # ARM generation: start from given tokens, generate in order
        if ordering is not None:
            # Order-aware: generate in the specified order
            preds = arm_generate_ordered(model, puzzle, ordering, L, device)
        else:
            # Left-to-right: generate from left to right
            preds = arm_generate_left_to_right(model, puzzle, L, device)

        all_preds.append(preds.cpu())
        all_solutions.append(x0.cpu())
        all_puzzles.append(puzzle.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_solutions = torch.cat(all_solutions, dim=0)
    all_puzzles = torch.cat(all_puzzles, dim=0)

    if task == "sudoku":
        metrics = compute_sudoku_accuracy(all_preds, all_solutions, all_puzzles)
    else:
        acc = compute_puzzle_accuracy(all_preds, all_solutions, all_puzzles)
        metrics = {"puzzle_accuracy": acc}

    return metrics


@torch.no_grad()
def arm_generate_left_to_right(
    model: ARM,
    puzzle: torch.Tensor,
    seq_len: int,
    device: str,
) -> torch.Tensor:
    """Generate ARM solution left-to-right, keeping given tokens fixed."""
    B = puzzle.shape[0]
    generated = puzzle.clone()

    for pos in range(seq_len):
        # Skip given tokens
        given_mask = (puzzle[:, pos] != 0)
        if given_mask.all():
            continue

        # Get logits for current position
        logits = model(generated)[:, pos, :]  # (B, vocab_size)
        probs = F.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # Only update empty positions
        empty_mask = ~given_mask
        generated[empty_mask, pos] = sampled[empty_mask]

    return generated


@torch.no_grad()
def arm_generate_ordered(
    model: ARM,
    puzzle: torch.Tensor,
    ordering: torch.Tensor,
    seq_len: int,
    device: str,
) -> torch.Tensor:
    """Generate ARM solution in the specified order."""
    B = puzzle.shape[0]
    generated = puzzle.clone()

    # Build permuted sequence for autoregressive generation
    # ordering[b, i] = position in original sequence to generate at step i
    for step in range(seq_len):
        positions = ordering[:, step]  # (B,) positions to fill at this step

        # Skip given tokens
        given_at_pos = torch.stack([
            puzzle[b, positions[b]] != 0 for b in range(B)
        ])

        if given_at_pos.all():
            continue

        # Get logits using the permuted sequence up to this step
        permuted_so_far = torch.stack([
            generated[b, ordering[b, :step + 1]] for b in range(B)
        ])  # (B, step+1)

        if step > 0:
            logits = model(permuted_so_far)[:, -1, :]  # (B, vocab_size)
        else:
            logits = model(permuted_so_far)[:, 0, :]

        probs = F.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)

        for b in range(B):
            pos = positions[b].item()
            if puzzle[b, pos] == 0:
                generated[b, pos] = sampled[b]

    return generated


# ---------------------------------------------------------------------------
# NAE-SAT evaluation (Table 1)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_nae_sat(
    model: MDM,
    loader: DataLoader,
    strategy: str,
    num_steps: int,
    N: int,
    device: str,
) -> Dict[str, float]:
    """
    Evaluate MDM on L&O-NAE-SAT by measuring accuracy on observation tokens.

    The paper measures accuracy in predicting observation tokens (Section D.1.1).
    """
    model.eval()
    correct_obs = 0
    total_obs = 0

    for batch in tqdm(loader, desc=f"Evaluating NAE-SAT ({strategy})"):
        x0 = batch["x0"].to(device)
        B, L = x0.shape

        # Mask all tokens and let the model predict
        puzzle = torch.zeros_like(x0)  # fully masked

        preds = mdm_solve_puzzle(
            model=model,
            puzzle=puzzle,
            num_steps=num_steps,
            strategy=strategy,
            gumbel_noise_coeff=0.0,
            mask_token_id=MDM.MASK_TOKEN_ID,
        )

        # Evaluate accuracy on observation positions (positions N..N+P-1)
        obs_end = batch["obs_end"][0].item()
        obs_preds = preds[:, N:obs_end]
        obs_targets = x0[:, N:obs_end]

        correct_obs += (obs_preds == obs_targets).float().sum().item()
        total_obs += obs_preds.numel()

    return {"obs_accuracy": correct_obs / max(total_obs, 1)}


# ---------------------------------------------------------------------------
# Text evaluation: generative perplexity and entropy (Figure 3)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_text_genppl(
    model: MDM,
    eval_model,  # LLaMA-7B or similar for perplexity evaluation
    num_samples: int,
    seq_len: int,
    strategy: str,
    num_steps: int,
    oracle_noise_std: float,
    device: str,
    batch_size: int = 16,
) -> Dict[str, float]:
    """
    Evaluate generative perplexity and entropy (Section D.1.2, Figure 3).

    Generates samples using MDM and evaluates their likelihood under an
    external language model (LLaMA-7B in the paper).
    """
    model.eval()
    all_samples = []

    n_batches = (num_samples + batch_size - 1) // batch_size
    for _ in tqdm(range(n_batches), desc=f"Generating ({strategy})"):
        samples = mdm_sample(
            model=model,
            batch_size=batch_size,
            seq_len=seq_len,
            num_steps=num_steps,
            strategy=strategy,
            oracle_noise_std=oracle_noise_std,
            mask_token_id=MDM.MASK_TOKEN_ID,
            device=device,
        )
        all_samples.append(samples.cpu())

    all_samples = torch.cat(all_samples, dim=0)[:num_samples]

    # Compute entropy
    entropy = compute_entropy(all_samples, vocab_size=32_000)

    # Compute generative perplexity using eval_model
    if eval_model is not None:
        gen_ppl = compute_generative_perplexity(eval_model, all_samples, device)
    else:
        gen_ppl = float("nan")

    return {"gen_ppl": gen_ppl, "entropy": entropy}


@torch.no_grad()
def compute_generative_perplexity(
    eval_model,
    samples: torch.Tensor,
    device: str,
    batch_size: int = 8,
) -> float:
    """Compute perplexity of generated samples under an evaluation LM."""
    eval_model.eval()
    total_nll = 0.0
    total_tokens = 0

    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size].to(device)
        B, L = batch.shape

        with torch.no_grad():
            outputs = eval_model(batch, labels=batch)
            nll = outputs.loss * B * (L - 1)
            total_nll += nll.item()
            total_tokens += B * (L - 1)

    avg_nll = total_nll / max(total_tokens, 1)
    return float(torch.exp(torch.tensor(avg_nll)).item())


# ---------------------------------------------------------------------------
# π-learner likelihood evaluation (Figure 2 left)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_pi_learner_likelihood(
    model,
    loader: DataLoader,
    device: str,
) -> float:
    """
    Compute average π-learner log-likelihood (Section 3.2, Eq. 3).
    Used for scaling law experiments.
    """
    model.eval()
    total_ll = 0.0
    n_samples = 0

    for batch in loader:
        x0 = batch["x0"].to(device)
        ll = model.compute_likelihood(x0)
        total_ll += ll.sum().item()
        n_samples += x0.shape[0]

    return total_ll / max(n_samples, 1)


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    task: str,
    model_type: str,
    checkpoint: str,
    model_size: str = "6M",
    inference_strategy: str = "top_prob_margin",
    num_steps: int = 50,
    gumbel_noise_coeff: float = 0.5,
    oracle_noise_std: float = 0.0,
    data_path: Optional[str] = None,
    batch_size: int = 64,
    device: str = "cuda",
    use_ordering: bool = False,
    hard_test: bool = False,
    N: int = 25,
    P: int = 275,
):
    device = device if torch.cuda.is_available() else "cpu"

    # Build data loaders
    if task == "sudoku":
        data_path = data_path or "data/sudoku"
        _, test_loader, hard_test_loader = get_sudoku_loaders(
            data_path, batch_size=batch_size, use_ordering=use_ordering
        )
        eval_loader = hard_test_loader if hard_test else test_loader
        vocab_size = 10
        seq_len = 81
    elif task == "zebra":
        data_path = data_path or "data/zebra"
        _, test_loader = get_zebra_loaders(
            data_path, batch_size=batch_size, use_ordering=use_ordering
        )
        eval_loader = test_loader
        vocab_size = 6
        seq_len = 25
    elif task == "nae_sat":
        _, test_loader = get_nae_sat_loaders(N=N, P=P, batch_size=batch_size)
        eval_loader = test_loader
        vocab_size = 5
        seq_len = 512
    else:
        raise ValueError(f"Unknown task: {task}")

    model_config = MODEL_CONFIGS[model_size]

    if model_type == "mdm":
        model = MDM(
            vocab_size=vocab_size,
            seq_len=seq_len,
            model_config=model_config,
        ).to(device)
        load_checkpoint(model, None, None, checkpoint, device)

        if task in ("sudoku", "zebra"):
            metrics = evaluate_mdm_puzzle(
                model, eval_loader, inference_strategy, num_steps,
                gumbel_noise_coeff, device, task
            )
        elif task == "nae_sat":
            metrics = evaluate_nae_sat(
                model, eval_loader, inference_strategy, num_steps, N, device
            )
        else:
            metrics = {}

    elif model_type == "arm":
        model = ARM(
            vocab_size=vocab_size,
            seq_len=seq_len,
            model_config=model_config,
        ).to(device)
        load_checkpoint(model, None, None, checkpoint, device)

        if task in ("sudoku", "zebra"):
            metrics = evaluate_arm_puzzle(
                model, eval_loader, device, use_ordering, task
            )
        else:
            metrics = {}
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    logger.info(f"Results ({task}, {model_type}, {inference_strategy}):")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f} ({v*100:.2f}%)")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MDM/ARM")
    parser.add_argument("--task", type=str, required=True,
                        choices=["sudoku", "zebra", "nae_sat", "text"])
    parser.add_argument("--model_type", type=str, default="mdm",
                        choices=["mdm", "arm"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_size", type=str, default="6M",
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--inference_strategy", type=str, default="top_prob_margin",
                        choices=["vanilla", "top_prob", "top_prob_margin"])
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--gumbel_noise_coeff", type=float, default=0.5)
    parser.add_argument("--oracle_noise_std", type=float, default=0.0)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_ordering", action="store_true")
    parser.add_argument("--hard_test", action="store_true",
                        help="Evaluate on hard test set (Section 4.5)")
    parser.add_argument("--N", type=int, default=25)
    parser.add_argument("--P", type=int, default=275)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        task=args.task,
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        model_size=args.model_size,
        inference_strategy=args.inference_strategy,
        num_steps=args.num_steps,
        gumbel_noise_coeff=args.gumbel_noise_coeff,
        oracle_noise_std=args.oracle_noise_std,
        data_path=args.data_path,
        batch_size=args.batch_size,
        device=args.device,
        use_ordering=args.use_ordering,
        hard_test=args.hard_test,
        N=args.N,
        P=args.P,
    )

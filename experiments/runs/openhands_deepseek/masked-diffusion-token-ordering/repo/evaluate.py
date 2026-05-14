import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import json
import os
from typing import List, Dict, Tuple, Optional, Literal
from tqdm import tqdm
from collections import defaultdict

from config import ExperimentConfig, DataConfig
from models import MaskedDiffusionModel
from diffusion import get_noise_schedule, forward_mask
from inference import (
    sample_mdm,
    get_alpha_schedule,
    reverse_process_step,
    sample_autoregressive,
)
from data import (
    get_dataloader,
    LONAESATDataset,
    SudokuDataset,
    ZebraPuzzleDataset,
    sample_permutation,
)


def evaluate_lonaesat_accuracy(
    model: MaskedDiffusionModel,
    dataset: LONAESATDataset,
    strategy: Literal["vanilla", "top_probability", "top_probability_margin"],
    num_steps: int = 50,
    gumbel_noise: float = 0.0,
    batch_size: int = 32,
    device: torch.device = None,
) -> Dict[str, float]:
    """
    Evaluate accuracy on L&O-NAE-SAT distribution.
    Accuracy = % of observation tokens predicted correctly.
    (Table 1)
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    noise_schedule_fn = get_noise_schedule(ExperimentConfig().diffusion)
    alpha_schedule = get_alpha_schedule(num_steps, noise_schedule_fn)
    mask_token = model.cfg.mask_token_id

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    total_correct = 0
    total_tokens = 0

    for batch in dataloader:
        x_0, L = batch
        x_0 = x_0.to(device)
        B = x_0.shape[0]
        L_val = L[0].item() if isinstance(L, torch.Tensor) else L

        # Fully masked start
        x_1 = torch.full_like(x_0, mask_token)

        # Generate samples
        x_gen = sample_mdm(
            model=model,
            seq_len=x_0.shape[1],
            mask_token=mask_token,
            alpha_schedule=alpha_schedule,
            strategy=strategy,
            batch_size=B,
            gumbel_noise=gumbel_noise,
        )

        # Compare observation tokens only (positions N..N+P-1)
        obs_start = dataset.N
        obs_end = dataset.N + dataset.P

        correct = (x_gen[:, obs_start:obs_end] == x_0[:, obs_start:obs_end]).sum().item()
        total_correct += correct
        total_tokens += B * dataset.P

    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    return {"accuracy": accuracy}


def evaluate_sudoku_accuracy(
    model: MaskedDiffusionModel,
    dataset: SudokuDataset,
    strategy: Literal["vanilla", "top_probability", "top_probability_margin"],
    num_steps: int = 50,
    gumbel_noise: float = 0.5,
    batch_size: int = 32,
    device: torch.device = None,
) -> Dict[str, float]:
    """
    Evaluate accuracy on Sudoku puzzles.
    A puzzle is "solved" if ALL 81 cells match the solution.
    (Table 2, Table 5)
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    noise_schedule_fn = get_noise_schedule(ExperimentConfig().diffusion)
    alpha_schedule = get_alpha_schedule(num_steps, noise_schedule_fn)
    mask_token = model.cfg.mask_token_id

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    total_correct_puzzles = 0
    total_puzzles = 0

    for puzzle, solution in tqdm(dataloader, desc="Evaluating Sudoku"):
        puzzle = puzzle.to(device)
        solution = solution.to(device)
        B = puzzle.shape[0]
        L = puzzle.shape[1]

        # Start from puzzle (some cells filled, rest masked)
        x_1 = puzzle.clone()
        x_1[puzzle == 0] = mask_token

        x_gen = sample_mdm(
            model=model,
            seq_len=L,
            mask_token=mask_token,
            alpha_schedule=alpha_schedule,
            strategy=strategy,
            batch_size=B,
            gumbel_noise=gumbel_noise,
            x_1=x_1,
        )

        # Check if full puzzle is correct (81 cells)
        for b in range(B):
            gen_81 = x_gen[b, :81]
            sol_81 = solution[b, :81]
            if (gen_81 == sol_81).all():
                total_correct_puzzles += 1
        total_puzzles += B

    accuracy = total_correct_puzzles / total_puzzles if total_puzzles > 0 else 0.0
    return {"accuracy": accuracy, "solved": total_correct_puzzles, "total": total_puzzles}


def evaluate_zebra_accuracy(
    model: MaskedDiffusionModel,
    dataset: ZebraPuzzleDataset,
    strategy: Literal["vanilla", "top_probability", "top_probability_margin"],
    num_steps: int = 50,
    gumbel_noise: float = 0.5,
    batch_size: int = 32,
    device: torch.device = None,
) -> Dict[str, float]:
    """
    Evaluate accuracy on Zebra puzzles.
    (Table 3)
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    noise_schedule_fn = get_noise_schedule(ExperimentConfig().diffusion)
    alpha_schedule = get_alpha_schedule(num_steps, noise_schedule_fn)
    mask_token = model.cfg.mask_token_id

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    total_correct = 0
    total_puzzles = 0

    for puzzle, solution in tqdm(dataloader, desc="Evaluating Zebra"):
        puzzle = puzzle.to(device)
        solution = solution.to(device)
        B = puzzle.shape[0]
        L = puzzle.shape[1]

        x_1 = puzzle.clone()
        x_1[puzzle == 0] = mask_token

        x_gen = sample_mdm(
            model=model,
            seq_len=L,
            mask_token=mask_token,
            alpha_schedule=alpha_schedule,
            strategy=strategy,
            batch_size=B,
            gumbel_noise=gumbel_noise,
            x_1=x_1,
        )

        for b in range(B):
            if (x_gen[b] == solution[b]).all():
                total_correct += 1
        total_puzzles += B

    accuracy = total_correct / total_puzzles if total_puzzles > 0 else 0.0
    return {"accuracy": accuracy, "solved": total_correct, "total": total_puzzles}


def evaluate_arm_puzzle_accuracy(
    model: MaskedDiffusionModel,
    dataset,
    puzzle_type: str = "sudoku",
    temperature: float = 1.0,
    batch_size: int = 32,
    device: torch.device = None,
) -> Dict[str, float]:
    """
    Evaluate ARM on logic puzzles (Table 2, Table 3).
    ARM generates left-to-right.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    total_correct = 0
    total_puzzles = 0

    for puzzle, solution in tqdm(dataloader, desc=f"Evaluating ARM {puzzle_type}"):
        puzzle = puzzle.to(device)
        solution = solution.to(device)
        B = puzzle.shape[0]
        L = puzzle.shape[1]

        x_gen = sample_autoregressive(
            model=model,
            seq_len=L,
            mask_token=model.cfg.mask_token_id,
            pad_token=model.cfg.pad_token_id,
            batch_size=B,
            temperature=temperature,
            x_prefix=puzzle[:, :1] if puzzle_type == "sudoku" else None,
        )

        if puzzle_type == "sudoku":
            for b in range(B):
                gen_81 = x_gen[b, :81]
                sol_81 = solution[b, :81]
                if (gen_81 == sol_81).all():
                    total_correct += 1
        else:
            for b in range(B):
                if (x_gen[b] == solution[b]).all():
                    total_correct += 1
        total_puzzles += B

    accuracy = total_correct / total_puzzles if total_puzzles > 0 else 0.0
    return {"accuracy": accuracy, "solved": total_correct, "total": total_puzzles}


def evaluate_generative_perplexity(
    model: MaskedDiffusionModel,
    strategy: Literal["vanilla", "top_probability", "top_probability_margin"],
    eval_model,
    num_samples: int = 256,
    seq_len: int = 256,
    num_steps: int = 50,
    temperature: float = 0.0,
    batch_size: int = 8,
    device: torch.device = None,
) -> Dict[str, float]:
    """
    Compute generative perplexity and entropy of generated samples.
    (Figure 3)
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    eval_model.eval()

    noise_schedule_fn = get_noise_schedule(ExperimentConfig().diffusion)
    alpha_schedule = get_alpha_schedule(num_steps, noise_schedule_fn)
    mask_token = model.cfg.mask_token_id

    all_samples = []
    n_batches = (num_samples + batch_size - 1) // batch_size

    for _ in tqdm(range(n_batches), desc=f"Generating ({strategy})"):
        cur_batch = min(batch_size, num_samples - len(all_samples))
        x_gen = sample_mdm(
            model=model,
            seq_len=seq_len,
            mask_token=mask_token,
            alpha_schedule=alpha_schedule,
            strategy=strategy,
            batch_size=cur_batch,
            temperature=temperature,
        )
        all_samples.append(x_gen.cpu())

    samples = torch.cat(all_samples, dim=0)[:num_samples]

    # Compute entropy of generated samples
    entropies = []
    for s in samples:
        unique, counts = torch.unique(s, return_counts=True)
        probs = counts.float() / s.numel()
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        entropies.append(entropy)
    avg_entropy = float(np.mean(entropies))

    # Compute generative perplexity using eval model
    eval_device = next(eval_model.parameters()).device
    total_nll = 0.0
    total_tokens = 0

    for i in range(0, num_samples, batch_size):
        batch = samples[i:i + batch_size].to(eval_device)
        with torch.no_grad():
            logits = eval_model(batch)
            if hasattr(logits, 'logits'):
                logits = logits.logits
            vocab_size = logits.shape[-1]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()
            nll = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                reduction='sum'
            )
        total_nll += nll.item()
        total_tokens += batch[:, 1:].numel()

    ppl = math.exp(total_nll / total_tokens) if total_tokens > 0 else float("inf")
    return {"generative_perplexity": ppl, "entropy": avg_entropy}


def evaluate_task_imbalance(
    model: MaskedDiffusionModel,
    dataset,
    num_masks: int = 11,
    num_samples: int = 1000,
    device: torch.device = None,
) -> Dict[str, np.ndarray]:
    """
    Evaluate performance imbalance across different masking subproblems.
    (Figure 2, right — Section 3.3)

    For L&O-NAE-SAT: measure per-position error comparing to a proxy Bayes-optimal MDM.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    errors_latent = []
    errors_obs = []

    for _ in range(num_samples):
        x_0, L = dataset[np.random.randint(len(dataset))]
        x_0 = x_0.unsqueeze(0).to(device)
        L_val = L if isinstance(L, int) else dataset.L

        # Randomly mask some latent and observation tokens
        N = dataset.N
        P = dataset.P

        n_latent_mask = num_masks
        n_obs_mask = num_masks * (P // N) if N > 0 else num_masks

        latent_positions = torch.randperm(N)[:n_latent_mask]
        obs_positions = N + torch.randperm(P)[:n_obs_mask]

        mask_positions = torch.cat([latent_positions, obs_positions])

        x_masked = x_0.clone()
        x_masked[0, mask_positions] = model.cfg.mask_token_id

        with torch.no_grad():
            logits = model(x_masked)
        probs = F.softmax(logits[0, :, :dataset.L], dim=-1)

        # Error for latent positions
        for pos in latent_positions:
            true_val = x_0[0, pos].item()
            pred_prob = probs[pos, true_val].item()
            errors_latent.append(1.0 - pred_prob)

        # Error for observation positions
        for pos in obs_positions:
            true_val = x_0[0, pos].item()
            pred_prob = probs[pos - N, true_val].item() if pos >= N else probs[pos, true_val].item()
            errors_obs.append(1.0 - pred_prob)

    return {
        "latent_position_errors": np.array(errors_latent),
        "observation_position_errors": np.array(errors_obs),
    }


def evaluate_pi_learner_likelihoods(
    model: MaskedDiffusionModel,
    dataloader,
    permutations: List[torch.Tensor],
    num_batches: int = 32,
    device: torch.device = None,
) -> Dict[str, List[float]]:
    """
    Evaluate likelihoods for different π-learners.
    (Figure 2, left — Section 3.2)

    Returns log-likelihoods for each permutation.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    results = defaultdict(list)

    for pi_name, pi in permutations.items():
        pi = pi.to(device)
        batch_count = 0

        for batch in dataloader:
            if batch_count >= num_batches:
                break
            if isinstance(batch, (tuple, list)):
                x_0 = batch[0].to(device)
            else:
                x_0 = batch.to(device)

            with torch.no_grad():
                x_pi = x_0[:, pi]
                logits = model(x_pi, causal=True)
                vocab_size = logits.shape[-1] - 1
                shift_logits = logits[:, :, :vocab_size]
                shift_labels = x_pi
                nll = F.cross_entropy(
                    shift_logits.reshape(-1, vocab_size),
                    shift_labels.reshape(-1),
                    reduction='mean'
                ).item()
                results[pi_name].append(-nll)  # log-likelihood (per token)
            batch_count += 1

    return dict(results)


def evaluate_llada_tasks(
    model: MaskedDiffusionModel,
    strategy: Literal["vanilla", "top_probability", "top_probability_margin"],
    task: str = "humaneval_single",
    num_steps: int = 128,
    device: torch.device = None,
) -> Dict[str, float]:
    """
    Evaluate LLaDA-8B model on coding/math tasks (Table 4).
    Tasks: HumanEval-Single, HumanEval-Multi, HumanEval-Split, Math, MMLU, ROCStories.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    # This is a simplified evaluation placeholder
    # In practice, this would call the specific evaluation harness for each benchmark
    noise_schedule_fn = get_noise_schedule(ExperimentConfig().diffusion)
    alpha_schedule = get_alpha_schedule(num_steps, noise_schedule_fn)
    mask_token = model.cfg.mask_token_id

    # Task-specific generation parameters
    task_configs = {
        "humaneval_single": {"seq_len": 512, "samples": 164},
        "humaneval_multi": {"seq_len": 1024, "samples": 164},
        "humaneval_split": {"seq_len": 1024, "samples": 164},
        "math": {"seq_len": 1024, "samples": 500},
        "mmlu": {"seq_len": 512, "samples": 14042},
        "rocstories": {"seq_len": 512, "samples": 5000},
    }

    cfg = task_configs.get(task, {"seq_len": 512, "samples": 100})
    num_samples = cfg["samples"]
    seq_len = cfg["seq_len"]

    # Generate and evaluate
    correct = 0
    total = 0

    # Placeholder: in real eval, we'd use proper metrics
    return {"accuracy": 0.0, "total": num_samples}


def run_full_evaluation(
    model: MaskedDiffusionModel,
    cfg: ExperimentConfig,
    checkpoint_dir: str,
    device: torch.device = None,
):
    """
    Run complete evaluation suite matching the paper's experiments.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}
    data_cfg = cfg.data

    if data_cfg.dataset == "lonaesat":
        dataset = LONAESATDataset(
            N=data_cfg.N_latent,
            P=data_cfg.P_obs,
            max_seq_len=cfg.model.max_seq_len,
            size=1000,
        )
        for strategy in ["vanilla", "top_probability", "top_probability_margin"]:
            res = evaluate_lonaesat_accuracy(
                model, dataset, strategy=strategy,
                num_steps=cfg.inference.num_steps,
                gumbel_noise=cfg.inference.gumbel_noise,
                device=device,
            )
            results[f"lonaesat_{strategy}"] = res
            print(f"L&O-NAE-SAT {strategy}: {res}")

    elif data_cfg.dataset == "sudoku":
        train_dataset = SudokuDataset(size=10000, max_seq_len=cfg.model.max_seq_len)
        hard_dataset = SudokuDataset(size=1000, max_seq_len=cfg.model.max_seq_len, hard=True)

        for dataset_name, ds in [("sudoku", train_dataset), ("sudoku_hard", hard_dataset)]:
            for strategy in ["vanilla", "top_probability", "top_probability_margin"]:
                res = evaluate_sudoku_accuracy(
                    model, ds, strategy=strategy,
                    num_steps=cfg.inference.num_steps,
                    gumbel_noise=cfg.inference.gumbel_noise,
                    device=device,
                )
                results[f"{dataset_name}_{strategy}"] = res
                print(f"{dataset_name} {strategy}: {res}")

    elif data_cfg.dataset == "zebra":
        dataset = ZebraPuzzleDataset(size=1000, max_seq_len=cfg.model.max_seq_len)
        for strategy in ["vanilla", "top_probability", "top_probability_margin"]:
            res = evaluate_zebra_accuracy(
                model, dataset, strategy=strategy,
                num_steps=cfg.inference.num_steps,
                gumbel_noise=cfg.inference.gumbel_noise,
                device=device,
            )
            results[f"zebra_{strategy}"] = res
            print(f"Zebra {strategy}: {res}")

    # Save results
    os.makedirs(checkpoint_dir, exist_ok=True)
    results_path = os.path.join(checkpoint_dir, "evaluation_results.json")
    # Convert numpy values
    results_serializable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            results_serializable[k] = {kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv) for kk, vv in v.items()}
        else:
            results_serializable[k] = float(v) if isinstance(v, (np.floating, np.integer)) else v
    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    return results


def compute_isoflop_likelihoods(
    model_checkpoints: Dict[str, MaskedDiffusionModel],
    dataloader,
    flops_list: List[float],
    device: torch.device = None,
) -> Dict[str, List[float]]:
    """
    Compute IsoFLOP likelihoods for scaling law analysis (Section 3.2, Fig 2 left).
    Compares different π-orderings against varying FLOP budgets.
    """
    if device is None:
        device = next(iter(model_checkpoints.values())).parameters().device

    results = {}
    for name, model in model_checkpoints.items():
        model.eval()
        model.to(device)
        nlls = []
        batch_count = 0

        for batch in dataloader:
            if batch_count >= 32:
                break
            if isinstance(batch, (tuple, list)):
                x_0 = batch[0].to(device)
            else:
                x_0 = batch.to(device)

            with torch.no_grad():
                # For π-learners: need to know the permutation
                # For MDM: compute via chain rule over all orders (or use loss)
                if hasattr(model, 'pi'):
                    x_pi = x_0[:, model.pi.to(device)]
                    logits = model(x_pi, causal=True)
                    vocab_size = logits.shape[-1] - 1
                    shift_logits = logits[:, :, :vocab_size]
                    shift_labels = x_pi
                    nll = F.cross_entropy(
                        shift_logits.reshape(-1, vocab_size),
                        shift_labels.reshape(-1),
                        reduction='mean'
                    ).item()
                else:
                    # MDM: approximate using ELBO
                    logits = model(x_0)
                    nll = F.cross_entropy(
                        logits[:, :, :model.cfg.vocab_size].reshape(-1, model.cfg.vocab_size),
                        x_0.reshape(-1),
                        reduction='mean'
                    ).item()
                nlls.append(nll)
            batch_count += 1

        results[name] = nlls

    return results

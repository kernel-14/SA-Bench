"""
SCoRe training: Stage I and Stage II multi-turn RL.

Stage I (Section 5.1):
    Optimize second-attempt reward while constraining first-attempt distribution
    to the base model via a strong KL penalty (β2).

    Objective (Equation 3):
        max_θ E[r(y2, y*) - β2 * KL(π_θ(·|x1) || π_ref(·|x1))]

Stage II (Section 5.2):
    Jointly optimize both attempts with reward shaping.

    Objective (Equation 4 + shaped reward):
        max_θ E[Σ r̂(yi, y*) - β1 * KL(π_θ(·|xi) || π_ref(·|xi))]

    where r̂(y2, y*) = r(y2, y*) + α * (r(y2, y*) - r(y1, y*))

The base RL algorithm is REINFORCE with a KL-divergence penalty against a
fixed reference model (Ahmadian et al., 2024), extended to multiple turns
following the hierarchical framework of Zhou et al. (2024).
"""

import argparse
import os
import random
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import MathConfig, MBPPConfig, SCoReConfig
from data import (
    Problem,
    SCoReDataset,
    collect_base_model_trajectories,
    load_math_dataset,
    load_mbpp_dataset,
)
from model import LLMPolicy, ReferencePolicy, load_policy_and_ref
from rewards import compute_reward, shaped_reward_t2


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def collect_rollout(
    policy: LLMPolicy,
    problem: Problem,
    temperature: float = 1.0,
    reward_fn=None,
) -> Dict:
    """
    Collect a single two-turn self-correction rollout.

    Returns a dict with:
        prompt_t1, response_t1, reward_t1,
        prompt_t2, response_t2, reward_t2
    """
    response_t1 = policy.generate(problem.prompt_t1, temperature=temperature)
    prompt_t2 = problem.build_prompt_t2(response_t1)
    response_t2 = policy.generate(prompt_t2, temperature=temperature)

    metadata = dict(problem.metadata or {})
    metadata["answer"] = problem.answer

    reward_t1 = reward_fn(response_t1, metadata, problem.task) if reward_fn else 0.0
    reward_t2 = reward_fn(response_t2, metadata, problem.task) if reward_fn else 0.0

    return {
        "problem_id": problem.problem_id,
        "prompt_t1": problem.prompt_t1,
        "response_t1": response_t1,
        "reward_t1": reward_t1,
        "prompt_t2": prompt_t2,
        "response_t2": response_t2,
        "reward_t2": reward_t2,
        "answer": problem.answer,
        "task": problem.task,
        "metadata": metadata,
    }


def collect_rollouts_batch(
    policy: LLMPolicy,
    problems: List[Problem],
    temperature: float = 1.0,
    reward_fn=None,
) -> List[Dict]:
    """Collect rollouts for a batch of problems."""
    rollouts = []
    for problem in problems:
        rollout = collect_rollout(policy, problem, temperature=temperature, reward_fn=reward_fn)
        rollouts.append(rollout)
    return rollouts


# ---------------------------------------------------------------------------
# REINFORCE loss computation
# ---------------------------------------------------------------------------

def compute_reinforce_loss_stage1(
    policy: LLMPolicy,
    ref_policy: ReferencePolicy,
    rollouts: List[Dict],
    beta1: float,
    beta2: float,
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute Stage I REINFORCE loss (Equation 3).

    Objective:
        max_θ E[r(y2, y*) - β2 * KL(π_θ(·|x1) || π_ref(·|x1))]

    The standard KL penalty (β1) is also applied to the second turn but with
    a smaller weight (omitted from Eq. 3 for clarity, per the paper).

    REINFORCE gradient:
        ∇_θ J ≈ (R - baseline) * ∇_θ log π_θ(y2 | x2)
                - β2 * ∇_θ KL(π_θ(·|x1) || π_ref(·|x1))
                - β1 * ∇_θ KL(π_θ(·|x2) || π_ref(·|x2))

    where baseline = mean(R) over the batch.
    """
    rewards_t2 = torch.tensor(
        [r["reward_t2"] for r in rollouts], dtype=torch.float32
    )
    baseline = rewards_t2.mean()
    advantages = rewards_t2 - baseline

    losses = []
    total_kl_t1 = 0.0
    total_kl_t2 = 0.0

    for rollout, advantage in zip(rollouts, advantages):
        # Second-turn policy gradient
        log_prob_t2 = policy.log_prob(rollout["prompt_t2"], rollout["response_t2"])
        pg_loss_t2 = -advantage.item() * log_prob_t2

        # KL penalty on first turn (strong, β2) — prevents first-turn drift
        kl_t1 = policy.kl_divergence_from_ref(
            rollout["prompt_t1"], rollout["response_t1"], ref_policy._policy
        )

        # KL penalty on second turn (standard, β1)
        kl_t2 = policy.kl_divergence_from_ref(
            rollout["prompt_t2"], rollout["response_t2"], ref_policy._policy
        )

        loss = pg_loss_t2 + beta2 * kl_t1 + beta1 * kl_t2
        losses.append(loss)
        total_kl_t1 += kl_t1.item()
        total_kl_t2 += kl_t2.item()

    n = len(rollouts)
    total_loss = torch.stack(losses).mean()

    metrics = {
        "loss": total_loss.item(),
        "reward_t2_mean": rewards_t2.mean().item(),
        "reward_t2_std": rewards_t2.std().item(),
        "kl_t1_mean": total_kl_t1 / n,
        "kl_t2_mean": total_kl_t2 / n,
        "baseline": baseline.item(),
    }
    return total_loss, metrics


def compute_reinforce_loss_stage2(
    policy: LLMPolicy,
    ref_policy: ReferencePolicy,
    rollouts: List[Dict],
    beta1: float,
    alpha: float,
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute Stage II REINFORCE loss (Equation 4 + reward shaping).

    Shaped reward at second attempt:
        r̂(y2, y*) = r(y2, y*) + α * (r(y2, y*) - r(y1, y*))

    Total reward per rollout:
        R = r(y1, y*) + r̂(y2, y*)

    REINFORCE gradient (with per-turn baselines):
        ∇_θ J ≈ (R1 - b1) * ∇_θ log π_θ(y1|x1)
               + (R2_shaped - b2) * ∇_θ log π_θ(y2|x2)
               - β1 * [∇_θ KL(π_θ(·|x1)||π_ref) + ∇_θ KL(π_θ(·|x2)||π_ref)]
    """
    rewards_t1 = torch.tensor([r["reward_t1"] for r in rollouts], dtype=torch.float32)
    rewards_t2 = torch.tensor([r["reward_t2"] for r in rollouts], dtype=torch.float32)

    # Shaped second-attempt rewards
    shaped_rewards_t2 = rewards_t2 + alpha * (rewards_t2 - rewards_t1)

    # Per-turn baselines (mean over batch)
    baseline_t1 = rewards_t1.mean()
    baseline_t2 = shaped_rewards_t2.mean()

    advantages_t1 = rewards_t1 - baseline_t1
    advantages_t2 = shaped_rewards_t2 - baseline_t2

    losses = []
    total_kl_t1 = 0.0
    total_kl_t2 = 0.0

    for rollout, adv_t1, adv_t2 in zip(rollouts, advantages_t1, advantages_t2):
        # First-turn policy gradient
        log_prob_t1 = policy.log_prob(rollout["prompt_t1"], rollout["response_t1"])
        pg_loss_t1 = -adv_t1.item() * log_prob_t1

        # Second-turn policy gradient (with shaped reward)
        log_prob_t2 = policy.log_prob(rollout["prompt_t2"], rollout["response_t2"])
        pg_loss_t2 = -adv_t2.item() * log_prob_t2

        # KL penalties on both turns (β1)
        kl_t1 = policy.kl_divergence_from_ref(
            rollout["prompt_t1"], rollout["response_t1"], ref_policy._policy
        )
        kl_t2 = policy.kl_divergence_from_ref(
            rollout["prompt_t2"], rollout["response_t2"], ref_policy._policy
        )

        loss = pg_loss_t1 + pg_loss_t2 + beta1 * (kl_t1 + kl_t2)
        losses.append(loss)
        total_kl_t1 += kl_t1.item()
        total_kl_t2 += kl_t2.item()

    n = len(rollouts)
    total_loss = torch.stack(losses).mean()

    metrics = {
        "loss": total_loss.item(),
        "reward_t1_mean": rewards_t1.mean().item(),
        "reward_t2_mean": rewards_t2.mean().item(),
        "shaped_reward_t2_mean": shaped_rewards_t2.mean().item(),
        "delta_t1_t2": (rewards_t2 - rewards_t1).mean().item(),
        "kl_t1_mean": total_kl_t1 / n,
        "kl_t2_mean": total_kl_t2 / n,
        "baseline_t1": baseline_t1.item(),
        "baseline_t2": baseline_t2.item(),
    }
    return total_loss, metrics


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def train_stage1(
    policy: LLMPolicy,
    ref_policy: ReferencePolicy,
    train_problems: List[Problem],
    eval_problems: List[Problem],
    cfg,
    reward_fn,
    output_dir: str,
    use_wandb: bool = False,
) -> LLMPolicy:
    """
    Stage I training: decouple first and second attempts.

    Trains the model to produce high-reward second attempts while constraining
    the first-attempt distribution to stay close to the base model.
    """
    optimizer = Adam(
        [p for p in policy.model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    os.makedirs(output_dir, exist_ok=True)
    best_delta = float("-inf")
    global_step = 0

    rng = random.Random(cfg.seed)
    problem_pool = list(train_problems)

    pbar = tqdm(total=cfg.stage1_steps, desc="Stage I")

    while global_step < cfg.stage1_steps:
        # Sample a mini-batch of problems
        batch_size = min(cfg.batch_size, len(problem_pool))
        batch_problems = rng.sample(problem_pool, batch_size)

        # Collect on-policy rollouts
        policy.model.eval()
        with torch.no_grad():
            rollouts = collect_rollouts_batch(
                policy, batch_problems,
                temperature=cfg.sampling_temperature,
                reward_fn=reward_fn,
            )

        # Compute loss and update
        policy.model.train()
        optimizer.zero_grad()
        loss, metrics = compute_reinforce_loss_stage1(
            policy, ref_policy, rollouts,
            beta1=cfg.beta1,
            beta2=cfg.beta2,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.model.parameters(), cfg.max_grad_norm)
        optimizer.step()

        global_step += 1
        pbar.update(1)

        if global_step % cfg.logging_steps == 0:
            pbar.set_postfix({
                "loss": f"{metrics['loss']:.4f}",
                "r_t2": f"{metrics['reward_t2_mean']:.3f}",
                "kl_t1": f"{metrics['kl_t1_mean']:.3f}",
            })
            if use_wandb:
                import wandb
                wandb.log({"stage1/" + k: v for k, v in metrics.items()}, step=global_step)

        if global_step % cfg.eval_steps == 0:
            eval_metrics = evaluate_self_correction(
                policy, eval_problems, reward_fn, temperature=0.0
            )
            delta = eval_metrics["delta_t1_t2"]
            print(
                f"\nStep {global_step} | Acc@t1={eval_metrics['acc_t1']:.3f} "
                f"Acc@t2={eval_metrics['acc_t2']:.3f} Δ={delta:.3f}"
            )
            if use_wandb:
                import wandb
                wandb.log({"stage1/eval/" + k: v for k, v in eval_metrics.items()}, step=global_step)

            if delta > best_delta:
                best_delta = delta
                policy.model.save_pretrained(os.path.join(output_dir, "best"))
                policy.tokenizer.save_pretrained(os.path.join(output_dir, "best"))

        if global_step % cfg.save_steps == 0:
            ckpt_dir = os.path.join(output_dir, f"step_{global_step}")
            policy.model.save_pretrained(ckpt_dir)
            policy.tokenizer.save_pretrained(ckpt_dir)

    pbar.close()
    return policy


def train_stage2(
    policy: LLMPolicy,
    ref_policy: ReferencePolicy,
    train_problems: List[Problem],
    eval_problems: List[Problem],
    cfg,
    reward_fn,
    output_dir: str,
    offline_first_attempts: Optional[List[Dict]] = None,
    use_wandb: bool = False,
) -> LLMPolicy:
    """
    Stage II training: joint multi-turn RL with reward shaping.

    Jointly optimizes both attempts. Uses shaped reward to prevent behavior
    collapse to the "direct" solution (Section 5.2).

    Optionally incorporates offline first-attempt solutions from the base model
    to augment coverage (Section 5.3).
    """
    optimizer = Adam(
        [p for p in policy.model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    os.makedirs(output_dir, exist_ok=True)
    best_delta = float("-inf")
    global_step = 0

    rng = random.Random(cfg.seed + 1)
    problem_pool = list(train_problems)

    # Build offline first-attempt lookup: problem_id -> list of response_t1
    offline_lookup: Dict[str, List[str]] = {}
    if offline_first_attempts:
        for item in offline_first_attempts:
            pid = item["problem_id"]
            if pid not in offline_lookup:
                offline_lookup[pid] = []
            offline_lookup[pid].append(item["response_t1"])

    pbar = tqdm(total=cfg.stage2_steps, desc="Stage II")

    while global_step < cfg.stage2_steps:
        batch_size = min(cfg.batch_size, len(problem_pool))
        batch_problems = rng.sample(problem_pool, batch_size)

        # Collect on-policy rollouts
        policy.model.eval()
        with torch.no_grad():
            rollouts = collect_rollouts_batch(
                policy, batch_problems,
                temperature=cfg.sampling_temperature,
                reward_fn=reward_fn,
            )

        # Optionally augment with offline first-attempt prompts (Section 5.3):
        # For problems where we have offline base-model first attempts, also
        # collect second-attempt rollouts conditioned on those fixed first attempts.
        if offline_lookup:
            offline_rollouts = _collect_offline_augmented_rollouts(
                policy, batch_problems, offline_lookup, reward_fn,
                temperature=cfg.sampling_temperature,
            )
            rollouts = rollouts + offline_rollouts

        # Compute loss and update
        policy.model.train()
        optimizer.zero_grad()
        loss, metrics = compute_reinforce_loss_stage2(
            policy, ref_policy, rollouts,
            beta1=cfg.beta1,
            alpha=cfg.alpha,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.model.parameters(), cfg.max_grad_norm)
        optimizer.step()

        global_step += 1
        pbar.update(1)

        if global_step % cfg.logging_steps == 0:
            pbar.set_postfix({
                "loss": f"{metrics['loss']:.4f}",
                "r_t1": f"{metrics['reward_t1_mean']:.3f}",
                "r_t2": f"{metrics['reward_t2_mean']:.3f}",
                "Δ": f"{metrics['delta_t1_t2']:.3f}",
            })
            if use_wandb:
                import wandb
                wandb.log({"stage2/" + k: v for k, v in metrics.items()}, step=global_step)

        if global_step % cfg.eval_steps == 0:
            eval_metrics = evaluate_self_correction(
                policy, eval_problems, reward_fn, temperature=0.0
            )
            delta = eval_metrics["delta_t1_t2"]
            print(
                f"\nStep {global_step} | Acc@t1={eval_metrics['acc_t1']:.3f} "
                f"Acc@t2={eval_metrics['acc_t2']:.3f} Δ={delta:.3f} "
                f"i→c={eval_metrics['delta_i2c']:.3f} c→i={eval_metrics['delta_c2i']:.3f}"
            )
            if use_wandb:
                import wandb
                wandb.log({"stage2/eval/" + k: v for k, v in eval_metrics.items()}, step=global_step)

            if delta > best_delta:
                best_delta = delta
                policy.model.save_pretrained(os.path.join(output_dir, "best"))
                policy.tokenizer.save_pretrained(os.path.join(output_dir, "best"))

        if global_step % cfg.save_steps == 0:
            ckpt_dir = os.path.join(output_dir, f"step_{global_step}")
            policy.model.save_pretrained(ckpt_dir)
            policy.tokenizer.save_pretrained(ckpt_dir)

    pbar.close()
    return policy


def _collect_offline_augmented_rollouts(
    policy: LLMPolicy,
    problems: List[Problem],
    offline_lookup: Dict[str, List[str]],
    reward_fn,
    temperature: float = 1.0,
) -> List[Dict]:
    """
    For each problem that has offline base-model first attempts, collect
    second-attempt rollouts conditioned on those fixed first attempts.

    This augments coverage of the state space used for on-policy RL (Section 5.3).
    """
    rollouts = []
    for problem in problems:
        if problem.problem_id not in offline_lookup:
            continue
        for response_t1 in offline_lookup[problem.problem_id]:
            prompt_t2 = problem.build_prompt_t2(response_t1)
            response_t2 = policy.generate(prompt_t2, temperature=temperature)

            metadata = dict(problem.metadata or {})
            metadata["answer"] = problem.answer

            reward_t1 = reward_fn(response_t1, metadata, problem.task)
            reward_t2 = reward_fn(response_t2, metadata, problem.task)

            rollouts.append({
                "problem_id": problem.problem_id,
                "prompt_t1": problem.prompt_t1,
                "response_t1": response_t1,
                "reward_t1": reward_t1,
                "prompt_t2": prompt_t2,
                "response_t2": response_t2,
                "reward_t2": reward_t2,
                "answer": problem.answer,
                "task": problem.task,
                "metadata": metadata,
            })
    return rollouts


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_self_correction(
    policy: LLMPolicy,
    problems: List[Problem],
    reward_fn,
    temperature: float = 0.0,
) -> Dict:
    """
    Evaluate self-correction performance on a set of problems.

    Returns metrics: acc_t1, acc_t2, delta_t1_t2, delta_i2c, delta_c2i
    (Section 3, Metrics).
    """
    policy.model.eval()
    correct_t1 = []
    correct_t2 = []

    with torch.no_grad():
        for problem in tqdm(problems, desc="Evaluating", leave=False):
            rollout = collect_rollout(
                policy, problem, temperature=temperature, reward_fn=reward_fn
            )
            correct_t1.append(rollout["reward_t1"] > 0.5)
            correct_t2.append(rollout["reward_t2"] > 0.5)

    n = len(problems)
    acc_t1 = sum(correct_t1) / n
    acc_t2 = sum(correct_t2) / n

    # Δ^(i→c): incorrect at t1, correct at t2
    delta_i2c = sum(
        (not c1) and c2 for c1, c2 in zip(correct_t1, correct_t2)
    ) / n

    # Δ^(c→i): correct at t1, incorrect at t2
    delta_c2i = sum(
        c1 and (not c2) for c1, c2 in zip(correct_t1, correct_t2)
    ) / n

    return {
        "acc_t1": acc_t1,
        "acc_t2": acc_t2,
        "delta_t1_t2": acc_t2 - acc_t1,
        "delta_i2c": delta_i2c,
        "delta_c2i": delta_c2i,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_reward_fn(task: str):
    """Build a reward function for the given task."""
    from rewards import compute_reward

    def reward_fn(response: str, metadata: dict, task_name: str) -> float:
        return compute_reward(response, metadata, task_name)

    return reward_fn


def main():
    parser = argparse.ArgumentParser(description="Train SCoRe")
    parser.add_argument("--task", type=str, default="math", choices=["math", "mbpp"])
    parser.add_argument("--stage", type=int, default=2, choices=[1, 2])
    parser.add_argument("--stage1_checkpoint", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="score")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    score_cfg = SCoReConfig(task=args.task, stage=args.stage)
    cfg = score_cfg.get_task_config()

    if args.model:
        cfg.base_model = args.model
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.seed:
        cfg.seed = args.seed

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=f"score_{args.task}_stage{args.stage}",
            config=asdict(cfg),
        )

    # Load datasets
    print("Loading datasets...")
    if args.task == "math":
        train_problems, eval_problems = load_math_dataset(
            train_problems_from_test=cfg.train_problems_from_test,
            eval_problems=cfg.eval_problems,
            seed=cfg.seed,
        )
    else:
        train_problems, eval_problems = load_mbpp_dataset(seed=cfg.seed)

    reward_fn = build_reward_fn(args.task)

    # Determine which model checkpoint to load
    if args.stage == 2 and args.stage1_checkpoint:
        model_path = args.stage1_checkpoint
    else:
        model_path = cfg.base_model

    print(f"Loading model from {model_path}...")
    policy, ref_policy = load_policy_and_ref(model_path)

    # Collect offline base-model first attempts for Stage II augmentation
    offline_first_attempts = None
    if args.stage == 2 and cfg.use_offline_first_attempts:
        print("Collecting offline base-model first attempts for Stage II augmentation...")
        base_policy, _ = load_policy_and_ref(cfg.base_model)
        offline_trajectories = collect_base_model_trajectories(
            base_policy,
            train_problems,
            num_samples=cfg.offline_samples_per_problem,
            temperature=cfg.sampling_temperature,
            reward_fn=reward_fn,
        )
        offline_first_attempts = offline_trajectories
        del base_policy

    output_dir = os.path.join(cfg.output_dir, f"stage{args.stage}")

    if args.stage == 1:
        print("Starting Stage I training...")
        policy = train_stage1(
            policy=policy,
            ref_policy=ref_policy,
            train_problems=train_problems,
            eval_problems=eval_problems,
            cfg=cfg,
            reward_fn=reward_fn,
            output_dir=output_dir,
            use_wandb=args.use_wandb,
        )
    else:
        print("Starting Stage II training...")
        policy = train_stage2(
            policy=policy,
            ref_policy=ref_policy,
            train_problems=train_problems,
            eval_problems=eval_problems,
            cfg=cfg,
            reward_fn=reward_fn,
            output_dir=output_dir,
            offline_first_attempts=offline_first_attempts,
            use_wandb=args.use_wandb,
        )

    # Final evaluation
    print("\nFinal evaluation...")
    final_metrics = evaluate_self_correction(
        policy, eval_problems, reward_fn, temperature=0.0
    )
    print(
        f"Acc@t1={final_metrics['acc_t1']:.4f} "
        f"Acc@t2={final_metrics['acc_t2']:.4f} "
        f"Δ(t1,t2)={final_metrics['delta_t1_t2']:.4f} "
        f"Δ^(i→c)={final_metrics['delta_i2c']:.4f} "
        f"Δ^(c→i)={final_metrics['delta_c2i']:.4f}"
    )

    if args.use_wandb:
        import wandb
        wandb.log({f"final/{k}": v for k, v in final_metrics.items()})
        wandb.finish()


if __name__ == "__main__":
    main()

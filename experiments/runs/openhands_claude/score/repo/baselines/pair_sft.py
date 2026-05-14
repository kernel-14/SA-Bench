"""
Pair-SFT baseline for self-correction.

Based on Welleck et al. (2023), adapted to train a single model (not a
separate corrector model) following the paper's description.

Algorithm:
1. Collect first-attempt responses from the base model
2. For each incorrect first attempt, pair it with a correct response
   (either from the base model or from the ground truth)
3. Construct synthetic repair traces: (problem, incorrect_t1, correct_t2)
4. Run SFT on the resulting dataset D_SFT

Variants:
- D_SFT: only (incorrect t1, correct t2) pairs
- D_SFT+: also includes (correct t1, correct t2) pairs

The paper runs one iteration of Pair-SFT following Welleck et al. (2023).
"""

import argparse
import os
import random
from typing import Dict, List, Optional

import torch
from torch.optim import Adam
from tqdm import tqdm

from config import PairSFTConfig
from data import (
    Problem,
    collect_base_model_trajectories,
    load_math_dataset,
    load_mbpp_dataset,
)
from evaluate import evaluate_on_problems
from model import LLMPolicy, load_policy_and_ref
from rewards import compute_reward


class PairSFTTrainer:
    """
    Trains a model for self-correction using the Pair-SFT approach.

    Constructs synthetic repair traces by pairing incorrect first-attempt
    responses with correct responses, then runs SFT.
    """

    def __init__(self, cfg: PairSFTConfig):
        self.cfg = cfg

    def _build_paired_dataset(
        self,
        trajectories: List[Dict],
        include_correct_pairs: bool = False,
    ) -> List[Dict]:
        """
        Build D_SFT or D_SFT+ from collected trajectories.

        For each problem, we pair:
        - An incorrect first attempt with a correct second attempt (if available)
        - Optionally: a correct first attempt with a correct second attempt

        This creates "synthetic" repair traces where the model must learn to
        correct the specific type of mistake it made in the first attempt.
        """
        # Group trajectories by problem_id
        by_problem: Dict[str, List[Dict]] = {}
        for traj in trajectories:
            pid = traj["problem_id"]
            if pid not in by_problem:
                by_problem[pid] = []
            by_problem[pid].append(traj)

        paired = []
        for pid, trajs in by_problem.items():
            incorrect_t1 = [t for t in trajs if t["reward_t1"] <= 0.5]
            correct_t2 = [t for t in trajs if t["reward_t2"] > 0.5]
            correct_t1 = [t for t in trajs if t["reward_t1"] > 0.5]

            # Pair incorrect first attempts with correct second attempts
            for inc in incorrect_t1:
                if correct_t2:
                    # Use the correct second attempt from any trajectory for this problem
                    cor = random.choice(correct_t2)
                    paired.append({
                        "problem_id": pid,
                        "prompt_t1": inc["prompt_t1"],
                        "response_t1": inc["response_t1"],
                        "prompt_t2": inc["prompt_t2"],  # prompt conditioned on incorrect t1
                        "response_t2": cor["response_t2"],  # correct response
                        "reward_t1": inc["reward_t1"],
                        "reward_t2": cor["reward_t2"],
                        "answer": inc["answer"],
                        "task": inc["task"],
                        "metadata": inc["metadata"],
                    })

            # Optionally include correct→correct pairs (D_SFT+)
            if include_correct_pairs:
                for cor in correct_t1:
                    if cor["reward_t2"] > 0.5:
                        paired.append(cor)

        return paired

    def _compute_sft_loss(
        self,
        policy: LLMPolicy,
        trajectories: List[Dict],
    ) -> torch.Tensor:
        """
        Compute SFT loss on paired trajectories.

        Maximizes log P(response_t2 | prompt_t2) where prompt_t2 is
        conditioned on the incorrect first attempt.
        """
        total_loss = torch.tensor(0.0, requires_grad=True)
        for traj in trajectories:
            log_prob_t2 = policy.log_prob(traj["prompt_t2"], traj["response_t2"])
            total_loss = total_loss + (-log_prob_t2)
        return total_loss / len(trajectories)

    def train(
        self,
        policy: LLMPolicy,
        train_problems: List[Problem],
        eval_problems: List[Problem],
        reward_fn,
        output_dir: str,
    ) -> LLMPolicy:
        """
        Run Pair-SFT training (one iteration).
        """
        cfg = self.cfg
        os.makedirs(output_dir, exist_ok=True)
        rng = random.Random(cfg.seed)

        # Step 1: Collect base model trajectories
        print("Collecting base model trajectories...")
        trajectories = collect_base_model_trajectories(
            policy,
            train_problems,
            num_samples=4,  # multiple samples to increase chance of getting correct t2
            temperature=1.0,
            reward_fn=reward_fn,
        )

        # Step 2: Build paired dataset
        paired = self._build_paired_dataset(
            trajectories,
            include_correct_pairs=cfg.include_correct_pairs,
        )
        print(
            f"Built {len(paired)} paired examples "
            f"({'D_SFT+' if cfg.include_correct_pairs else 'D_SFT'})"
        )

        if not paired:
            print("No valid pairs found.")
            return policy

        # Step 3: SFT
        optimizer = Adam(
            [p for p in policy.model.parameters() if p.requires_grad],
            lr=cfg.learning_rate,
        )

        policy.model.train()
        step = 0
        pbar = tqdm(total=cfg.training_steps, desc="Pair-SFT")

        while step < cfg.training_steps:
            batch = rng.sample(paired, min(cfg.batch_size, len(paired)))
            optimizer.zero_grad()
            loss = self._compute_sft_loss(policy, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 1.0)
            optimizer.step()
            step += 1
            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if step % 200 == 0:
                eval_metrics = evaluate_on_problems(
                    policy, eval_problems, reward_fn, temperature=0.0
                )
                print(
                    f"\nStep {step} | Acc@t1={eval_metrics['acc_t1']:.3f} "
                    f"Acc@t2={eval_metrics['acc_t2']:.3f} "
                    f"Δ={eval_metrics['delta_t1_t2']:.3f}"
                )

        pbar.close()

        policy.model.save_pretrained(output_dir)
        policy.tokenizer.save_pretrained(output_dir)
        return policy


def main():
    parser = argparse.ArgumentParser(description="Train Pair-SFT baseline")
    parser.add_argument("--task", type=str, default="math", choices=["math", "mbpp"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="checkpoints/pair_sft")
    parser.add_argument("--include_correct_pairs", action="store_true",
                        help="Use D_SFT+ (include correct→correct pairs)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = PairSFTConfig(
        task=args.task,
        include_correct_pairs=args.include_correct_pairs,
        seed=args.seed,
    )
    if args.model:
        cfg.base_model = args.model

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    print("Loading datasets...")
    if args.task == "math":
        train_problems, eval_problems = load_math_dataset(seed=cfg.seed)
    else:
        train_problems, eval_problems = load_mbpp_dataset(seed=cfg.seed)

    def reward_fn(response, metadata, task):
        return compute_reward(response, metadata, task)

    print(f"Loading model from {cfg.base_model}...")
    policy, _ = load_policy_and_ref(cfg.base_model)

    trainer = PairSFTTrainer(cfg)
    policy = trainer.train(
        policy, train_problems, eval_problems, reward_fn, args.output_dir
    )

    print("\nFinal evaluation...")
    final_metrics = evaluate_on_problems(
        policy, eval_problems, reward_fn, temperature=0.0
    )
    print(
        f"Acc@t1={final_metrics['acc_t1']:.4f} "
        f"Acc@t2={final_metrics['acc_t2']:.4f} "
        f"Δ(t1,t2)={final_metrics['delta_t1_t2']:.4f} "
        f"Δ^(i→c)={final_metrics['delta_i2c']:.4f} "
        f"Δ^(c→i)={final_metrics['delta_c2i']:.4f}"
    )


if __name__ == "__main__":
    main()

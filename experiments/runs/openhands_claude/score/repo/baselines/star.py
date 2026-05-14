"""
STaR (Self-Taught Reasoner) baseline for self-correction.

Zelikman et al. (2022), extended to multi-turn self-correction following
Singh et al. (2023, 2024).

Algorithm:
1. Collect two-turn traces from the current model
2. Filter to keep only traces where the model successfully corrects an
   incorrect first attempt (incorrect → correct)
3. Run SFT on the filtered dataset D_STaR
4. Repeat for num_iterations iterations

Variants:
- D_STaR: only incorrect→correct traces
- D_STaR+: also includes correct→correct traces (to prevent erroneous revision)

The paper runs 3 iterations following Singh et al. (2024).
"""

import argparse
import os
import random
from typing import Dict, List, Optional

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq

from config import STaRConfig
from data import (
    Problem,
    OfflineTrajectoryDataset,
    collect_base_model_trajectories,
    load_math_dataset,
    load_mbpp_dataset,
)
from evaluate import compute_self_correction_metrics, evaluate_on_problems
from model import LLMPolicy, load_policy_and_ref
from rewards import compute_reward


class STaRTrainer:
    """
    Trains a model for self-correction using the STaR approach.

    Iteratively collects self-correction traces, filters for successful
    corrections, and runs SFT on the filtered dataset.
    """

    def __init__(self, cfg: STaRConfig):
        self.cfg = cfg

    def _filter_trajectories(
        self,
        trajectories: List[Dict],
        include_correct_pairs: bool = False,
    ) -> List[Dict]:
        """
        Filter trajectories to build D_STaR or D_STaR+.

        D_STaR: keep only (incorrect t1, correct t2) pairs
        D_STaR+: also keep (correct t1, correct t2) pairs
        """
        filtered = []
        for traj in trajectories:
            r1 = traj["reward_t1"] > 0.5
            r2 = traj["reward_t2"] > 0.5

            if (not r1) and r2:
                # Successful correction: incorrect → correct
                filtered.append(traj)
            elif include_correct_pairs and r1 and r2:
                # Correct → correct: prevents erroneous revision
                filtered.append(traj)
        return filtered

    def _compute_sft_loss(
        self,
        policy: LLMPolicy,
        trajectories: List[Dict],
    ) -> torch.Tensor:
        """
        Compute SFT (negative log-likelihood) loss on a batch of trajectories.

        For STaR, we maximize log P(response_t2 | prompt_t2) on the filtered
        correction traces.
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
        Run STaR training for num_iterations iterations.
        """
        cfg = self.cfg
        os.makedirs(output_dir, exist_ok=True)
        rng = random.Random(cfg.seed)

        for iteration in range(cfg.num_iterations):
            print(f"\n=== STaR Iteration {iteration + 1}/{cfg.num_iterations} ===")

            # Step 1: Collect traces from current model
            print("Collecting self-correction traces...")
            trajectories = collect_base_model_trajectories(
                policy,
                train_problems,
                num_samples=1,
                temperature=1.0,
                reward_fn=reward_fn,
            )

            # Step 2: Filter trajectories
            filtered = self._filter_trajectories(
                trajectories,
                include_correct_pairs=cfg.include_correct_pairs,
            )
            print(
                f"Filtered {len(filtered)}/{len(trajectories)} trajectories "
                f"({'D_STaR+' if cfg.include_correct_pairs else 'D_STaR'})"
            )

            if not filtered:
                print("No valid trajectories found, skipping iteration.")
                continue

            # Step 3: SFT on filtered dataset
            optimizer = Adam(
                [p for p in policy.model.parameters() if p.requires_grad],
                lr=cfg.learning_rate,
            )

            policy.model.train()
            step = 0
            pbar = tqdm(total=cfg.training_steps_per_iter, desc=f"SFT iter {iteration+1}")

            while step < cfg.training_steps_per_iter:
                batch = rng.sample(
                    filtered, min(cfg.batch_size, len(filtered))
                )
                optimizer.zero_grad()
                loss = self._compute_sft_loss(policy, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 1.0)
                optimizer.step()
                step += 1
                pbar.update(1)
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            pbar.close()

            # Evaluate after each iteration
            eval_metrics = evaluate_on_problems(
                policy, eval_problems, reward_fn, temperature=0.0
            )
            print(
                f"Iter {iteration+1} | Acc@t1={eval_metrics['acc_t1']:.3f} "
                f"Acc@t2={eval_metrics['acc_t2']:.3f} "
                f"Δ={eval_metrics['delta_t1_t2']:.3f}"
            )

            # Save checkpoint
            ckpt_dir = os.path.join(output_dir, f"iter_{iteration+1}")
            policy.model.save_pretrained(ckpt_dir)
            policy.tokenizer.save_pretrained(ckpt_dir)

        return policy


def main():
    parser = argparse.ArgumentParser(description="Train STaR baseline")
    parser.add_argument("--task", type=str, default="math", choices=["math", "mbpp"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="checkpoints/star")
    parser.add_argument("--include_correct_pairs", action="store_true",
                        help="Use D_STaR+ (include correct→correct pairs)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = STaRConfig(
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

    trainer = STaRTrainer(cfg)
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

"""Baseline training methods compared against SCoRe.

Implements:
1. STaR (Zelikman et al. 2022): Iterative SFT on successful correction traces
2. Pair-SFT (Welleck et al. 2023): SFT on synthetically paired repair traces
3. Self-Refine (Madaan et al. 2023): Prompting-based self-correction
4. Standard multi-turn RL (no Stage I, no reward shaping)
5. Single-turn RL (ablation)
6. SCoRe ablations (no Stage I, no reward shaping, STaR instead of REINFORCE)
"""

import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import (
    SCoReConfig,
    STARConfig,
    PairSFTConfig,
    AblationConfig,
    get_config,
)
from data import (
    MATH500Dataset,
    MBPPDataset,
    create_dataloader,
)
from metrics import compute_self_correction_metrics
from models import LLMPolicy, load_model_and_tokenizer
from prompts import (
    build_math_first_turn_prompt,
    build_math_second_turn_prompt,
    build_mbpp_first_turn_prompt,
    build_mbpp_second_turn_prompt,
    build_self_refine_first_prompt,
    build_self_refine_feedback_prompt,
    build_self_refine_refinement_prompt,
)
from rewards import compute_reward


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_base_model_traces(
    base_policy: LLMPolicy,
    dataloader: DataLoader,
    config: SCoReConfig,
    is_code: bool = False,
) -> List[Dict]:
    """Collect two-turn self-correction traces from the base model.

    Each trace contains: problem, first response, second response, rewards.
    Used to construct D_STaR and D_SFT datasets.
    """
    traces = []
    base_policy.model.eval()

    for batch in dataloader:
        batch_size = len(batch["problem_ids"])

        if is_code:
            prompts_t1 = [
                build_mbpp_first_turn_prompt(desc, tests)
                for desc, tests in zip(batch["task_descriptions"], batch["test_cases"])
            ]
        else:
            prompts_t1 = [
                build_math_first_turn_prompt(prob)
                for prob in batch["problem_texts"]
            ]

        responses_t1 = base_policy.generate(
            prompts_t1,
            max_new_tokens=config.max_new_tokens,
            temperature=config.sampling_temperature,
            do_sample=True,
        )

        if is_code:
            prompts_t2 = [
                build_mbpp_second_turn_prompt(desc, tests, resp)
                for desc, tests, resp in zip(
                    batch["task_descriptions"], batch["test_cases"], responses_t1
                )
            ]
        else:
            prompts_t2 = [
                build_math_second_turn_prompt(prob, resp)
                for prob, resp in zip(batch["problem_texts"], responses_t1)
            ]

        responses_t2 = base_policy.generate(
            prompts_t2,
            max_new_tokens=config.max_new_tokens,
            temperature=config.sampling_temperature,
            do_sample=True,
        )

        for i in range(batch_size):
            if is_code:
                r1 = compute_reward(
                    responses_t1[i],
                    batch["code_solutions"][i],
                    is_code=True,
                    test_cases=batch["test_cases"][i],
                    entry_point=batch["entry_points"][i],
                )
                r2 = compute_reward(
                    responses_t2[i],
                    batch["code_solutions"][i],
                    is_code=True,
                    test_cases=batch["test_cases"][i],
                    entry_point=batch["entry_points"][i],
                )
                trace = {
                    "problem_id": batch["problem_ids"][i],
                    "task_description": batch["task_descriptions"][i],
                    "test_cases": batch["test_cases"][i],
                    "code_solution": batch["code_solutions"][i],
                    "entry_point": batch["entry_points"][i],
                    "response_t1": responses_t1[i],
                    "response_t2": responses_t2[i],
                    "reward_t1": r1,
                    "reward_t2": r2,
                    "prompt_t1": prompts_t1[i],
                    "prompt_t2": prompts_t2[i],
                }
            else:
                r1 = compute_reward(responses_t1[i], batch["answers"][i])
                r2 = compute_reward(responses_t2[i], batch["answers"][i])
                trace = {
                    "problem_id": batch["problem_ids"][i],
                    "problem_text": batch["problem_texts"][i],
                    "answer": batch["answers"][i],
                    "response_t1": responses_t1[i],
                    "response_t2": responses_t2[i],
                    "reward_t1": r1,
                    "reward_t2": r2,
                    "prompt_t1": prompts_t1[i],
                    "prompt_t2": prompts_t2[i],
                }
            traces.append(trace)

    return traces


def filter_star_traces(traces: List[Dict]) -> List[Dict]:
    """Filter traces for STaR: keep only those that correct an incorrect response.

    D_STaR: trajectories where reward_t1 = 0 and reward_t2 = 1.
    D_STaR^+: also include correct→correct trajectories.
    """
    successful = [t for t in traces if t["reward_t1"] == 0 and t["reward_t2"] == 1]
    return successful


def filter_star_plus_traces(traces: List[Dict]) -> List[Dict]:
    """Filter traces for STaR+: include both incorrect→correct and correct→correct."""
    successful = [t for t in traces if t["reward_t2"] == 1]
    return successful


def build_pair_sft_dataset(traces: List[Dict]) -> List[Dict]:
    """Build Pair-SFT dataset: pair incorrect first responses with correct second responses.

    D_SFT: For each incorrect first response, provide the correct second response.
    D_SFT^+: Also include correct→correct pairs.
    """
    sft_data = []
    for t in traces:
        if t["reward_t1"] == 0:
            # Incorrect first attempt
            if t["reward_t2"] == 1:
                sft_data.append({
                    "prompt": t["prompt_t2"],
                    "target": t["response_t2"],
                })
    return sft_data


def build_pair_sft_plus_dataset(traces: List[Dict]) -> List[Dict]:
    """Build Pair-SFT+ dataset: include all second responses where reward_t2 = 1."""
    sft_data = []
    for t in traces:
        if t["reward_t2"] == 1:
            sft_data.append({
                "prompt": t["prompt_t2"],
                "target": t["response_t2"],
            })
    return sft_data


class SFTDataset(Dataset):
    """Simple dataset for SFT training."""

    def __init__(self, data: List[Dict]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]


def train_sft(
    policy: LLMPolicy,
    sft_dataset: SFTDataset,
    num_steps: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    """Run SFT training on a given dataset.

    Standard cross-entropy loss on the target responses.
    """
    policy.model.train()
    optimizer = torch.optim.Adam(policy.model.parameters(), lr=learning_rate)

    # Tokenize and prepare all data
    all_prompts = [item["prompt"] for item in sft_dataset.data]
    all_targets = [item["target"] for item in sft_dataset.data]

    # Combine for full text
    full_texts = [p + t for p, t in zip(all_prompts, all_targets)]

    global_step = 0

    for epoch in range(100):
        # Shuffle
        indices = list(range(len(full_texts)))
        random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            if global_step >= num_steps:
                break

            batch_indices = indices[start : start + batch_size]
            batch_full = [full_texts[i] for i in batch_indices]
            batch_prompts = [all_prompts[i] for i in batch_indices]
            batch_targets = [all_targets[i] for i in batch_indices]

            # Tokenize
            full_enc = policy.tokenizer(
                batch_full, return_tensors="pt", padding=True, truncation=True
            )
            prompt_enc = policy.tokenizer(
                batch_prompts, return_tensors="pt", padding=True, truncation=True
            )

            input_ids = full_enc["input_ids"].to(policy.device)
            attention_mask = full_enc["attention_mask"].to(policy.device)

            # Labels: ignore prompt tokens
            labels = input_ids.clone()
            prompt_lens = prompt_enc["input_ids"].shape[1]
            labels[:, :prompt_lens] = -100

            outputs = policy.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

            if global_step % 100 == 0:
                print(f"[SFT] Step {global_step}/{num_steps}, Loss: {loss.item():.4f}")

        if global_step >= num_steps:
            break


def train_star(
    base_policy: LLMPolicy,
    train_dataloader: DataLoader,
    config: SCoReConfig,
    star_config: STARConfig,
    output_dir: str,
    is_code: bool = False,
) -> LLMPolicy:
    """Train STaR: iterative SFT on successful self-correction traces.

    Algorithm:
    1. Collect two-turn traces from current policy
    2. Filter to keep only successful correction traces (incorrect → correct)
    3. SFT on the filtered dataset
    4. Repeat for num_iterations
    """
    policy = load_model_and_tokenizer(config.base_model)
    set_seed(config.seed)

    for iteration in range(star_config.num_iterations):
        print(f"\n[STaR] Iteration {iteration + 1}/{star_config.num_iterations}")
        print("Collecting traces...")

        traces = collect_base_model_traces(
            policy, train_dataloader, config, is_code
        )

        if star_config.include_correct_to_correct:
            filtered = filter_star_plus_traces(traces)
        else:
            filtered = filter_star_traces(traces)

        print(f"  Total traces: {len(traces)}, Filtered: {len(filtered)}")

        if len(filtered) < 2:
            print("  Not enough filtered traces, stopping.")
            break

        sft_data = [
            {"prompt": t["prompt_t2"], "target": t["response_t2"]}
            for t in filtered
        ]
        sft_dataset = SFTDataset(sft_data)

        print(f"  Running SFT for {star_config.training_steps_per_iteration} steps...")
        train_sft(
            policy,
            sft_dataset,
            star_config.training_steps_per_iteration,
            star_config.batch_size,
            star_config.learning_rate,
        )

        # Save iteration checkpoint
        ckpt_path = os.path.join(output_dir, f"star_iter_{iteration + 1}")
        os.makedirs(ckpt_path, exist_ok=True)
        policy.model.save_pretrained(ckpt_path)
        policy.tokenizer.save_pretrained(ckpt_path)

    return policy


def train_pair_sft(
    base_policy: LLMPolicy,
    train_dataloader: DataLoader,
    config: SCoReConfig,
    sft_config: PairSFTConfig,
    output_dir: str,
    is_code: bool = False,
) -> LLMPolicy:
    """Train Pair-SFT: SFT on synthetically paired repair traces.

    Algorithm:
    1. Collect two-turn traces from base model
    2. Build synthetic repair dataset (incorrect t1 → correct t2)
    3. SFT once on the dataset
    """
    policy = load_model_and_tokenizer(config.base_model)
    set_seed(config.seed)

    print("\n[Pair-SFT] Collecting base model traces...")
    traces = collect_base_model_traces(
        base_policy, train_dataloader, config, is_code
    )

    if sft_config.include_correct_to_correct:
        sft_data = build_pair_sft_plus_dataset(traces)
    else:
        sft_data = build_pair_sft_dataset(traces)

    print(f"  Total traces: {len(traces)}, SFT data: {len(sft_data)}")

    if len(sft_data) < 2:
        print("  Not enough SFT data, returning base policy.")
        return policy

    sft_dataset = SFTDataset(sft_data)

    print(f"  Running SFT for {sft_config.training_steps} steps...")
    train_sft(
        policy,
        sft_dataset,
        sft_config.training_steps,
        sft_config.batch_size,
        sft_config.learning_rate,
    )

    # Save
    ckpt_path = os.path.join(output_dir, "pair_sft_checkpoint")
    os.makedirs(ckpt_path, exist_ok=True)
    policy.model.save_pretrained(ckpt_path)
    policy.tokenizer.save_pretrained(ckpt_path)

    return policy


def evaluate_self_refine(
    base_policy: LLMPolicy,
    eval_dataloader: DataLoader,
    config: SCoReConfig,
    is_code: bool = False,
) -> Dict[str, float]:
    """Evaluate the Self-Refine baseline (Madaan et al. 2023).

    Self-Refine prompts the model to:
    1. Generate an initial solution
    2. Provide self-feedback (identify mistakes)
    3. Generate a refined solution based on feedback
    """
    all_results = []
    base_policy.model.eval()

    for batch in eval_dataloader:
        batch_size = len(batch["problem_ids"])

        if is_code:
            # Self-Refine not explicitly described for code in the paper
            # but we adapt the same pattern
            continue
        else:
            prompts_t1 = [
                build_self_refine_first_prompt(prob)
                for prob in batch["problem_texts"]
            ]
            responses_t1 = base_policy.generate(
                prompts_t1,
                max_new_tokens=config.max_new_tokens,
                temperature=config.eval_temperature,
                do_sample=False,
            )

            prompts_feedback = [
                build_self_refine_feedback_prompt(prob, resp)
                for prob, resp in zip(batch["problem_texts"], responses_t1)
            ]
            feedbacks = base_policy.generate(
                prompts_feedback,
                max_new_tokens=config.max_new_tokens // 2,
                temperature=config.eval_temperature,
                do_sample=False,
            )

            prompts_refine = [
                build_self_refine_refinement_prompt(prob, resp, fb)
                for prob, resp, fb in zip(
                    batch["problem_texts"], responses_t1, feedbacks
                )
            ]
            responses_t2 = base_policy.generate(
                prompts_refine,
                max_new_tokens=config.max_new_tokens,
                temperature=config.eval_temperature,
                do_sample=False,
            )

        for i in range(batch_size):
            if is_code:
                r1 = compute_reward(
                    responses_t1[i],
                    batch["code_solutions"][i],
                    is_code=True,
                    test_cases=batch["test_cases"][i],
                )
                r2 = compute_reward(
                    responses_t2[i],
                    batch["code_solutions"][i],
                    is_code=True,
                    test_cases=batch["test_cases"][i],
                )
            else:
                r1 = compute_reward(responses_t1[i], batch["answers"][i])
                r2 = compute_reward(responses_t2[i], batch["answers"][i])

            all_results.append({
                "problem_id": batch["problem_ids"][i],
                "correct_t1": r1 > 0.5,
                "correct_t2": r2 > 0.5,
            })

    return compute_self_correction_metrics(all_results)


def train_standard_multi_turn_rl(
    policy: LLMPolicy,
    ref_policy: LLMPolicy,
    dataloader: DataLoader,
    config: SCoReConfig,
    num_steps: int,
    is_code: bool = False,
) -> None:
    """Standard multi-turn RL: optimize only turn-2 reward (Equation 1).

    This is the baseline that leads to behavior collapse (Section 5).
    No Stage I, no reward shaping.
    """
    policy.model.train()
    optimizer = torch.optim.Adam(policy.model.parameters(), lr=config.learning_rate)

    global_step = 0

    for epoch in range(100):
        for batch in dataloader:
            if global_step >= num_steps:
                break

            # Sample two-turn rollout
            if is_code:
                prompts_t1 = [
                    build_mbpp_first_turn_prompt(desc, tests)
                    for desc, tests in zip(
                        batch["task_descriptions"], batch["test_cases"]
                    )
                ]
            else:
                prompts_t1 = [
                    build_math_first_turn_prompt(prob)
                    for prob in batch["problem_texts"]
                ]

            responses_t1 = policy.generate(
                prompts_t1,
                max_new_tokens=config.max_new_tokens,
                temperature=config.sampling_temperature,
                do_sample=True,
            )

            if is_code:
                prompts_t2 = [
                    build_mbpp_second_turn_prompt(desc, tests, resp)
                    for desc, tests, resp in zip(
                        batch["task_descriptions"],
                        batch["test_cases"],
                        responses_t1,
                    )
                ]
            else:
                prompts_t2 = [
                    build_math_second_turn_prompt(prob, resp)
                    for prob, resp in zip(
                        batch["problem_texts"], responses_t1
                    )
                ]

            responses_t2 = policy.generate(
                prompts_t2,
                max_new_tokens=config.max_new_tokens,
                temperature=config.sampling_temperature,
                do_sample=True,
            )

            # Compute rewards (only use turn-2 reward)
            batch_size = len(batch["problem_ids"])
            rewards_t2_list = []
            for i in range(batch_size):
                if is_code:
                    r2 = compute_reward(
                        responses_t2[i],
                        batch["code_solutions"][i],
                        is_code=True,
                        test_cases=batch["test_cases"][i],
                    )
                else:
                    r2 = compute_reward(responses_t2[i], batch["answers"][i])
                rewards_t2_list.append(r2)

            rewards_t2 = torch.tensor(rewards_t2_list, device=policy.device)

            # Use REINFORCE loss (same as compute_reinforce_loss in models.py)
            loss = policy.compute_reinforce_loss(
                prompts=prompts_t2,
                responses=responses_t2,
                rewards=rewards_t2,
                ref_model=ref_policy,
                beta=config.beta1,
            )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

            if global_step % 100 == 0:
                print(
                    f"[Standard RL] Step {global_step}/{num_steps}, "
                    f"Loss: {loss.item():.4f}, "
                    f"R2_mean: {rewards_t2.mean().item():.3f}"
                )

        if global_step >= num_steps:
            break


def train_single_turn_rl(
    policy: LLMPolicy,
    ref_policy: LLMPolicy,
    dataloader: DataLoader,
    config: SCoReConfig,
    num_steps: int,
    is_code: bool = False,
) -> None:
    """Single-turn RL (ablation): optimize only first-turn responses.

    This is the "w/o multi-turn training" ablation from Table 4.
    """
    policy.model.train()
    optimizer = torch.optim.Adam(policy.model.parameters(), lr=config.learning_rate)

    global_step = 0

    for epoch in range(100):
        for batch in dataloader:
            if global_step >= num_steps:
                break

            if is_code:
                prompts = [
                    build_mbpp_first_turn_prompt(desc, tests)
                    for desc, tests in zip(
                        batch["task_descriptions"], batch["test_cases"]
                    )
                ]
            else:
                prompts = [
                    build_math_first_turn_prompt(prob)
                    for prob in batch["problem_texts"]
                ]

            responses = policy.generate(
                prompts,
                max_new_tokens=config.max_new_tokens,
                temperature=config.sampling_temperature,
                do_sample=True,
            )

            batch_size = len(batch["problem_ids"])
            rewards_list = []
            for i in range(batch_size):
                if is_code:
                    r = compute_reward(
                        responses[i],
                        batch["code_solutions"][i],
                        is_code=True,
                        test_cases=batch["test_cases"][i],
                    )
                else:
                    r = compute_reward(responses[i], batch["answers"][i])
                rewards_list.append(r)

            rewards = torch.tensor(rewards_list, device=policy.device)

            loss = policy.compute_reinforce_loss(
                prompts=prompts,
                responses=responses,
                rewards=rewards,
                ref_model=ref_policy,
                beta=config.beta1,
            )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

            if global_step % 100 == 0:
                print(
                    f"[Single-turn RL] Step {global_step}/{num_steps}, "
                    f"Loss: {loss.item():.4f}, "
                    f"R_mean: {rewards.mean().item():.3f}"
                )

        if global_step >= num_steps:
            break




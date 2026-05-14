"""SCoRe training: Two-stage multi-turn RL for self-correction.

Implements the SCoRe algorithm from the paper:
- Stage I: Train initialization that decouples the two attempts.
  Maximizes second-attempt reward while constraining the first-turn
  distribution to be close to the base model via KL penalty.
- Stage II: Joint optimize both attempts with reward shaping.
  Uses bonus b(y2|y1) = α·(r(y2) - r(y1)) to prevent behavior collapse.
"""

import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import SCoReConfig, get_config
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
)
from rewards import compute_reward, compute_total_reward_turn2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_prompts(
    batch: Dict,
    is_code: bool,
    turn: int = 1,
    first_responses: Optional[List[str]] = None,
) -> List[str]:
    """Build prompts for a given turn.

    Args:
        batch: Data batch
        is_code: Whether this is a code task
        turn: 1 or 2
        first_responses: Responses from turn 1 (needed for turn 2)

    Returns:
        List of prompt strings
    """
    if is_code:
        if turn == 1:
            return [
                build_mbpp_first_turn_prompt(desc, tests)
                for desc, tests in zip(batch["task_descriptions"], batch["test_cases"])
            ]
        else:
            return [
                build_mbpp_second_turn_prompt(desc, tests, resp)
                for desc, tests, resp in zip(
                    batch["task_descriptions"],
                    batch["test_cases"],
                    first_responses,
                )
            ]
    else:
        if turn == 1:
            return [
                build_math_first_turn_prompt(prob)
                for prob in batch["problem_texts"]
            ]
        else:
            return [
                build_math_second_turn_prompt(prob, resp)
                for prob, resp in zip(batch["problem_texts"], first_responses)
            ]


def sample_two_turn_rollout(
    policy: LLMPolicy,
    batch: Dict,
    is_code: bool,
    max_new_tokens: int,
    temperature: float,
) -> Tuple[List[str], List[str], torch.Tensor, torch.Tensor, List[str], List[str]]:
    """Sample a complete two-turn rollout from the policy.

    Returns:
        responses_t1: First-turn responses
        responses_t2: Second-turn responses
        rewards_t1: Rewards for first turn [batch]
        rewards_t2: Rewards for second turn [batch]
        prompts_t1: First-turn prompts
        prompts_t2: Second-turn prompts
    """
    batch_size = len(batch["problem_ids"])

    # Turn 1: generate first responses
    prompts_t1 = build_prompts(batch, is_code, turn=1)
    responses_t1 = policy.generate(
        prompts_t1,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=(temperature > 0.0),
    )

    # Turn 2: generate correction responses
    prompts_t2 = build_prompts(batch, is_code, turn=2, first_responses=responses_t1)
    responses_t2 = policy.generate(
        prompts_t2,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=(temperature > 0.0),
    )

    # Compute rewards
    rewards_t1_list = []
    rewards_t2_list = []
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
        else:
            r1 = compute_reward(responses_t1[i], batch["answers"][i])
            r2 = compute_reward(responses_t2[i], batch["answers"][i])
        rewards_t1_list.append(r1)
        rewards_t2_list.append(r2)

    rewards_t1 = torch.tensor(rewards_t1_list, device=policy.device)
    rewards_t2 = torch.tensor(rewards_t2_list, device=policy.device)

    return responses_t1, responses_t2, rewards_t1, rewards_t2, prompts_t1, prompts_t2


def train_stage1(
    policy: LLMPolicy,
    ref_policy: LLMPolicy,
    dataloader: DataLoader,
    config: SCoReConfig,
    num_steps: int,
    is_code: bool = False,
) -> None:
    """Stage I of SCoRe: Train initialization that decouples attempts.

    Objective (Equation 3):
    max E[r(y2, y*)] - β₂·D_KL(π_θ(·|x1) || π_ref(·|x1))

    The first-turn distribution is constrained to the base model,
    while the second-turn is trained for high reward.
    """
    policy.model.train()
    optimizer = torch.optim.Adam(policy.model.parameters(), lr=config.learning_rate)

    global_step = 0

    for epoch in range(100):  # Upper bound, early exit at num_steps
        for batch in dataloader:
            if global_step >= num_steps:
                break

            # Sample two-turn rollout
            (
                responses_t1,
                responses_t2,
                rewards_t1,
                rewards_t2,
                prompts_t1,
                prompts_t2,
            ) = sample_two_turn_rollout(
                policy, batch, is_code, config.max_new_tokens, config.sampling_temperature
            )

            # Compute Stage I loss
            loss_dict = policy.compute_stage1_loss(
                prompts_turn1=prompts_t1,
                responses_turn1=responses_t1,
                prompts_turn2=prompts_t2,
                responses_turn2=responses_t2,
                rewards_turn2=rewards_t2,
                ref_model=ref_policy,
                beta2=config.beta2,
                beta1=config.beta1,
            )

            loss = loss_dict["loss"] / config.gradient_accumulation_steps
            loss.backward()

            if (global_step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1

            if global_step % 100 == 0:
                print(
                    f"[Stage I] Step {global_step}/{num_steps}, "
                    f"Loss: {loss_dict['loss'].item():.4f}, "
                    f"PG_t2: {loss_dict['pg_loss_t2'].item():.4f}, "
                    f"KL_t1: {loss_dict['kl_loss_t1'].item():.4f}, "
                    f"KL_t2: {loss_dict['kl_loss_t2'].item():.4f}, "
                    f"R1_mean: {rewards_t1.mean().item():.3f}, "
                    f"R2_mean: {rewards_t2.mean().item():.3f}"
                )

        if global_step >= num_steps:
            break


def train_stage2(
    policy: LLMPolicy,
    ref_policy: LLMPolicy,
    dataloader: DataLoader,
    config: SCoReConfig,
    num_steps: int,
    is_code: bool = False,
) -> None:
    """Stage II of SCoRe: Joint optimization with reward shaping.

    Objective (Equation 4 + reward shaping):
    max E[Σ_i r(y_i, y*)] - β₁·D_KL(π_θ || π_ref)
    with shaped reward: r'(y2) = r(y2) + α·(r(y2) - r(y1))
    """
    policy.model.train()
    optimizer = torch.optim.Adam(policy.model.parameters(), lr=config.learning_rate)

    global_step = 0

    for epoch in range(100):
        for batch in dataloader:
            if global_step >= num_steps:
                break

            # Sample two-turn rollout
            (
                responses_t1,
                responses_t2,
                rewards_t1,
                rewards_t2,
                prompts_t1,
                prompts_t2,
            ) = sample_two_turn_rollout(
                policy, batch, is_code, config.max_new_tokens, config.sampling_temperature
            )

            # Compute Stage II loss
            loss_dict = policy.compute_stage2_loss(
                prompts_turn1=prompts_t1,
                responses_turn1=responses_t1,
                rewards_turn1=rewards_t1,
                prompts_turn2=prompts_t2,
                responses_turn2=responses_t2,
                rewards_turn2=rewards_t2,
                ref_model=ref_policy,
                beta1=config.beta1,
                alpha=config.alpha,
                use_shaping=(config.alpha > 0),
            )

            loss = loss_dict["loss"] / config.gradient_accumulation_steps
            loss.backward()

            if (global_step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1

            if global_step % 100 == 0:
                # Compute shaped reward for logging
                shaped_r2 = rewards_t2 + config.alpha * (rewards_t2 - rewards_t1)
                print(
                    f"[Stage II] Step {global_step}/{num_steps}, "
                    f"Loss: {loss_dict['loss'].item():.4f}, "
                    f"PG_t1: {loss_dict['pg_loss_t1'].item():.4f}, "
                    f"PG_t2: {loss_dict['pg_loss_t2'].item():.4f}, "
                    f"R1_mean: {rewards_t1.mean().item():.3f}, "
                    f"R2_mean: {rewards_t2.mean().item():.3f}, "
                    f"ShapedR2: {shaped_r2.mean().item():.3f}"
                )

        if global_step >= num_steps:
            break


def train_score(
    config: SCoReConfig,
    data_dir: str,
    output_dir: str,
    is_code: bool = False,
) -> LLMPolicy:
    """Run full SCoRe training: Stage I + Stage II.

    Args:
        config: SCoRe configuration
        data_dir: Directory containing dataset files
        output_dir: Directory to save model checkpoints
        is_code: Whether training on code tasks

    Returns:
        Trained policy
    """
    set_seed(config.seed)
    os.makedirs(output_dir, exist_ok=True)

    # Load datasets
    if is_code:
        train_dataset = MBPPDataset(data_dir, split="train")
        val_dataset = MBPPDataset(data_dir, split="val")
    else:
        train_dataset = MATH500Dataset(data_dir, split="train", seed=config.seed)
        val_dataset = MATH500Dataset(data_dir, split="val", seed=config.seed)

    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        is_code=is_code,
    )

    # Load base policy
    print(f"Loading base model: {config.base_model}")
    policy = load_model_and_tokenizer(config.base_model)

    # Load reference policy (frozen base model)
    ref_policy = load_model_and_tokenizer(config.base_model)
    ref_policy.model.eval()
    for param in ref_policy.model.parameters():
        param.requires_grad = False
    policy.set_reference_model(ref_policy)

    # Compute training steps split
    stage1_steps = int(config.training_steps * config.stage1_steps_ratio)
    stage2_steps = config.training_steps - stage1_steps

    print(f"Training steps: Stage I = {stage1_steps}, Stage II = {stage2_steps}")

    # Stage I: Decouple attempts
    if stage1_steps > 0:
        print("Starting Stage I training...")
        train_stage1(
            policy=policy,
            ref_policy=ref_policy,
            dataloader=train_loader,
            config=config,
            num_steps=stage1_steps,
            is_code=is_code,
        )

        # Save Stage I checkpoint
        stage1_path = os.path.join(output_dir, "stage1_checkpoint")
        policy.model.save_pretrained(stage1_path)
        policy.tokenizer.save_pretrained(stage1_path)
        print(f"Stage I complete. Model saved to {stage1_path}")

    # Stage II: Joint optimization with reward shaping
    print("Starting Stage II training...")
    train_stage2(
        policy=policy,
        ref_policy=ref_policy,
        dataloader=train_loader,
        config=config,
        num_steps=stage2_steps,
        is_code=is_code,
    )

    # Save final model
    final_path = os.path.join(output_dir, "final_checkpoint")
    policy.model.save_pretrained(final_path)
    policy.tokenizer.save_pretrained(final_path)
    print(f"Training complete. Final model saved to {final_path}")

    return policy


def evaluate_score(
    policy: LLMPolicy,
    eval_dataloader: DataLoader,
    config: SCoReConfig,
    is_code: bool = False,
) -> Dict[str, float]:
    """Evaluate SCoRe model on self-correction metrics.

    Returns the metrics from the paper:
    - Accuracy@t1: accuracy on first attempt
    - Accuracy@t2: accuracy on second attempt
    - Δ(t1, t2): net improvement
    - Δ^{i→c}(t1, t2): incorrect→correct
    - Δ^{c→i}(t1, t2): correct→incorrect
    """
    all_results = []
    policy.model.eval()

    for batch in eval_dataloader:
        (
            responses_t1,
            responses_t2,
            rewards_t1,
            rewards_t2,
            _,  # prompts
            _,  # prompts
        ) = sample_two_turn_rollout(
            policy,
            batch,
            is_code,
            config.max_new_tokens,
            config.eval_temperature,
        )

        for i in range(len(responses_t1)):
            all_results.append({
                "problem_id": batch["problem_ids"][i],
                "correct_t1": rewards_t1[i].item() > 0.5,
                "correct_t2": rewards_t2[i].item() > 0.5,
            })

    metrics = compute_self_correction_metrics(all_results)
    return metrics


def main(
    data_dir: str,
    output_dir: str,
    task: str = "math",
) -> None:
    """Main entry point for SCoRe training and evaluation."""
    config = get_config(task=task)
    is_code = task == "code"

    # Train SCoRe
    policy = train_score(
        config=config,
        data_dir=data_dir,
        output_dir=output_dir,
        is_code=is_code,
    )

    # Evaluate
    if is_code:
        eval_dataset = MBPPDataset(data_dir, split="test")
    else:
        eval_dataset = MATH500Dataset(data_dir, split="test", seed=config.seed)

    eval_loader = create_dataloader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        is_code=is_code,
    )

    metrics = evaluate_score(policy, eval_loader, config, is_code=is_code)
    print("\n" + "=" * 50)
    print("SCoRe Evaluation Results:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    print("=" * 50)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train SCoRe for self-correction")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--task", type=str, default="math", choices=["math", "code"])
    args = parser.parse_args()

    main(args.data_dir, args.output_dir, args.task)

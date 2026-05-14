"""
Stage 3: PPO / MA-PPO Training.

Implements the full MA-RLHF training loop (Algorithm 1, §E).

Key flags:
  --use_macro_actions   : enable MA-PPO (default: False → vanilla PPO)
  --termination         : ngram | randomized_ngram | parser | ppl
  --n_gram N            : fixed n-gram length (default: 5)
  --sigma_assignment    : equal | unit | position_decayed (default: equal)

Usage (MA-PPO):
    python train_ppo.py \
        --task tldr \
        --policy_model_path outputs/sft \
        --critic_model_path outputs/rm \
        --reward_model_path outputs/rm \
        --use_macro_actions \
        --termination ngram \
        --n_gram 5 \
        --output_dir outputs/ma_ppo

Usage (vanilla PPO):
    python train_ppo.py \
        --task tldr \
        --policy_model_path outputs/sft \
        --critic_model_path outputs/rm \
        --reward_model_path outputs/rm \
        --output_dir outputs/ppo
"""

import argparse
import os
from typing import Optional

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from config import (
    PPOConfig, MacroActionConfig,
    TASK_TLDR, TASK_HH_RLHF, TASK_WEBGPT, TASK_APPS,
    CodeRewardConfig,
)
from data import get_dataset
from model import PolicyModel, CriticModel, RewardModel, ReferenceModel, load_tokenizer
from trainer import PPOTrainer, MAPPOTrainer, run_ppo_epoch
from evaluate import compute_rm_scores_on_dataset, evaluate_code_pass_at_k


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO / MA-PPO training for MA-RLHF")

    # Task and models
    parser.add_argument("--task", type=str, default=TASK_TLDR,
                        choices=[TASK_TLDR, TASK_HH_RLHF, TASK_WEBGPT, TASK_APPS])
    parser.add_argument("--policy_model_path", type=str, required=True)
    parser.add_argument("--critic_model_path", type=str, required=True)
    parser.add_argument("--reward_model_path", type=str, default=None,
                        help="Path to reward model (not needed for APPS)")
    parser.add_argument("--output_dir", type=str, default="outputs/ppo")

    # Sequence lengths
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_response_length", type=int, default=512)

    # Training
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--ppo_epochs", type=int, default=1)
    parser.add_argument("--total_steps", type=int, default=4600,
                        help="Total PPO update steps (paper trains ~4.6k steps for TL;DR)")
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--gae_gamma", type=float, default=1.0)
    parser.add_argument("--kl_coef", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=200)

    # Sampling
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)

    # Macro actions
    parser.add_argument("--use_macro_actions", action="store_true", default=False)
    parser.add_argument("--termination", type=str, default="ngram",
                        choices=["ngram", "randomized_ngram", "parser", "ppl"])
    parser.add_argument("--n_gram", type=int, default=5)
    parser.add_argument("--parser_cutoff", type=int, default=5)
    parser.add_argument("--sigma_assignment", type=str, default="equal",
                        choices=["equal", "unit", "position_decayed"])
    parser.add_argument("--use_full_sequence", action="store_true", default=False,
                        help="n=∞: treat entire response as one macro action (REINFORCE)")

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)

    return parser.parse_args()


def build_ppo_config(args: argparse.Namespace) -> PPOConfig:
    cfg = PPOConfig()
    cfg.model_name = args.policy_model_path
    cfg.task = args.task
    cfg.policy_model_path = args.policy_model_path
    cfg.critic_model_path = args.critic_model_path
    cfg.reward_model_path = args.reward_model_path or args.critic_model_path
    cfg.output_dir = args.output_dir
    cfg.max_prompt_length = args.max_prompt_length
    cfg.max_response_length = args.max_response_length
    cfg.batch_size = args.batch_size
    cfg.ppo_epochs = args.ppo_epochs
    cfg.clip_ratio = args.clip_ratio
    cfg.gae_lambda = args.gae_lambda
    cfg.gae_gamma = args.gae_gamma
    cfg.kl_coef_default = args.kl_coef
    cfg.warmup_steps = args.warmup_steps
    cfg.temperature = args.temperature
    cfg.top_p = args.top_p
    cfg.top_k = args.top_k
    cfg.seed = args.seed
    return cfg


def build_macro_config(args: argparse.Namespace) -> MacroActionConfig:
    cfg = MacroActionConfig()
    cfg.termination = args.termination
    cfg.n_gram = args.n_gram
    cfg.parser_cutoff = args.parser_cutoff
    cfg.sigma_assignment = args.sigma_assignment
    cfg.use_full_sequence = args.use_full_sequence
    return cfg


def load_reward_model_for_apps(device: torch.device) -> None:
    """APPS uses compiler signal instead of a learned RM (§B.5)."""
    return None


def compute_apps_reward(
    generated_code: str,
    test_cases: str,
    code_reward_cfg: CodeRewardConfig,
) -> float:
    """Adaptive compiler reward for APPS (§B.5, Eq. 5).

    R(x,y) = -0.3 + 1.3 * (N_pass / (N_pass + N_fail))  if compiled
    R(x,y) = -0.6                                         if runtime error
    R(x,y) = -1.0                                         if compile error
    """
    import json
    import subprocess
    import tempfile

    try:
        compile(generated_code, "<string>", "exec")
    except SyntaxError:
        return code_reward_cfg.compile_error_reward

    try:
        test_data = json.loads(test_cases)
        inputs = test_data.get("inputs", [])
        outputs = test_data.get("outputs", [])
    except Exception:
        return code_reward_cfg.compile_error_reward

    if not inputs:
        return code_reward_cfg.compile_error_reward

    n_pass = 0
    n_fail = 0
    for inp, expected in zip(inputs, outputs):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(generated_code)
            fname = f.name
        try:
            result = subprocess.run(
                ["python", fname],
                input=str(inp),
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                n_fail += 1
            elif result.stdout.strip() == str(expected).strip():
                n_pass += 1
            else:
                n_fail += 1
        except subprocess.TimeoutExpired:
            n_fail += 1
        except Exception:
            return code_reward_cfg.runtime_error_reward
        finally:
            os.unlink(fname)

    if n_pass + n_fail == 0:
        return code_reward_cfg.compile_error_reward

    return (
        code_reward_cfg.partial_pass_base
        + code_reward_cfg.partial_pass_scale * n_pass / (n_pass + n_fail)
    )


def train_ppo(args: argparse.Namespace):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ppo_cfg = build_ppo_config(args)
    macro_cfg = build_macro_config(args) if args.use_macro_actions else None

    tokenizer = load_tokenizer(args.policy_model_path)

    # Load models
    policy = PolicyModel(args.policy_model_path).to(device)
    critic = CriticModel(args.critic_model_path).to(device)
    ref_model = ReferenceModel(args.policy_model_path).to(device)

    if args.gradient_checkpointing:
        policy.gradient_checkpointing_enable()
        critic.gradient_checkpointing_enable()

    # Reward model (or None for APPS)
    if args.task == TASK_APPS:
        reward_model = None
        code_reward_cfg = CodeRewardConfig()
    else:
        reward_model = RewardModel(args.reward_model_path or args.critic_model_path).to(device)
        reward_head_path = os.path.join(
            args.reward_model_path or args.critic_model_path, "reward_head.pt"
        )
        if os.path.exists(reward_head_path):
            reward_model.reward_head.load_state_dict(torch.load(reward_head_path, map_location=device))
        reward_model.eval()
        code_reward_cfg = CodeRewardConfig()  # unused for non-APPS tasks

    # Build trainer
    if args.task == TASK_APPS:
        # APPS uses compiler signal as reward (§B.5)
        tokenizer_ref = tokenizer
        code_reward_cfg_ref = code_reward_cfg

        def apps_reward_fn(generated_ids: torch.Tensor) -> float:
            code = tokenizer_ref.decode(generated_ids[0], skip_special_tokens=True)
            return compute_apps_reward(code, "{}", code_reward_cfg_ref)

        reward_fn = apps_reward_fn
    else:
        reward_fn = None

    if args.use_macro_actions:
        trainer = MAPPOTrainer(
            policy=policy,
            critic=critic,
            reward_model=reward_model,
            ref_model=ref_model,
            ppo_config=ppo_cfg,
            macro_config=macro_cfg,
            device=device,
            reward_fn=reward_fn,
        )
        print(f"Using MA-PPO with termination={args.termination}, n_gram={args.n_gram}")
    else:
        trainer = PPOTrainer(
            policy=policy,
            critic=critic,
            reward_model=reward_model,
            ref_model=ref_model,
            config=ppo_cfg,
            device=device,
            reward_fn=reward_fn,
        )
        print("Using vanilla PPO")

    # Dataset
    dataset = get_dataset(
        task=args.task,
        stage="ppo",
        tokenizer=tokenizer,
        max_prompt_length=args.max_prompt_length,
        seed=args.seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    # Validation dataset for RM scoring
    if args.task != TASK_APPS:
        val_dataset = get_dataset(
            task=args.task,
            stage="ppo",
            tokenizer=tokenizer,
            max_prompt_length=args.max_prompt_length,
            seed=args.seed + 1,
        )

    os.makedirs(args.output_dir, exist_ok=True)
    global_step = 0

    while global_step < args.total_steps:
        global_step, metrics = run_ppo_epoch(
            trainer=trainer,
            dataloader=dataloader,
            global_step=global_step,
            log_interval=args.log_interval,
        )

        # Periodic evaluation
        if global_step % args.eval_interval == 0 and args.task != TASK_APPS:
            rm_score = compute_rm_scores_on_dataset(
                policy=policy,
                reward_model=reward_model,
                tokenizer=tokenizer,
                dataset=val_dataset,
                device=device,
                num_samples=200,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=args.max_response_length,
            )
            print(f"Step {global_step}: Validation RM score = {rm_score:.4f}")

        # Periodic checkpoint
        if global_step % args.save_interval == 0:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            policy.model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"Checkpoint saved to {ckpt_dir}")

        if global_step >= args.total_steps:
            break

    # Final save
    policy.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Final model saved to {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    train_ppo(args)

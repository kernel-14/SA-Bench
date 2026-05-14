"""Main script to run all SCoRe experiments and baselines.

Reproduces the experiments from Table 2 (MATH), Table 3 (HumanEval/MBPP),
Table 4 (ablations), and Figure 1 (inference-compute scaling).
"""

import argparse
import json
import os
from typing import Dict, Optional

import torch

from config import (
    SCoReConfig,
    STARConfig,
    PairSFTConfig,
    get_config,
)
from data import (
    MATH500Dataset,
    MBPPDataset,
    HumanEvalDataset,
    create_dataloader,
)
from metrics import compute_self_correction_metrics
from models import load_model_and_tokenizer
from prompts import build_mbpp_second_turn_prompt
from rewards import compute_reward
from train_score import (
    train_score,
    evaluate_score,
    set_seed,
    sample_two_turn_rollout,
)
from train_baselines import (
    train_star,
    train_pair_sft,
    evaluate_self_refine,
    train_standard_multi_turn_rl,
    train_single_turn_rl,
    collect_base_model_traces,
)
from utils import compute_inference_scaling


def run_math_experiments(
    data_dir: str,
    output_dir: str,
) -> Dict:
    """Run all MATH experiments (Table 2 and Table 4)."""
    os.makedirs(output_dir, exist_ok=True)
    config = get_config(task="math")
    all_results = {}

    set_seed(config.seed)

    # Load data
    train_dataset = MATH500Dataset(data_dir, split="train", seed=config.seed)
    test_dataset = MATH500Dataset(data_dir, split="test", seed=config.seed)
    train_loader = create_dataloader(train_dataset, config.batch_size, shuffle=True)
    test_loader = create_dataloader(test_dataset, config.batch_size, shuffle=False)

    # Load base model
    base_policy = load_model_and_tokenizer(config.base_model)

    # 1. Evaluate base model
    print("\n" + "=" * 60)
    print("Evaluating BASE MODEL (Gemini 1.5 Flash)")
    print("=" * 60)

    base_results = []
    for batch in test_loader:
        res_t1, res_t2, r_t1, r_t2, _, _ = sample_two_turn_rollout(
            base_policy, batch, is_code=False,
            max_new_tokens=config.max_new_tokens,
            temperature=config.eval_temperature,
        )
        for i in range(len(res_t1)):
            base_results.append({
                "problem_id": batch["problem_ids"][i],
                "correct_t1": r_t1[i].item() > 0.5,
                "correct_t2": r_t2[i].item() > 0.5,
            })
    base_metrics = compute_self_correction_metrics(base_results)
    all_results["base_model"] = base_metrics
    print(f"Base model: Acc@t1={base_metrics['accuracy_t1']:.3f}, "
          f"Acc@t2={base_metrics['accuracy_t2']:.3f}, "
          f"Δ={base_metrics['delta_t1_t2']:.4f}")

    # 2. Self-Refine baseline
    print("\n" + "=" * 60)
    print("Evaluating SELF-REFINE (Madaan et al. 2023)")
    print("=" * 60)
    sr_metrics = evaluate_self_refine(base_policy, test_loader, config)
    all_results["self_refine"] = sr_metrics
    print(f"Self-Refine: Acc@t1={sr_metrics['accuracy_t1']:.3f}, "
          f"Acc@t2={sr_metrics['accuracy_t2']:.3f}, "
          f"Δ={sr_metrics['delta_t1_t2']:.4f}")

    # 3. STaR (Zelikman et al. 2022)
    print("\n" + "=" * 60)
    print("Training STaR (Zelikman et al. 2022)")
    print("=" * 60)
    star_config = STARConfig()
    star_output = os.path.join(output_dir, "star")
    star_policy = train_star(
        base_policy, train_loader, config, star_config, star_output, is_code=False
    )
    star_results = []
    for batch in test_loader:
        res_t1, res_t2, r_t1, r_t2, _, _ = sample_two_turn_rollout(
            star_policy, batch, is_code=False,
            max_new_tokens=config.max_new_tokens,
            temperature=config.eval_temperature,
        )
        for i in range(len(res_t1)):
            star_results.append({
                "problem_id": batch["problem_ids"][i],
                "correct_t1": r_t1[i].item() > 0.5,
                "correct_t2": r_t2[i].item() > 0.5,
            })
    star_metrics = compute_self_correction_metrics(star_results)
    all_results["star"] = star_metrics
    print(f"STaR: Acc@t1={star_metrics['accuracy_t1']:.3f}, "
          f"Acc@t2={star_metrics['accuracy_t2']:.3f}, "
          f"Δ={star_metrics['delta_t1_t2']:.4f}")

    # 3b. STaR with D_STaR^+
    print("\n" + "=" * 60)
    print("Training STaR w/ D_STaR^+ (correct→correct data included)")
    print("=" * 60)
    star_plus_config = STARConfig(include_correct_to_correct=True)
    star_plus_output = os.path.join(output_dir, "star_plus")
    star_plus_policy = train_star(
        base_policy, train_loader, config, star_plus_config,
        star_plus_output, is_code=False,
    )
    star_plus_results = []
    for batch in test_loader:
        res_t1, res_t2, r_t1, r_t2, _, _ = sample_two_turn_rollout(
            star_plus_policy, batch, is_code=False,
            max_new_tokens=config.max_new_tokens,
            temperature=config.eval_temperature,
        )
        for i in range(len(res_t1)):
            star_plus_results.append({
                "problem_id": batch["problem_ids"][i],
                "correct_t1": r_t1[i].item() > 0.5,
                "correct_t2": r_t2[i].item() > 0.5,
            })
    star_plus_metrics = compute_self_correction_metrics(star_plus_results)
    all_results["star_plus"] = star_plus_metrics
    print(f"STaR+: Acc@t1={star_plus_metrics['accuracy_t1']:.3f}, "
          f"Acc@t2={star_plus_metrics['accuracy_t2']:.3f}, "
          f"Δ={star_plus_metrics['delta_t1_t2']:.4f}")

    # 4. Pair-SFT (Welleck et al. 2023)
    print("\n" + "=" * 60)
    print("Training Pair-SFT (Welleck et al. 2023)")
    print("=" * 60)
    sft_config = PairSFTConfig()
    sft_output = os.path.join(output_dir, "pair_sft")
    sft_policy = train_pair_sft(
        base_policy, train_loader, config, sft_config, sft_output, is_code=False
    )
    sft_results = []
    for batch in test_loader:
        res_t1, res_t2, r_t1, r_t2, _, _ = sample_two_turn_rollout(
            sft_policy, batch, is_code=False,
            max_new_tokens=config.max_new_tokens,
            temperature=config.eval_temperature,
        )
        for i in range(len(res_t1)):
            sft_results.append({
                "problem_id": batch["problem_ids"][i],
                "correct_t1": r_t1[i].item() > 0.5,
                "correct_t2": r_t2[i].item() > 0.5,
            })
    sft_metrics = compute_self_correction_metrics(sft_results)
    all_results["pair_sft"] = sft_metrics
    print(f"Pair-SFT: Acc@t1={sft_metrics['accuracy_t1']:.3f}, "
          f"Acc@t2={sft_metrics['accuracy_t2']:.3f}, "
          f"Δ={sft_metrics['delta_t1_t2']:.4f}")

    # 4b. Pair-SFT with D_SFT^+
    sft_plus_config = PairSFTConfig(include_correct_to_correct=True)
    sft_plus_output = os.path.join(output_dir, "pair_sft_plus")
    sft_plus_policy = train_pair_sft(
        base_policy, train_loader, config, sft_plus_config,
        sft_plus_output, is_code=False,
    )
    sft_plus_results = []
    for batch in test_loader:
        res_t1, res_t2, r_t1, r_t2, _, _ = sample_two_turn_rollout(
            sft_plus_policy, batch, is_code=False,
            max_new_tokens=config.max_new_tokens,
            temperature=config.eval_temperature,
        )
        for i in range(len(res_t1)):
            sft_plus_results.append({
                "problem_id": batch["problem_ids"][i],
                "correct_t1": r_t1[i].item() > 0.5,
                "correct_t2": r_t2[i].item() > 0.5,
            })
    sft_plus_metrics = compute_self_correction_metrics(sft_plus_results)
    all_results["pair_sft_plus"] = sft_plus_metrics
    print(f"Pair-SFT+: Acc@t1={sft_plus_metrics['accuracy_t1']:.3f}, "
          f"Acc@t2={sft_plus_metrics['accuracy_t2']:.3f}, "
          f"Δ={sft_plus_metrics['delta_t1_t2']:.4f}")

    # 5. SCoRe (Ours)
    print("\n" + "=" * 60)
    print("Training SCoRe")
    print("=" * 60)
    score_output = os.path.join(output_dir, "score")
    score_policy = train_score(
        config=config,
        data_dir=data_dir,
        output_dir=score_output,
        is_code=False,
    )
    score_metrics = evaluate_score(score_policy, test_loader, config, is_code=False)
    all_results["score"] = score_metrics
    print(f"SCoRe: Acc@t1={score_metrics['accuracy_t1']:.3f}, "
          f"Acc@t2={score_metrics['accuracy_t2']:.3f}, "
          f"Δ={score_metrics['delta_t1_t2']:.4f}")

    # 6. Ablations (Table 4)
    print("\n" + "=" * 60)
    print("Running ABLATIONS (Table 4)")
    print("=" * 60)

    # Ablation: w/o multi-turn training (single-turn RL)
    print("\n--- Ablation: w/o Multi-turn Training ---")
    single_turn_output = os.path.join(output_dir, "ablation_single_turn")
    single_turn_policy = load_model_and_tokenizer(config.base_model)
    ref_policy = load_model_and_tokenizer(config.base_model)
    ref_policy.model.eval()
    for p in ref_policy.model.parameters():
        p.requires_grad = False
    single_turn_policy.set_reference_model(ref_policy)
    train_single_turn_rl(
        single_turn_policy, ref_policy, train_loader, config,
        num_steps=config.training_steps, is_code=False,
    )
    single_turn_results = []
    for batch in test_loader:
        res_t1, res_t2, r_t1, r_t2, _, _ = sample_two_turn_rollout(
            single_turn_policy, batch, is_code=False,
            max_new_tokens=config.max_new_tokens,
            temperature=config.eval_temperature,
        )
        for i in range(len(res_t1)):
            single_turn_results.append({
                "problem_id": batch["problem_ids"][i],
                "correct_t1": r_t1[i].item() > 0.5,
                "correct_t2": r_t2[i].item() > 0.5,
            })
    st_metrics = compute_self_correction_metrics(single_turn_results)
    all_results["ablation_single_turn"] = st_metrics
    print(f"  w/o multi-turn: Acc@t1={st_metrics['accuracy_t1']:.3f}, "
          f"Acc@t2={st_metrics['accuracy_t2']:.3f}, Δ={st_metrics['delta_t1_t2']:.4f}")

    # Ablation: w/o Stage I
    print("\n--- Ablation: w/o Stage I ---")
    ablate_no_stage1_config = get_config(task="math", ablation="skip_stage1")
    no_stage1_output = os.path.join(output_dir, "ablation_no_stage1")
    no_stage1_policy = train_score(
        config=ablate_no_stage1_config,
        data_dir=data_dir,
        output_dir=no_stage1_output,
        is_code=False,
    )
    no_s1_metrics = evaluate_score(no_stage1_policy, test_loader, ablate_no_stage1_config, is_code=False)
    all_results["ablation_no_stage1"] = no_s1_metrics
    print(f"  w/o Stage I: Acc@t1={no_s1_metrics['accuracy_t1']:.3f}, "
          f"Acc@t2={no_s1_metrics['accuracy_t2']:.3f}, Δ={no_s1_metrics['delta_t1_t2']:.4f}")

    # Ablation: w/o reward shaping
    print("\n--- Ablation: w/o Reward Shaping ---")
    ablate_no_shape_config = get_config(task="math", ablation="no_reward_shaping")
    no_shape_output = os.path.join(output_dir, "ablation_no_shaping")
    no_shape_policy = train_score(
        config=ablate_no_shape_config,
        data_dir=data_dir,
        output_dir=no_shape_output,
        is_code=False,
    )
    no_s_metrics = evaluate_score(no_shape_policy, test_loader, ablate_no_shape_config, is_code=False)
    all_results["ablation_no_shaping"] = no_s_metrics
    print(f"  w/o reward shaping: Acc@t1={no_s_metrics['accuracy_t1']:.3f}, "
          f"Acc@t2={no_s_metrics['accuracy_t2']:.3f}, Δ={no_s_metrics['delta_t1_t2']:.4f}")

    # Ablation: STaR instead of REINFORCE in Stage II
    print("\n--- Ablation: STaR instead of REINFORCE Stage II ---")
    # First train Stage I normally
    stage1_only_config = get_config(task="math")
    stage1_only_config.training_steps = int(
        config.training_steps * config.stage1_steps_ratio
    )
    stage1_output = os.path.join(output_dir, "ablation_star_stage2_stage1")
    stage1_policy = train_score(
        config=stage1_only_config,
        data_dir=data_dir,
        output_dir=stage1_output,
        is_code=False,
    )
    # Then run STaR from Stage I checkpoint
    star_stage2_config = STARConfig()
    star_stage2_output = os.path.join(output_dir, "ablation_star_stage2")
    star_stage2_policy = train_star(
        stage1_policy, train_loader, config, star_stage2_config,
        star_stage2_output, is_code=False,
    )
    star_s2_results = []
    for batch in test_loader:
        res_t1, res_t2, r_t1, r_t2, _, _ = sample_two_turn_rollout(
            star_stage2_policy, batch, is_code=False,
            max_new_tokens=config.max_new_tokens,
            temperature=config.eval_temperature,
        )
        for i in range(len(res_t1)):
            star_s2_results.append({
                "problem_id": batch["problem_ids"][i],
                "correct_t1": r_t1[i].item() > 0.5,
                "correct_t2": r_t2[i].item() > 0.5,
            })
    star_s2_metrics = compute_self_correction_metrics(star_s2_results)
    all_results["ablation_star_stage2"] = star_s2_metrics
    print(f"  STaR Stage II: Acc@t1={star_s2_metrics['accuracy_t1']:.3f}, "
          f"Acc@t2={star_s2_metrics['accuracy_t2']:.3f}, Δ={star_s2_metrics['delta_t1_t2']:.4f}")

    # 7. Inference-compute scaling (Section 6.2)
    print("\n" + "=" * 60)
    print("Running INFERENCE-COMPUTE SCALING (Section 6.2)")
    print("=" * 60)
    test_problems = [test_dataset[i] for i in range(min(100, len(test_dataset)))]
    scaling_metrics = compute_inference_scaling(
        score_policy, test_problems, is_code=False,
        max_new_tokens=config.max_new_tokens,
        num_samples=32,
        temperature=0.7,
    )
    all_results["inference_scaling"] = scaling_metrics
    print(f"  Parallel (32): {scaling_metrics['parallel_accuracy']:.3f}")
    print(f"  Sequential (16+16): {scaling_metrics['sequential_accuracy']:.3f}")
    print(f"  Improvement: {scaling_metrics['improvement']:.3f}")

    # Save all results
    results_path = os.path.join(output_dir, "all_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {results_path}")

    return all_results


def run_code_experiments(
    data_dir: str,
    output_dir: str,
) -> Dict:
    """Run all code experiments (Table 3).

    Trains on MBPP, evaluates on HumanEval and MBPP-R.
    """
    os.makedirs(output_dir, exist_ok=True)
    config = get_config(task="code")
    all_results = {}

    set_seed(config.seed)

    # Load data
    train_dataset = MBPPDataset(data_dir, split="train")
    human_eval_dataset = HumanEvalDataset(data_dir)

    train_loader = create_dataloader(
        train_dataset, config.batch_size, shuffle=True, is_code=True
    )

    # For HumanEval evaluation, treat each problem as a batch of 1
    # since prompts have different lengths
    he_loader = create_dataloader(
        human_eval_dataset, config.batch_size, shuffle=False, is_code=True
    )

    # Load base model
    base_policy = load_model_and_tokenizer(config.base_model)

    from train_score import sample_two_turn_rollout

    # 1. Base model on HumanEval
    print("\n=== Base Model (Gemini 1.0 Pro) on HumanEval ===")
    base_results = []
    for batch in he_loader:
        res_t1, res_t2, r_t1, r_t2, _, _ = sample_two_turn_rollout(
            base_policy, batch, is_code=True,
            max_new_tokens=config.max_new_tokens,
            temperature=config.eval_temperature,
        )
        for i in range(len(res_t1)):
            base_results.append({
                "problem_id": batch["problem_ids"][i],
                "correct_t1": r_t1[i].item() > 0.5,
                "correct_t2": r_t2[i].item() > 0.5,
            })
    base_metrics = compute_self_correction_metrics(base_results)
    all_results["base_model"] = base_metrics
    print(f"  Base: Acc@t1={base_metrics['accuracy_t1']:.3f}, "
          f"Acc@t2={base_metrics['accuracy_t2']:.3f}, Δ={base_metrics['delta_t1_t2']:.4f}")

    # 2. Self-Refine (evaluate with prompting)
    # Note: Self-Refine is evaluated on the base model with specific prompts
    sr_results = []
    for batch in he_loader:
        res_t1, res_t2, r_t1, r_t2, _, _ = sample_two_turn_rollout(
            base_policy, batch, is_code=True,
            max_new_tokens=config.max_new_tokens,
            temperature=config.eval_temperature,
        )
        for i in range(len(res_t1)):
            sr_results.append({
                "problem_id": batch["problem_ids"][i],
                "correct_t1": r_t1[i].item() > 0.5,
                "correct_t2": r_t2[i].item() > 0.5,
            })
    sr_metrics = compute_self_correction_metrics(sr_results)
    all_results["self_refine"] = sr_metrics

    # 3. Pair-SFT
    print("\n=== Pair-SFT on Code ===")
    sft_config = PairSFTConfig(
        training_steps=config.training_steps,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
    )
    sft_output = os.path.join(output_dir, "pair_sft_code")
    sft_policy = train_pair_sft(
        base_policy, train_loader, config, sft_config, sft_output, is_code=True
    )
    sft_results = []
    for batch in he_loader:
        res_t1, res_t2, r_t1, r_t2, _, _ = sample_two_turn_rollout(
            sft_policy, batch, is_code=True,
            max_new_tokens=config.max_new_tokens,
            temperature=config.eval_temperature,
        )
        for i in range(len(res_t1)):
            sft_results.append({
                "problem_id": batch["problem_ids"][i],
                "correct_t1": r_t1[i].item() > 0.5,
                "correct_t2": r_t2[i].item() > 0.5,
            })
    sft_metrics = compute_self_correction_metrics(sft_results)
    all_results["pair_sft"] = sft_metrics
    print(f"  Pair-SFT: Acc@t1={sft_metrics['accuracy_t1']:.3f}, "
          f"Acc@t2={sft_metrics['accuracy_t2']:.3f}, Δ={sft_metrics['delta_t1_t2']:.4f}")

    # 4. SCoRe
    print("\n=== SCoRe on Code ===")
    score_output = os.path.join(output_dir, "score_code")
    score_policy = train_score(
        config=config,
        data_dir=data_dir,
        output_dir=score_output,
        is_code=True,
    )
    score_he_metrics = evaluate_score(score_policy, he_loader, config, is_code=True)
    all_results["score"] = score_he_metrics
    print(f"  SCoRe: Acc@t1={score_he_metrics['accuracy_t1']:.3f}, "
          f"Acc@t2={score_he_metrics['accuracy_t2']:.3f}, Δ={score_he_metrics['delta_t1_t2']:.4f}")

    # 5. MBPP-R evaluation (offline repair)
    print("\n=== MBPP-R Evaluation ===")
    mbpp_test = MBPPDataset(data_dir, split="test")
    mbpp_test_loader = create_dataloader(
        mbpp_test, config.batch_size, shuffle=False, is_code=True
    )

    for name, pol in [
        ("base", base_policy), ("sft", sft_policy), ("score", score_policy)
    ]:
        mbpp_r_results = []
        for batch in mbpp_test_loader:
            batch_size = len(batch["problem_ids"])
            prompts = [
                build_mbpp_second_turn_prompt(
                    batch["task_descriptions"][i],
                    batch["test_cases"][i],
                    "def solution():\n    pass",  # incorrect placeholder
                )
                for i in range(batch_size)
            ]
            responses = pol.generate(
                prompts,
                max_new_tokens=config.max_new_tokens,
                temperature=config.eval_temperature,
                do_sample=False,
            )
            for i in range(batch_size):
                r = compute_reward(
                    responses[i],
                    batch["code_solutions"][i],
                    is_code=True,
                    test_cases=batch["test_cases"][i],
                )
                mbpp_r_results.append({
                    "problem_id": batch["problem_ids"][i],
                    "correct": r > 0.5,
                })
        mbpp_r_acc = sum(1 for r in mbpp_r_results if r["correct"]) / len(mbpp_r_results)
        all_results[f"mbpp_r_{name}"] = mbpp_r_acc
        print(f"  MBPP-R {name}: {mbpp_r_acc:.3f}")

    # Save results
    results_path = os.path.join(output_dir, "all_results_code.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll code results saved to {results_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run SCoRe experiments from the paper"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to data directory containing MATH/ and MBPP/ datasets"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results",
        help="Directory to save results and checkpoints"
    )
    parser.add_argument(
        "--task", type=str, default="all",
        choices=["all", "math", "code"],
        help="Which experiments to run"
    )
    parser.add_argument(
        "--skip_baselines", action="store_true",
        help="Skip baseline training (only run SCoRe)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    args = parser.parse_args()

    results = {}

    if args.task in ["all", "math"]:
        print("\n" + "#" * 70)
        print("# MATH Experiments")
        print("#" * 70)
        math_output = os.path.join(args.output_dir, "math")
        math_results = run_math_experiments(args.data_dir, math_output)
        results["math"] = math_results

    if args.task in ["all", "code"]:
        print("\n" + "#" * 70)
        print("# Code Experiments")
        print("#" * 70)
        code_output = os.path.join(args.output_dir, "code")
        code_results = run_code_experiments(args.data_dir, code_output)
        results["code"] = code_results

    # Summary
    print("\n" + "#" * 70)
    print("# Summary of Results")
    print("#" * 70)

    if "math" in results:
        print("\nMATH Results (Table 2):")
        for method in ["base_model", "self_refine", "star_plus", "pair_sft", "score"]:
            if method in results["math"]:
                m = results["math"][method]
                print(f"  {method:20s}: Acc@t1={m['accuracy_t1']:.3f}, "
                      f"Acc@t2={m['accuracy_t2']:.3f}, Δ={m['delta_t1_t2']:.4f}")

        print("\nMATH Ablations (Table 4):")
        for method in ["score", "ablation_single_turn", "ablation_no_stage1",
                       "ablation_no_shaping", "ablation_star_stage2"]:
            if method in results["math"]:
                m = results["math"][method]
                print(f"  {method:25s}: Acc@t1={m['accuracy_t1']:.3f}, "
                      f"Acc@t2={m['accuracy_t2']:.3f}, Δ={m['delta_t1_t2']:.4f}")

    if "code" in results:
        print("\nCode Results (Table 3):")
        for method in ["base_model", "pair_sft", "score"]:
            if method in results["code"]:
                m = results["code"][method]
                print(f"  {method:20s}: Acc@t1={m['accuracy_t1']:.3f}, "
                      f"Acc@t2={m['accuracy_t2']:.3f}, Δ={m['delta_t1_t2']:.4f}")

        for key in ["mbpp_r_base", "mbpp_r_sft", "mbpp_r_score"]:
            if key in results["code"]:
                print(f"  {key:20s}: {results['code'][key]:.3f}")


if __name__ == "__main__":
    main()

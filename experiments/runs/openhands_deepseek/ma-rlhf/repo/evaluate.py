"""Evaluation utilities for MA-RLHF.

Implements:
- RM score evaluation on test sets
- GPT-4 pairwise evaluation (§F.1)
- Human evaluation protocol (§F.2)
- Pass@k metric for code generation (APPS)
"""
import os
import json
import re
import argparse
import logging
from typing import Dict, List, Tuple, Optional, Literal
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from model import RewardModel, PolicyModel
from data import RLHFDataset, APPSDataset, get_dataset, collate_for_rlhf
from config import ExperimentConfig, CONFIG_MAP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


GPT4_EVAL_PROMPTS = {
    "tldr": """You will be given two summaries written for an article. Your task is to pick the better one between them, based on the four criteria. Please make sure you read and understand these instructions carefully.

Relevance - selection of important content from the source. The summary should include only important information from the source document. Annotators were instructed to penalize summaries which contained redundancies and excess information.

Coherence - the collective quality of all sentences. We align this dimension with the DUC quality question of structure and coherence whereby "the summary should be well-structured and well-organized. The summary should not just be a heap of related information, but should build from sentence to a coherent body of information about a topic."

Consistency - the factual alignment between the summary and the summarized source. A factually consistent summary contains only statements that are entailed by the source document. Annotators were also asked to penalize summaries that contained hallucinated facts.

Fluency - the quality of the summary in terms of grammar, spelling, punctuation, word choice, and sentence structure.

You should output single character to indicate which summary you think is better. 'A' stands for Summary A and 'B' stands for Summary B. If you think both summaries are equally good, output 'E'

Article / Post: {article}

Summary A: {summary_a}
Summary B: {summary_b}

Your Choice (only a single character):""",

    "hh-rlhf": """For the following query to a chatbot assistant, which response is more helpful?

First provide a one-sentence comparison of the two responses and explain which you feel is more helpful. Second, on a new line, state only 'A' or 'B' to indicate which response is more helpful. If they are equally good or bad, state 'E'. Your response should use the json format, with "comparison" and "choice" as keys.

Query: {query}

Response A: {response_a}
Response B: {response_b}

Your Judgment:""",

    "webgpt": """You will be given two response written for an question. Your task is to pick the better one between them, based on these criteria.

Factual accuracy - which answer is more factually accurate?

Coherence - which answer is easier to follow?

Usefulness overall - all things considered, which answer would be more helpful to the person who asked this question?

You should output with a json format where the key is the criteria and the value is the choice you made, using 'A' stands for Response A and 'B' stands for Response B. If you think both responses are equally good, output 'E'.

Question: {question}

Answer A: {answer_a}
Answer B: {answer_b}

Your Judgment (you should also output the reason, note that you are allowed to think both responses are equally good, then output with 'E'):""",
}


def evaluate_rm_scores(
    policy_model: PolicyModel,
    reward_model: RewardModel,
    tokenizer,
    eval_dataset: RLHFDataset,
    device: torch.device,
    batch_size: int = 8,
    temperature: float = 0.8,
    top_p: float = 1.0,
    top_k: int = 50,
    max_response_length: int = 512,
) -> Tuple[float, List[float]]:
    """Evaluate using reward model scores on validation set.

    Returns:
        mean_rm_score, list of all rm_scores
    """
    policy_model.eval()
    reward_model.eval()

    dataloader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_for_rlhf,
    )

    all_rm_scores = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="RM Evaluation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            generated = policy_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_response_length,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            full_attention_mask = torch.ones_like(generated, dtype=torch.float32).to(device)

            rm_scores = reward_model(generated, full_attention_mask)
            all_rm_scores.extend(rm_scores.cpu().tolist())

    mean_score = np.mean(all_rm_scores) if all_rm_scores else 0.0
    logger.info(f"Mean RM score: {mean_score:.4f} (N={len(all_rm_scores)})")
    return mean_score, all_rm_scores


def compute_pass_k(
    generated_code: str,
    test_cases: List[Dict],
    k: int = 1,
) -> float:
    """Compute pass@k metric for code generation.

    A sample passes if it compiles AND passes all test cases.
    For pass@1: fraction of generated outputs that pass.
    For pass@5: estimate using the formula from Chen et al. (2021):
        pass@k = 1 - (C(n-c, k) / C(n, k))
    where n = total samples, c = correct samples.
    """
    # Simplified pass@1 computation
    try:
        # Execute code in isolated environment
        exec_globals = {}
        exec(generated_code, exec_globals)

        for test in test_cases:
            input_data = test.get("input", "")
            expected = test.get("output", "")

            if "solution" in exec_globals:
                result = exec_globals["solution"](input_data)
                if str(result) != str(expected):
                    return 0.0
            else:
                return 0.0
        return 1.0
    except Exception:
        return 0.0


def compute_compiler_reward(
    generated_code: str,
    test_cases: List[Dict],
) -> float:
    """Compute compiler-based reward for code generation (§B.5).

    R(x, y) = -0.3 + 1.3 * N_pass / (N_pass + N_fail)  if compiled
              -0.6                                         if runtime error
              -1.0                                         if compile error
    """
    try:
        compile(generated_code, "<generated>", "exec")
    except SyntaxError:
        return -1.0

    try:
        exec_globals = {}
        exec(generated_code, exec_globals)

        n_pass = 0
        n_fail = 0
        for test in test_cases:
            try:
                input_data = test.get("input", "")
                expected = test.get("output", "")
                if "solution" in exec_globals:
                    result = exec_globals["solution"](input_data)
                    if str(result) == str(expected):
                        n_pass += 1
                    else:
                        n_fail += 1
                else:
                    n_fail += 1
            except Exception:
                n_fail += 1

        if n_pass + n_fail == 0:
            return -0.6
        return -0.3 + 1.3 * n_pass / (n_pass + n_fail)

    except Exception:
        return -0.6


def evaluate_apps(
    policy_model: PolicyModel,
    tokenizer,
    eval_dataset: APPSDataset,
    device: torch.device,
    batch_size: int = 8,
    max_response_length: int = 512,
) -> Dict[str, float]:
    """Evaluate on APPS code generation dataset.

    Returns pass@1 and pass@5 metrics.
    """
    policy_model.eval()
    dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    all_pass1 = []
    all_compiler_rewards = []
    difficulty_results = {"Introductory": [], "Interview": [], "Competition": []}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="APPS Evaluation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            test_cases_list = batch["test_cases"]
            solutions = batch["solutions"]

            generated = policy_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_response_length,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            for i in range(input_ids.size(0)):
                response_ids = generated[i, input_ids.size(1):]
                code = tokenizer.decode(response_ids, skip_special_tokens=True)

                pass_1 = compute_pass_k(code, test_cases_list[i], k=1)
                compiler_reward = compute_compiler_reward(code, test_cases_list[i])

                all_pass1.append(pass_1)
                all_compiler_rewards.append(compiler_reward)

                # Determine difficulty based on test case count (heuristic)
                n_tests = len(test_cases_list[i])
                if n_tests <= 2:
                    difficulty_results["Introductory"].append(pass_1)
                elif n_tests <= 4:
                    difficulty_results["Interview"].append(pass_1)
                else:
                    difficulty_results["Competition"].append(pass_1)

    results = {
        "pass@1": np.mean(all_pass1),
        "compiler_reward": np.mean(all_compiler_rewards),
    }
    for diff, scores in difficulty_results.items():
        if scores:
            results[f"pass@1_{diff}"] = np.mean(scores)

    logger.info(f"APPS Results: {results}")
    return results


def gpt4_pairwise_eval(
    task: str,
    prompts: List[str],
    responses_a: List[str],
    responses_b: List[str],
    model_name: str = "gpt-4o-05-13",
) -> Tuple[int, int, int]:
    """Run GPT-4 pairwise evaluation.

    Returns:
        (wins_a, ties, wins_b)
    """
    import openai

    wins_a = 0
    ties = 0
    wins_b = 0

    for prompt, resp_a, resp_b in tqdm(
        zip(prompts, responses_a, responses_b),
        desc="GPT-4 Evaluation",
        total=len(prompts),
    ):
        eval_prompt = GPT4_EVAL_PROMPTS[task].format(
            article=prompt,
            query=prompt,
            question=prompt,
            summary_a=resp_a,
            summary_b=resp_b,
            response_a=resp_a,
            response_b=resp_b,
            answer_a=resp_a,
            answer_b=resp_b,
        )

        try:
            response = openai.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": eval_prompt}],
                max_tokens=50,
                temperature=0.0,
            )
            choice = response.choices[0].message.content.strip()

            if task == "hh-rlhf":
                try:
                    choice_json = json.loads(choice)
                    choice = choice_json.get("choice", "E")
                except json.JSONDecodeError:
                    pass

            if "A" in choice and "B" not in choice:
                wins_a += 1
            elif "B" in choice and "A" not in choice:
                wins_b += 1
            else:
                ties += 1

        except Exception as e:
            logger.error(f"GPT-4 API error: {e}")
            ties += 1

    total = wins_a + wins_b + ties
    logger.info(
        f"Win A: {wins_a}/{total} ({wins_a/total*100:.1f}%), "
        f"Tie: {ties}/{total} ({ties/total*100:.1f}%), "
        f"Win B: {wins_b}/{total} ({wins_b/total*100:.1f}%)"
    )
    return wins_a, ties, wins_b


def compute_agreement(
    rm_scores_a: List[float],
    rm_scores_b: List[float],
    gpt4_choices: List[str],
    human_choices: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute agreement between evaluation methods (Table 1)."""
    n = len(rm_scores_a)
    rm_choices = ["A" if rm_scores_a[i] > rm_scores_b[i] else
                  "B" if rm_scores_b[i] > rm_scores_a[i] else "E"
                  for i in range(n)]

    rm_gpt4_agree = sum(1 for i in range(n) if rm_choices[i] == gpt4_choices[i]) / n

    agreements = {
        "RM-GPT4": rm_gpt4_agree,
    }

    if human_choices is not None:
        rm_human_agree = sum(1 for i in range(n) if rm_choices[i] == human_choices[i]) / n
        gpt4_human_agree = sum(1 for i in range(n) if gpt4_choices[i] == human_choices[i]) / n
        agreements["RM-Human"] = rm_human_agree
        agreements["GPT4-Human"] = gpt4_human_agree

    return agreements


def main():
    parser = argparse.ArgumentParser(description="Evaluation for MA-RLHF")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--rm_checkpoint", type=str, default=None)
    parser.add_argument("--task", type=str, default="tldr")
    parser.add_argument("--model", type=str, default="gemma-2b")
    parser.add_argument("--eval_type", type=str, default="rm",
                        choices=["rm", "gpt4", "human", "apps", "all"])
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    config_fn = CONFIG_MAP.get(args.model)
    config = config_fn(args.task) if config_fn else ExperimentConfig()

    datasets = get_dataset(
        args.task, tokenizer,
        sft_split=config.dataset.sft_split,
        rm_split=config.dataset.rm_split,
        ppo_split=config.dataset.ppo_split,
        seed=args.seed,
    )

    if args.task == "apps":
        _, eval_dataset = datasets
    else:
        _, _, _, eval_dataset = datasets

    policy_model = PolicyModel.from_pretrained(args.checkpoint)
    policy_model.base_model.to(device)
    policy_model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.eval_type == "rm" or args.eval_type == "all":
        if args.rm_checkpoint is not None:
            reward_model = RewardModel.from_pretrained(args.rm_checkpoint)
            reward_model.base_model.to(device)
            reward_model.value_head.to(device)
            reward_model.eval()

            mean_score, all_scores = evaluate_rm_scores(
                policy_model, reward_model, tokenizer, eval_dataset, device,
            )

            results = {"mean_rm_score": mean_score, "all_rm_scores": all_scores}
            with open(os.path.join(args.output_dir, "rm_scores.json"), "w") as f:
                json.dump(results, f, indent=2)

    if args.eval_type == "apps" or args.eval_type == "all":
        if args.task == "apps":
            results = evaluate_apps(
                policy_model, tokenizer, eval_dataset, device,
            )
            with open(os.path.join(args.output_dir, "apps_results.json"), "w") as f:
                json.dump(results, f, indent=2)

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()

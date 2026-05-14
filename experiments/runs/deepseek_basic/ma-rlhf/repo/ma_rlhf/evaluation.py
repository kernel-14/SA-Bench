"""
Evaluation utilities for MA-RLHF.

Implements the evaluation methods described in Section 4.1:
- RM score evaluation
- GPT-4 pairwise evaluation
- Human evaluation (annotation guidelines)
- Best-of-N (rejection sampling) evaluation
- Pass@k for code generation

Reference: Section 4.1, Appendix F.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional


# ---------------------------------------------------------------------------
# GPT-4 Evaluation Prompts (from Appendix F.1)
# ---------------------------------------------------------------------------

TLDR_EVAL_PROMPT = """You will be given two summaries written for an article. Your task is to pick the better one between them, based on the four criteria. Please make sure you read and understand these instructions carefully.

Relevance - selection of important content from the source. The summary should include only important information from the source document. Annotators were instructed to penalize summaries which contained redundancies and excess information.

Coherence - the collective quality of all sentences. We align this dimension with the DUC quality question of structure and coherence whereby "the summary should be well-structured and well-organized. The summary should not just be a heap of related information, but should build from sentence to a coherent body of information about a topic."

Consistency - the factual alignment between the summary and the summarized source. A factually consistent summary contains only statements that are entailed by the source document. Annotators were also asked to penalize summaries that contained hallucinated facts.

Fluency - the quality of the summary in terms of grammar, spelling, punctuation, word choice, and sentence structure.

You should output single character to indicate which summary you think is better. 'A' stands for Summary A and 'B' stands for Summary B. If you think both summaries are equally good, output 'E'

Article / Post:{article}

Summary A:{summary_a}
Summary B:{summary_b}

Your Choice (only a single character):"""


HHRHLF_EVAL_PROMPT = """For the following query to a chatbot assistant, which response is more helpful?

First provide a one-sentence comparison of the two responses and explain which you feel is more helpful. Second, on a new line, state only 'A' or 'B' to indicate which response is more helpful. If they are equally good or bad, state 'E'. Your response should use the json format, with "comparison" and "choice" as keys.

Query: {query}
Response A: {response_a}
Response B: {response_b}
Your Judgment:"""


WEBGPT_EVAL_PROMPT = """You will be given two response written for an question. Your task is to pick the better one between them, based on these criteria.

Factual accuracy - which answer is more factually accurate?

Coherence - which answer is easier to follow?

Usefulness overall - all things considered, which answer would be more helpful to the person who asked this question?

You should output with a json format where the key is the criteria and the value is the choice you made, using 'A' stands for Response A and 'B' stands for Response B. If you think both responses are equally good, output 'E'.

Question: {question}
Answer A: {answer_a}
Answer B: {answer_b}
Your Judgment (you should also output the reason, note that you are allowed to think both responses are equally good, then output with 'E'):"""


# ---------------------------------------------------------------------------
# RM Score Evaluation
# ---------------------------------------------------------------------------

def compute_rm_scores(
    reward_model,
    tokenizer,
    prompts: List[str],
    responses: List[str],
    max_length: int = 1024,
) -> List[float]:
    """
    Compute reward model scores for a set of prompt-response pairs.
    
    Args:
        reward_model: The trained reward model.
        tokenizer: Tokenizer matching the model.
        prompts: List of input prompts.
        responses: List of generated responses.
        max_length: Maximum sequence length.
    
    Returns:
        List of RM scores.
    """
    scores = []
    reward_model.eval()
    
    with torch.no_grad():
        for prompt, response in zip(prompts, responses):
            text = prompt + response
            encoded = tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            if torch.cuda.is_available():
                encoded = {k: v.cuda() for k, v in encoded.items()}
            
            outputs = reward_model(**encoded)
            score = outputs.logits[:, -1].squeeze().item()
            scores.append(score)
    
    return scores


# ---------------------------------------------------------------------------
# Best-of-N (Rejection Sampling)
# ---------------------------------------------------------------------------

def best_of_n_sampling(
    model,
    tokenizer,
    reward_model,
    prompt: str,
    n: int = 8,
    temperature: float = 0.8,
    max_new_tokens: int = 512,
    top_p: float = 1.0,
    top_k: int = 50,
) -> Tuple[str, float]:
    """
    Best-of-N (rejection sampling): Generate N responses and return
    the one with the highest reward model score.
    
    Args:
        model: The policy model.
        tokenizer: Tokenizer.
        reward_model: Reward model for scoring.
        prompt: Input prompt.
        n: Number of samples.
        temperature: Sampling temperature.
        max_new_tokens: Maximum new tokens to generate.
        top_p: Nucleus sampling parameter.
        top_k: Top-k sampling parameter.
    
    Returns:
        best_response: The highest-scoring response.
        best_score: The corresponding RM score.
    
    Reference: Section 4.4.
    """
    encoded = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        encoded = {k: v.cuda() for k, v in encoded.items()}
    
    responses = []
    scores = []
    
    model.eval()
    with torch.no_grad():
        for _ in range(n):
            outputs = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
            response_ids = outputs[0][encoded["input_ids"].size(1):]
            response = tokenizer.decode(response_ids, skip_special_tokens=True)
            responses.append(response)
            
            # Get RM score
            full_text = prompt + response
            full_encoded = tokenizer(
                full_text,
                max_length=1024,
                truncation=True,
                return_tensors="pt",
            )
            if torch.cuda.is_available():
                full_encoded = {k: v.cuda() for k, v in full_encoded.items()}
            
            rm_output = reward_model(**full_encoded)
            score = rm_output.logits[:, -1].squeeze().item()
            scores.append(score)
    
    best_idx = np.argmax(scores)
    return responses[best_idx], scores[best_idx]


def evaluate_best_of_n(
    model,
    tokenizer,
    reward_model,
    prompts: List[str],
    n_values: List[int] = [4, 8, 16, 32],
    temperatures: List[float] = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
    max_new_tokens: int = 512,
) -> Dict[Tuple[int, float], List[float]]:
    """
    Comprehensive Best-of-N evaluation across temperatures and sample sizes.
    
    Returns:
        Dictionary mapping (N, temperature) -> list of RM scores.
    
    Reference: Section 4.4, Figure 8.
    """
    results = {}
    for n in n_values:
        for temp in temperatures:
            scores = []
            for prompt in prompts:
                _, score = best_of_n_sampling(
                    model=model,
                    tokenizer=tokenizer,
                    reward_model=reward_model,
                    prompt=prompt,
                    n=n,
                    temperature=temp,
                    max_new_tokens=max_new_tokens,
                )
                scores.append(score)
            results[(n, temp)] = scores
    return results


# ---------------------------------------------------------------------------
# Pass@k for Code Generation
# ---------------------------------------------------------------------------

def compute_pass_at_k(
    n: int,
    c: int,
    k: int,
) -> float:
    """
    Compute pass@k metric with unbiased estimator.
    
    pass@k = 1 - C(n-c, k) / C(n, k)
    
    where n = total samples, c = number of correct samples.
    
    Args:
        n: Total number of samples.
        c: Number of correct samples.
        k: k in pass@k.
    
    Returns:
        Estimated pass@k value.
    
    Reference: Chen et al. (2021).
    """
    if n - c < k:
        return 1.0
    
    from math import comb
    return 1.0 - comb(n - c, k) / comb(n, k)


def evaluate_code_generation(
    model,
    tokenizer,
    test_cases: List[Dict],
    k_values: List[int] = [1, 5],
    n_samples: int = 200,
) -> Dict[int, float]:
    """
    Evaluate code generation using pass@k metric.
    
    Each test case includes a prompt and unit tests.
    
    Args:
        model: Code generation model.
        tokenizer: Tokenizer.
        test_cases: List of {"prompt": ..., "test_cases": [...]} dicts.
        k_values: List of k values for pass@k.
        n_samples: Number of samples per problem.
    
    Returns:
        Dictionary mapping k -> pass@k value.
    
    Reference: Appendix B.5, Table 3.
    """
    pass_at_k_results = {k: [] for k in k_values}
    
    for case in test_cases:
        prompt = case["prompt"]
        unit_tests = case["test_cases"]
        
        correct_count = 0
        for _ in range(n_samples):
            # Generate code
            encoded = tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                encoded = {k: v.cuda() for k, v in encoded.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **encoded,
                    max_new_tokens=512,
                    temperature=1.0,
                    do_sample=True,
                )
            
            code = tokenizer.decode(
                outputs[0][encoded["input_ids"].size(1):],
                skip_special_tokens=True
            )
            
            # Execute and check test cases
            try:
                exec_globals = {}
                exec(code, exec_globals)
                all_passed = True
                for test in unit_tests:
                    try:
                        result = eval(test, exec_globals)
                        if not result:
                            all_passed = False
                            break
                    except Exception:
                        all_passed = False
                        break
                if all_passed:
                    correct_count += 1
            except Exception:
                pass
        
        for k in k_values:
            pass_at_k = compute_pass_at_k(n_samples, correct_count, k)
            pass_at_k_results[k].append(pass_at_k)
    
    return {k: np.mean(pass_at_k_results[k]) for k in k_values}


# ---------------------------------------------------------------------------
# Win Rate Computation
# ---------------------------------------------------------------------------

def compute_win_rate(
    evaluation_results: List[str],
) -> Dict[str, float]:
    """
    Compute win rates from pairwise evaluation results.
    
    Args:
        evaluation_results: List of evaluation outcomes
            ('A', 'B', 'E' for tie, or 'win', 'loss', 'tie').
    
    Returns:
        Dictionary with win_rate, loss_rate, tie_rate.
    """
    total = len(evaluation_results)
    wins = sum(1 for r in evaluation_results if r in ['A', 'win'])
    losses = sum(1 for r in evaluation_results if r in ['B', 'loss'])
    ties = sum(1 for r in evaluation_results if r in ['E', 'tie'])
    
    return {
        "win_rate": wins / total if total > 0 else 0,
        "loss_rate": losses / total if total > 0 else 0,
        "tie_rate": ties / total if total > 0 else 0,
    }

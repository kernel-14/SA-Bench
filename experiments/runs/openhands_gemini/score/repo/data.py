
import torch
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer
from typing import Dict, List, Tuple, Union

from config import config
from prompts import (
    format_math_problem,
    format_math_self_correction_prompt,
    format_code_problem,
    format_code_self_correction_prompt,
)

def load_and_prepare_math_dataset(
    tokenizer: PreTrainedTokenizer,
    split: str = "train",
    is_eval: bool = False
) -> Dataset:
    """
    Loads and prepares the MATH dataset.

    Args:
        tokenizer: The tokenizer for the language model.
        split: The dataset split to load ('train', 'test').
        is_eval: If True, prepares the dataset for evaluation (only problems).

    Returns:
        A Hugging Face Dataset object with processed prompts.
    """
    # The paper mentions augmenting MATH training with 4500 problems from test
    # and reporting on the remaining 500 (MATH500). This suggests a custom split.
    # For now, we'll load the standard split.
    # "competition_math" often has 'question' and 'solution' fields.
    dataset = load_dataset(config.MATH_DATASET_PATH, split=split)

    processed_data = []
    for entry in dataset:
        problem_text = entry["question"]
        ground_truth_solution = entry["solution"] # This might need parsing to get final answer

        if is_eval:
            # For evaluation, we just need the problem and ground truth to calculate metrics
            processed_data.append({
                "problem": problem_text,
                "ground_truth_solution": ground_truth_solution
            })
        else:
            # For training, we need prompts for both turns.
            # Initially, for SFT or Stage I/II, we might only need the problem.
            # RL data collection will involve model generations.
            processed_data.append({
                "problem": problem_text,
                "ground_truth_solution": ground_truth_solution,
                "prompt_t1": format_math_problem(problem_text),
                # prompt_t2 will be dynamically generated based on t1_response during RL
            })

    return Dataset.from_list(processed_data)


def load_and_prepare_code_dataset(
    tokenizer: PreTrainedTokenizer,
    dataset_name: str, # "mbpp" for training, "humaneval" for evaluation
    split: str = "train",
    is_eval: bool = False
) -> Dataset:
    """
    Loads and prepares the Code dataset (MBPP or HumanEval).

    Args:
        tokenizer: The tokenizer for the language model.
        dataset_name: 'mbpp' or 'humaneval'.
        split: The dataset split to load ('train', 'test').
        is_eval: If True, prepares the dataset for evaluation.

    Returns:
        A Hugging Face Dataset object with processed prompts.
    """
    if dataset_name == "mbpp":
        # MBPP dataset usually has 'prompt', 'test_code', 'code' (canonical solution)
        dataset = load_dataset(config.MBPP_DATASET_PATH, split=split)
    elif dataset_name == "humaneval":
        # HumanEval dataset usually has 'prompt', 'test', 'canonical_solution'
        dataset = load_dataset(config.HUMANEVAL_DATASET_PATH, split=split)
    else:
        raise ValueError(f"Unknown code dataset: {dataset_name}")

    processed_data = []
    for entry in dataset:
        problem_description = entry["prompt"]
        test_cases = entry["test_code"] if dataset_name == "mbpp" else entry["test"]
        canonical_solution = entry["code"] if dataset_name == "mbpp" else entry["canonical_solution"]
        
        # For MBPP, the prompt includes a 3-shot example which needs to be handled
        # This implementation assumes the `prompt` field in MBPP dataset might already be few-shot or needs manual prep.
        # For a truly faithful reproduction, one might need to manually construct the 3-shot prompt.
        
        # For simplicity, we'll use the single-shot template defined in prompts.py.
        # The paper says: "canonical three-shot prompt for first-attempt training samples on MBPP".
        # This indicates we need to append 3 examples to the prompt for MBPP training.
        # For now, this will generate a single-shot equivalent.
        
        # In a full implementation, `format_code_problem` would take a list of examples
        # to construct the few-shot prompt for MBPP training.
        
        # Let's add a placeholder for 3-shot MBPP.
        if dataset_name == "mbpp" and not is_eval:
             # This is where 3-shot examples would be prepended.
             # For now, we'll just use the single problem.
             # The `seed_code` in MBPP can be empty or a partial solution.
             seed_code = "" # Or entry.get("seed_code", "") if available
        else:
            seed_code = ""

        if is_eval:
            processed_data.append({
                "problem_description": problem_description,
                "test_cases": test_cases,
                "ground_truth_solution": canonical_solution,
                "prompt_t1": format_code_problem(problem_description, test_cases, seed_code) # For evaluation, we evaluate T1
            })
        else:
            processed_data.append({
                "problem_description": problem_description,
                "test_cases": test_cases,
                "ground_truth_solution": canonical_solution,
                "prompt_t1": format_code_problem(problem_description, test_cases, seed_code),
            })

    return Dataset.from_list(processed_data)


def preprocess_for_sft(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    task_type: str = "math" # or "code"
) -> Dataset:
    """
    Preprocesses a dataset for Supervised Fine-Tuning (SFT).
    This assumes a dataset containing 'prompt' and 'response' fields.
    """
    def tokenize_function(examples):
        # For SFT, we fine-tune on (prompt, correct_response) pairs.
        # The prompt here would be the initial problem statement.
        # The response would be the ground_truth_solution.
        # We concatenate them for causal language modeling.

        if task_type == "math":
            prompts = [format_math_problem(p) for p in examples["problem"]]
            responses = examples["ground_truth_solution"]
        elif task_type == "code":
            prompts = [
                format_code_problem(desc, tests, "") # Assuming no seed code for SFT
                for desc, tests in zip(examples["problem_description"], examples["test_cases"])
            ]
            responses = examples["ground_truth_solution"]
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        # Concatenate prompt and response for training the model to generate the response given the prompt
        full_texts = [p + r + tokenizer.eos_token for p, r in zip(prompts, responses)]
        
        return tokenizer(
            full_texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

    # Apply tokenization
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Set labels for causal language modeling
    tokenized_dataset = tokenized_dataset.map(lambda examples: {"labels": examples["input_ids"].clone()})
    
    return tokenized_dataset


def create_rl_pipeline_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    task_type: str = "math" # or "code"
) -> Dataset:
    """
    Prepares a dataset for the RL pipeline.
    This typically involves just the initial prompts that the model will respond to.
    """
    processed_data = []
    for entry in dataset:
        if task_type == "math":
            initial_prompt = format_math_problem(entry["problem"])
        elif task_type == "code":
            initial_prompt = format_code_problem(entry["problem_description"], entry["test_cases"])
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        processed_data.append({
            "query": initial_prompt,
            "ground_truth": entry["ground_truth_solution"],
            "problem_metadata": entry # Keep original entry for reward calculation or further use
        })
    return Dataset.from_list(processed_data)


"""
Data loading and preprocessing utilities for MA-RLHF.

Handles loading and preprocessing of:
- TL;DR summarization dataset (Reddit posts + summaries)
- HH-RLHF dialogue dataset (helpful/harmless responses)
- WebGPT Comparisons dataset (question answering)
- APPS dataset (code generation)

Reference: Appendix B.1
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DatasetStats:
    """Statistics for each dataset as reported in Table 4."""
    name: str
    num_comparisons: int
    num_train: int
    num_test: int
    avg_prompt_tokens: float
    avg_chosen_tokens: float
    avg_rejected_tokens: float


DATASET_STATS = {
    "hhrlhf": DatasetStats(
        name="Anthropic HH-RLHF",
        num_comparisons=127500,
        num_train=112000,
        num_test=12500,
        avg_prompt_tokens=160,
        avg_chosen_tokens=83,
        avg_rejected_tokens=75,
    ),
    "tldr": DatasetStats(
        name="OpenAI Summarization",
        num_comparisons=179000,
        num_train=92900,
        num_test=86100,
        avg_prompt_tokens=325,
        avg_chosen_tokens=35,
        avg_rejected_tokens=33,
    ),
    "webgpt": DatasetStats(
        name="OpenAI WebGPT",
        num_comparisons=19600,
        num_train=18500,
        num_test=979,
        avg_prompt_tokens=49,
        avg_chosen_tokens=149,
        avg_rejected_tokens=137,
    ),
    "apps": DatasetStats(
        name="APPS",
        num_comparisons=10000,
        num_train=5000,
        num_test=5000,
        avg_prompt_tokens=453,
        avg_chosen_tokens=203,
        avg_rejected_tokens=-1,  # No rejected for APPS
    ),
}


def format_tldr_data(post: str, summary: str) -> str:
    """
    Format TL;DR data following Stiennon et al. (2020).
    
    Concatenates Reddit post and summary with appropriate formatting.
    
    Args:
        post: Reddit post text.
        summary: Human-written summary.
    
    Returns:
        Formatted text string.
    
    Reference: Appendix B.2, SFT Training.
    """
    return f"SUBREDDIT: r/posts\nTITLE: \nPOST: {post}\nTL;DR: {summary}"


def format_dialogue_data(
    prompt: str,
    response: str,
    use_chat_template: bool = True,
) -> str:
    """
    Format dialogue data using human-assistant chat template.
    
    Args:
        prompt: User query/message.
        response: Assistant response.
        use_chat_template: Whether to use chat template format.
    
    Returns:
        Formatted text string.
    
    Reference: Appendix B.2, SFT Training.
    """
    if use_chat_template:
        return f"Human: {prompt}\nAssistant: {response}"
    else:
        return f"{prompt}\n{response}"


def format_qa_data(question: str, answer: str) -> str:
    """
    Format question-answering data.
    
    Args:
        question: Question text.
        answer: Answer text.
    
    Returns:
        Formatted text string.
    """
    return f"Question: {question}\nAnswer: {answer}"


def format_code_data(prompt: str, code: str) -> str:
    """
    Format code generation data following Hendrycks et al. (2021).
    
    Args:
        prompt: Natural language problem description.
        code: Python code solution.
    
    Returns:
        Formatted text string.
    
    Reference: Appendix B.2.
    """
    return f"{prompt}\n\n```python\n{code}\n```"


def split_dataset(
    data: List[Dict],
    sft_ratio: float = 0.2,
    rm_ratio: float = 0.4,
    ppo_ratio: float = 0.4,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split dataset into SFT, RM, and PPO portions.
    
    Default split: 20% SFT, 40% RM, 40% PPO
    (as described in Appendix B.2)
    
    For APPS: 20% SFT, 0% RM, 80% PPO (no RM needed)
    
    Args:
        data: Full dataset.
        sft_ratio: Fraction for SFT.
        rm_ratio: Fraction for reward modeling.
        ppo_ratio: Fraction for PPO training.
        seed: Random seed.
    
    Returns:
        Tuple of (sft_data, rm_data, ppo_data).
    """
    import random
    random.seed(seed)
    
    n = len(data)
    indices = list(range(n))
    random.shuffle(indices)
    
    sft_end = int(n * sft_ratio)
    rm_end = sft_end + int(n * rm_ratio)
    
    sft_data = [data[i] for i in indices[:sft_end]]
    rm_data = [data[i] for i in indices[sft_end:rm_end]]
    ppo_data = [data[i] for i in indices[rm_end:]]
    
    return sft_data, rm_data, ppo_data

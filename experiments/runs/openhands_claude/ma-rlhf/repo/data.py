"""
Dataset loading and preprocessing for MA-RLHF.

Supports four tasks (§4.1, §B.1):
  - TL;DR summarization       (openai/summarize_from_feedback)
  - HH-RLHF dialogue          (Anthropic/hh-rlhf)
  - WebGPT QA comparisons     (openai/webgpt_comparisons)
  - APPS code generation      (codeparrot/apps)

Data split: 20% SFT / 40% RM / 40% PPO (§B.2).
For APPS: 80% PPO (no RM stage).

Each dataset exposes three PyTorch Dataset classes:
  - <Task>SFTDataset
  - <Task>RMDataset
  - <Task>PPODataset
"""

import random
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from datasets import load_dataset, DatasetDict
from transformers import PreTrainedTokenizer

from config import (
    TASK_TLDR, TASK_HH_RLHF, TASK_WEBGPT, TASK_APPS,
    DATASET_PATHS,
    SFT_SPLIT_RATIO, RM_SPLIT_RATIO, PPO_SPLIT_RATIO,
    APPS_PPO_SPLIT_RATIO,
)


# ---------------------------------------------------------------------------
# Prompt templates (§B.2)
# ---------------------------------------------------------------------------

TLDR_PROMPT_TEMPLATE = "SUBREDDIT: r/{subreddit}\n\nTITLE: {title}\n\nPOST: {post}\n\nTL;DR:"

HH_RLHF_PROMPT_TEMPLATE = "{dialogue}"

WEBGPT_PROMPT_TEMPLATE = "Human: {question}\n\nAssistant:"

APPS_PROMPT_TEMPLATE = (
    "Write a Python solution for the following problem:\n\n{problem}\n\n"
    "Your solution:\n```python\n"
)


# ---------------------------------------------------------------------------
# Utility: split a list of indices into SFT / RM / PPO portions
# ---------------------------------------------------------------------------

def _split_indices(
    n: int,
    sft_ratio: float = SFT_SPLIT_RATIO,
    rm_ratio: float = RM_SPLIT_RATIO,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    sft_end = int(n * sft_ratio)
    rm_end = sft_end + int(n * rm_ratio)
    return indices[:sft_end], indices[sft_end:rm_end], indices[rm_end:]


def _split_indices_apps(n: int, seed: int = 42) -> Tuple[List[int], List[int]]:
    """APPS has no RM stage: 20% SFT, 80% PPO."""
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    sft_end = int(n * (1.0 - APPS_PPO_SPLIT_RATIO))
    return indices[:sft_end], indices[sft_end:]


# ---------------------------------------------------------------------------
# Base dataset helpers
# ---------------------------------------------------------------------------

def _tokenize_and_pad(
    tokenizer: PreTrainedTokenizer,
    text: str,
    max_length: int,
    padding_side: str = "right",
) -> Dict[str, torch.Tensor]:
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = padding_side
    enc = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    tokenizer.padding_side = old_padding_side
    return {k: v.squeeze(0) for k, v in enc.items()}


# ---------------------------------------------------------------------------
# TL;DR Summarization (§B.1)
# ---------------------------------------------------------------------------

class TLDRSFTDataset(Dataset):
    """SFT dataset for TL;DR: (prompt + chosen_summary) pairs."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        max_length: int = 560,
        seed: int = 42,
    ):
        raw = load_dataset(DATASET_PATHS[TASK_TLDR], "comparisons", split=split)
        indices, _, _ = _split_indices(len(raw), seed=seed)
        self.data = [raw[i] for i in indices]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        post = item["info"]["post"]
        subreddit = item["info"].get("subreddit", "")
        title = item["info"].get("title", "")
        chosen = item["summaries"][0]["text"]

        prompt = TLDR_PROMPT_TEMPLATE.format(
            subreddit=subreddit, title=title, post=post
        )
        full_text = prompt + " " + chosen
        return _tokenize_and_pad(self.tokenizer, full_text, self.max_length)


class TLDRRMDataset(Dataset):
    """RM dataset for TL;DR: (prompt, chosen, rejected) triples."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        max_prompt_length: int = 512,
        max_response_length: int = 512,
        seed: int = 42,
    ):
        raw = load_dataset(DATASET_PATHS[TASK_TLDR], "comparisons", split=split)
        _, indices, _ = _split_indices(len(raw), seed=seed)
        self.data = [raw[i] for i in indices]
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        post = item["info"]["post"]
        subreddit = item["info"].get("subreddit", "")
        title = item["info"].get("title", "")
        prompt = TLDR_PROMPT_TEMPLATE.format(
            subreddit=subreddit, title=title, post=post
        )

        summaries = item["summaries"]
        chosen_idx = item.get("choice", 0)
        rejected_idx = 1 - chosen_idx
        chosen = summaries[chosen_idx]["text"]
        rejected = summaries[rejected_idx]["text"]

        chosen_enc = _tokenize_and_pad(
            self.tokenizer, prompt + " " + chosen,
            self.max_prompt_length + self.max_response_length
        )
        rejected_enc = _tokenize_and_pad(
            self.tokenizer, prompt + " " + rejected,
            self.max_prompt_length + self.max_response_length
        )
        return {
            "chosen_input_ids": chosen_enc["input_ids"],
            "chosen_attention_mask": chosen_enc["attention_mask"],
            "rejected_input_ids": rejected_enc["input_ids"],
            "rejected_attention_mask": rejected_enc["attention_mask"],
        }


class TLDRPPODataset(Dataset):
    """PPO prompt dataset for TL;DR."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        max_prompt_length: int = 512,
        seed: int = 42,
    ):
        raw = load_dataset(DATASET_PATHS[TASK_TLDR], "comparisons", split=split)
        _, _, indices = _split_indices(len(raw), seed=seed)
        self.data = [raw[i] for i in indices]
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        post = item["info"]["post"]
        subreddit = item["info"].get("subreddit", "")
        title = item["info"].get("title", "")
        prompt = TLDR_PROMPT_TEMPLATE.format(
            subreddit=subreddit, title=title, post=post
        )
        enc = _tokenize_and_pad(
            self.tokenizer, prompt, self.max_prompt_length, padding_side="left"
        )
        return enc


# ---------------------------------------------------------------------------
# HH-RLHF Dialogue (§B.1)
# ---------------------------------------------------------------------------

def _extract_hh_dialogue(item: Dict) -> Tuple[str, str, str]:
    """Extract (prompt, chosen_response, rejected_response) from HH-RLHF item."""
    chosen_text: str = item["chosen"]
    rejected_text: str = item["rejected"]

    # The last "Assistant:" turn is the response; everything before is the prompt
    split_token = "\n\nAssistant:"
    if split_token in chosen_text:
        prompt, chosen_response = chosen_text.rsplit(split_token, 1)
        prompt = prompt + split_token
    else:
        prompt = ""
        chosen_response = chosen_text

    if split_token in rejected_text:
        _, rejected_response = rejected_text.rsplit(split_token, 1)
    else:
        rejected_response = rejected_text

    return prompt, chosen_response.strip(), rejected_response.strip()


class HHRLHFSFTDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        max_length: int = 512,
        seed: int = 42,
    ):
        raw = load_dataset(DATASET_PATHS[TASK_HH_RLHF], split=split)
        indices, _, _ = _split_indices(len(raw), seed=seed)
        self.data = [raw[i] for i in indices]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        prompt, chosen, _ = _extract_hh_dialogue(self.data[idx])
        full_text = prompt + " " + chosen
        return _tokenize_and_pad(self.tokenizer, full_text, self.max_length)


class HHRLHFRMDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        max_length: int = 512,
        seed: int = 42,
    ):
        raw = load_dataset(DATASET_PATHS[TASK_HH_RLHF], split=split)
        _, indices, _ = _split_indices(len(raw), seed=seed)
        self.data = [raw[i] for i in indices]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        prompt, chosen, rejected = _extract_hh_dialogue(self.data[idx])
        chosen_enc = _tokenize_and_pad(
            self.tokenizer, prompt + " " + chosen, self.max_length
        )
        rejected_enc = _tokenize_and_pad(
            self.tokenizer, prompt + " " + rejected, self.max_length
        )
        return {
            "chosen_input_ids": chosen_enc["input_ids"],
            "chosen_attention_mask": chosen_enc["attention_mask"],
            "rejected_input_ids": rejected_enc["input_ids"],
            "rejected_attention_mask": rejected_enc["attention_mask"],
        }


class HHRLHFPPODataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        max_prompt_length: int = 512,
        seed: int = 42,
    ):
        raw = load_dataset(DATASET_PATHS[TASK_HH_RLHF], split=split)
        _, _, indices = _split_indices(len(raw), seed=seed)
        self.data = [raw[i] for i in indices]
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        prompt, _, _ = _extract_hh_dialogue(self.data[idx])
        enc = _tokenize_and_pad(
            self.tokenizer, prompt, self.max_prompt_length, padding_side="left"
        )
        return enc


# ---------------------------------------------------------------------------
# WebGPT Comparisons (§B.1)
# ---------------------------------------------------------------------------

class WebGPTSFTDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        max_length: int = 768,
        seed: int = 42,
    ):
        raw = load_dataset(DATASET_PATHS[TASK_WEBGPT], split=split)
        indices, _, _ = _split_indices(len(raw), seed=seed)
        self.data = [raw[i] for i in indices]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        question = item["question"]["full_text"]
        # Use the preferred answer (score > 0 means answer_0 preferred)
        score = item.get("score_0", 0)
        answer = item["answer_0"] if score >= 0 else item["answer_1"]
        prompt = WEBGPT_PROMPT_TEMPLATE.format(question=question)
        full_text = prompt + " " + answer
        return _tokenize_and_pad(self.tokenizer, full_text, self.max_length)


class WebGPTRMDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        max_length: int = 768,
        seed: int = 42,
    ):
        raw = load_dataset(DATASET_PATHS[TASK_WEBGPT], split=split)
        _, indices, _ = _split_indices(len(raw), seed=seed)
        self.data = [raw[i] for i in indices]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        question = item["question"]["full_text"]
        prompt = WEBGPT_PROMPT_TEMPLATE.format(question=question)
        score = item.get("score_0", 0)
        if score >= 0:
            chosen, rejected = item["answer_0"], item["answer_1"]
        else:
            chosen, rejected = item["answer_1"], item["answer_0"]

        chosen_enc = _tokenize_and_pad(
            self.tokenizer, prompt + " " + chosen, self.max_length
        )
        rejected_enc = _tokenize_and_pad(
            self.tokenizer, prompt + " " + rejected, self.max_length
        )
        return {
            "chosen_input_ids": chosen_enc["input_ids"],
            "chosen_attention_mask": chosen_enc["attention_mask"],
            "rejected_input_ids": rejected_enc["input_ids"],
            "rejected_attention_mask": rejected_enc["attention_mask"],
        }


class WebGPTPPODataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        max_prompt_length: int = 512,
        seed: int = 42,
    ):
        raw = load_dataset(DATASET_PATHS[TASK_WEBGPT], split=split)
        _, _, indices = _split_indices(len(raw), seed=seed)
        self.data = [raw[i] for i in indices]
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        question = item["question"]["full_text"]
        prompt = WEBGPT_PROMPT_TEMPLATE.format(question=question)
        enc = _tokenize_and_pad(
            self.tokenizer, prompt, self.max_prompt_length, padding_side="left"
        )
        return enc


# ---------------------------------------------------------------------------
# APPS Code Generation (§B.1, §B.5)
# ---------------------------------------------------------------------------

class APPSSFTDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        max_length: int = 1024,
        seed: int = 42,
    ):
        raw = load_dataset(DATASET_PATHS[TASK_APPS], split=split)
        sft_indices, _ = _split_indices_apps(len(raw), seed=seed)
        self.data = [raw[i] for i in sft_indices]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        problem = item["question"]
        solutions = item.get("solutions", "[]")
        import json
        try:
            sol_list = json.loads(solutions)
            solution = sol_list[0] if sol_list else ""
        except Exception:
            solution = ""
        prompt = APPS_PROMPT_TEMPLATE.format(problem=problem)
        full_text = prompt + solution + "\n```"
        return _tokenize_and_pad(self.tokenizer, full_text, self.max_length)


class APPSPPODataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        max_prompt_length: int = 600,
        seed: int = 42,
    ):
        raw = load_dataset(DATASET_PATHS[TASK_APPS], split=split)
        _, ppo_indices = _split_indices_apps(len(raw), seed=seed)
        self.data = [raw[i] for i in ppo_indices]
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        problem = item["question"]
        prompt = APPS_PROMPT_TEMPLATE.format(problem=problem)
        enc = _tokenize_and_pad(
            self.tokenizer, prompt, self.max_prompt_length, padding_side="left"
        )
        return enc


class APPSTestDataset(Dataset):
    """Test set for pass@k evaluation (§4.1, §4.5)."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        split: str = "test",
        max_prompt_length: int = 600,
    ):
        raw = load_dataset(DATASET_PATHS[TASK_APPS], split=split)
        self.data = list(raw)
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        problem = item["question"]
        prompt = APPS_PROMPT_TEMPLATE.format(problem=problem)
        enc = _tokenize_and_pad(
            self.tokenizer, prompt, self.max_prompt_length, padding_side="left"
        )
        return {
            **enc,
            "problem_id": idx,
            "difficulty": item.get("difficulty", "unknown"),
            "test_cases": item.get("input_output", "{}"),
        }


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def get_dataset(
    task: str,
    stage: str,
    tokenizer: PreTrainedTokenizer,
    split: str = "train",
    max_prompt_length: int = 512,
    max_response_length: int = 512,
    seed: int = 42,
) -> Dataset:
    """Factory that returns the appropriate Dataset for a given task and stage.

    Args:
        task: one of 'tldr', 'hh_rlhf', 'webgpt', 'apps'.
        stage: one of 'sft', 'rm', 'ppo', 'test'.
    """
    max_length = max_prompt_length + max_response_length

    mapping = {
        (TASK_TLDR, "sft"): lambda: TLDRSFTDataset(tokenizer, split, max_length, seed),
        (TASK_TLDR, "rm"): lambda: TLDRRMDataset(tokenizer, split, max_prompt_length, max_response_length, seed),
        (TASK_TLDR, "ppo"): lambda: TLDRPPODataset(tokenizer, split, max_prompt_length, seed),
        (TASK_HH_RLHF, "sft"): lambda: HHRLHFSFTDataset(tokenizer, split, max_length, seed),
        (TASK_HH_RLHF, "rm"): lambda: HHRLHFRMDataset(tokenizer, split, max_length, seed),
        (TASK_HH_RLHF, "ppo"): lambda: HHRLHFPPODataset(tokenizer, split, max_prompt_length, seed),
        (TASK_WEBGPT, "sft"): lambda: WebGPTSFTDataset(tokenizer, split, max_length, seed),
        (TASK_WEBGPT, "rm"): lambda: WebGPTRMDataset(tokenizer, split, max_length, seed),
        (TASK_WEBGPT, "ppo"): lambda: WebGPTPPODataset(tokenizer, split, max_prompt_length, seed),
        (TASK_APPS, "sft"): lambda: APPSSFTDataset(tokenizer, split, max_length, seed),
        (TASK_APPS, "ppo"): lambda: APPSPPODataset(tokenizer, split, max_prompt_length, seed),
        (TASK_APPS, "test"): lambda: APPSTestDataset(tokenizer, "test", max_prompt_length),
    }

    key = (task, stage)
    if key not in mapping:
        raise ValueError(f"No dataset for task={task}, stage={stage}")
    return mapping[key]()

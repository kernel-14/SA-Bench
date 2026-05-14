"""Dataset loading and preprocessing for MA-RLHF.

Supports: TL;DR, HH-RLHF, WebGPT Comparisons, APPS.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple
import numpy as np


PROMPT_TEMPLATES = {
    "tldr": "SUBREDDIT: r/{subreddit}\nTITLE: {title}\nPOST: {post}\nTL;DR:",
    "hh-rlhf": "<|user|>\n{query}\n<|assistant|>\n",
    "webgpt": "Question: {question}\nAnswer:",
    "apps": "{question}\n",
}


class SFTDataset(Dataset):
    """Dataset for Supervised Fine-Tuning stage."""
    def __init__(
        self,
        data: List[Dict],
        tokenizer: AutoTokenizer,
        max_length: int = 1024,
        task: str = "tldr",
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        prompt = self._format_prompt(item)
        response = item.get("chosen", item.get("answer", ""))
        full_text = prompt + response
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        prompt_len = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        labels = tokenized["input_ids"].clone()
        labels[:, :prompt_len] = -100  # Mask prompt tokens
        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }

    def _format_prompt(self, item: Dict) -> str:
        if self.task == "tldr":
            return PROMPT_TEMPLATES["tldr"].format(
                subreddit=item.get("subreddit", ""),
                title=item.get("title", ""),
                post=item.get("post", item.get("content", "")),
            )
        elif self.task == "hh-rlhf":
            return PROMPT_TEMPLATES["hh-rlhf"].format(query=item["prompt"])
        elif self.task == "webgpt":
            return PROMPT_TEMPLATES["webgpt"].format(question=item["question"])
        elif self.task == "apps":
            return PROMPT_TEMPLATES["apps"].format(question=item["question"])
        return ""


class PreferenceDataset(Dataset):
    """Dataset with chosen/rejected pairs for Reward Modeling."""
    def __init__(
        self,
        data: List[Dict],
        tokenizer: AutoTokenizer,
        max_length: int = 1024,
        task: str = "tldr",
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        prompt = self._format_prompt(item)
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_len = len(prompt_tokens)

        chosen_text = prompt + item["chosen"]
        rejected_text = prompt + item["rejected"]

        chosen_enc = self.tokenizer(
            chosen_text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        rejected_enc = self.tokenizer(
            rejected_text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )

        return {
            "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
            "prompt_len": prompt_len,
        }

    def _format_prompt(self, item: Dict) -> str:
        if self.task == "tldr":
            return PROMPT_TEMPLATES["tldr"].format(
                subreddit=item.get("subreddit", ""),
                title=item.get("title", ""),
                post=item.get("post", item.get("content", "")),
            )
        elif self.task == "hh-rlhf":
            return PROMPT_TEMPLATES["hh-rlhf"].format(query=item["prompt"])
        elif self.task == "webgpt":
            return PROMPT_TEMPLATES["webgpt"].format(question=item["question"])
        return ""


class RLHFDataset(Dataset):
    """Dataset for RLHF training — prompts only, responses generated on-the-fly."""
    def __init__(
        self,
        data: List[Dict],
        tokenizer: AutoTokenizer,
        max_prompt_length: int = 512,
        task: str = "tldr",
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.task = task

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        prompt = self._format_prompt(item)
        tokenized = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_prompt_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "prompt_text": prompt,
        }

    def _format_prompt(self, item: Dict) -> str:
        if self.task == "tldr":
            return PROMPT_TEMPLATES["tldr"].format(
                subreddit=item.get("subreddit", ""),
                title=item.get("title", ""),
                post=item.get("post", item.get("content", "")),
            )
        elif self.task == "hh-rlhf":
            return PROMPT_TEMPLATES["hh-rlhf"].format(query=item["prompt"])
        elif self.task == "webgpt":
            return PROMPT_TEMPLATES["webgpt"].format(question=item["question"])
        elif self.task == "apps":
            return PROMPT_TEMPLATES["apps"].format(question=item["question"])
        return ""


class APPSDataset(Dataset):
    """Dataset for APPS code generation task."""
    def __init__(
        self,
        data: List[Dict],
        tokenizer: AutoTokenizer,
        max_prompt_length: int = 600,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        prompt = PROMPT_TEMPLATES["apps"].format(question=item["question"])
        tokenized = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_prompt_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "prompt_text": prompt,
            "test_cases": item.get("test_cases", []),
            "solutions": item.get("solutions", []),
        }


def load_tldr_dataset(
    tokenizer: AutoTokenizer,
    sft_split: float = 0.2,
    rm_split: float = 0.4,
    ppo_split: float = 0.4,
    seed: int = 42,
) -> Tuple[SFTDataset, PreferenceDataset, RLHFDataset, RLHFDataset]:
    """Load TL;DR summarization dataset.

    Returns: (sft_dataset, rm_dataset, ppo_train_dataset, ppo_eval_dataset)
    """
    dataset = load_dataset("openai/summarize_from_feedback", "comparisons")
    train_data = dataset["train"]
    eval_data = dataset["validation"]

    train_list = [dict(item) for item in train_data]
    eval_list = [dict(item) for item in eval_data]

    np.random.seed(seed)
    np.random.shuffle(train_list)

    n = len(train_list)
    n_sft = int(n * sft_split)
    n_rm = int(n * rm_split)

    sft_data = train_list[:n_sft]
    rm_data = train_list[n_sft:n_sft + n_rm]
    ppo_data = train_list[n_sft + n_rm:]

    sft_dataset = SFTDataset(sft_data, tokenizer, task="tldr")
    rm_dataset = PreferenceDataset(rm_data, tokenizer, task="tldr")
    ppo_train = RLHFDataset(ppo_data, tokenizer, task="tldr")
    ppo_eval = RLHFDataset(eval_list, tokenizer, task="tldr")

    return sft_dataset, rm_dataset, ppo_train, ppo_eval


def load_hhrlhf_dataset(
    tokenizer: AutoTokenizer,
    sft_split: float = 0.2,
    rm_split: float = 0.4,
    ppo_split: float = 0.4,
    seed: int = 42,
) -> Tuple[SFTDataset, PreferenceDataset, RLHFDataset, RLHFDataset]:
    """Load Anthropic HH-RLHF dataset for dialogue."""
    dataset = load_dataset("Anthropic/hh-rlhf")
    train_data = dataset["train"]
    eval_data = dataset["test"]

    # Rename fields to standard format
    def rename_fields(example):
        parts = example["chosen"].split("\n\nAssistant: ")
        if len(parts) >= 2:
            prompt = parts[0]
            response = parts[1]
        else:
            prompt = example["chosen"]
            response = ""
        parts_r = example["rejected"].split("\n\nAssistant: ")
        if len(parts_r) >= 2:
            response_r = parts_r[1]
        else:
            response_r = ""
        return {"prompt": prompt, "chosen": response, "rejected": response_r}

    train_data = train_data.map(rename_fields)
    eval_data = eval_data.map(rename_fields)

    train_list = [dict(item) for item in train_data]
    eval_list = [dict(item) for item in eval_data]

    np.random.seed(seed)
    np.random.shuffle(train_list)

    n = len(train_list)
    n_sft = int(n * sft_split)
    n_rm = int(n * rm_split)

    sft_data = train_list[:n_sft]
    rm_data = train_list[n_sft:n_sft + n_rm]
    ppo_data = train_list[n_sft + n_rm:]

    sft_dataset = SFTDataset(sft_data, tokenizer, task="hh-rlhf")
    rm_dataset = PreferenceDataset(rm_data, tokenizer, task="hh-rlhf")
    ppo_train = RLHFDataset(ppo_data, tokenizer, task="hh-rlhf")
    ppo_eval = RLHFDataset(eval_list, tokenizer, task="hh-rlhf")

    return sft_dataset, rm_dataset, ppo_train, ppo_eval


def load_webgpt_dataset(
    tokenizer: AutoTokenizer,
    sft_split: float = 0.2,
    rm_split: float = 0.4,
    ppo_split: float = 0.4,
    seed: int = 42,
) -> Tuple[SFTDataset, PreferenceDataset, RLHFDataset, RLHFDataset]:
    """Load WebGPT Comparisons dataset for QA."""
    dataset = load_dataset("openai/webgpt_comparisons")
    full_data = dataset["train"]
    full_list = [dict(item) for item in full_data]

    np.random.seed(seed)
    np.random.shuffle(full_list)

    # 5% for validation (paper §B.1)
    n = len(full_list)
    n_eval = int(n * 0.05)
    eval_list = full_list[:n_eval]
    train_list = full_list[n_eval:]

    n_train = len(train_list)
    n_sft = int(n_train * sft_split)
    n_rm = int(n_train * rm_split)

    sft_data = train_list[:n_sft]
    rm_data = train_list[n_sft:n_sft + n_rm]
    ppo_data = train_list[n_sft + n_rm:]

    # WebGPT has different field names
    def rename_webgpt(item):
        question = item.get("question", {}).get("full_text", "")
        chosen = item.get("answer_0", "")
        rejected = item.get("answer_1", "")
        if item.get("score_0", 0) < item.get("score_1", 0):
            chosen, rejected = rejected, chosen
        return {"question": question, "chosen": chosen, "rejected": rejected}

    sft_data = [rename_webgpt(item) for item in sft_data]
    rm_data = [rename_webgpt(item) for item in rm_data]
    ppo_data = [rename_webgpt(item) for item in ppo_data]
    eval_list = [rename_webgpt(item) for item in eval_list]

    sft_dataset = SFTDataset(sft_data, tokenizer, task="webgpt")
    rm_dataset = PreferenceDataset(rm_data, tokenizer, task="webgpt")
    ppo_train = RLHFDataset(ppo_data, tokenizer, task="webgpt")
    ppo_eval = RLHFDataset(eval_list, tokenizer, task="webgpt")

    return sft_dataset, rm_dataset, ppo_train, ppo_eval


def load_apps_dataset(
    tokenizer: AutoTokenizer,
    max_prompt_length: int = 600,
) -> Tuple[APPSDataset, APPSDataset]:
    """Load APPS code generation dataset."""
    dataset = load_dataset("codeparrot/apps")
    train_data = dataset["train"]
    eval_data = dataset["test"]

    train_list = [dict(item) for item in train_data]
    eval_list = [dict(item) for item in eval_data]

    train_dataset = APPSDataset(train_list, tokenizer, max_prompt_length)
    eval_dataset = APPSDataset(eval_list, tokenizer, max_prompt_length)

    return train_dataset, eval_dataset


def get_dataset(
    task: str,
    tokenizer: AutoTokenizer,
    sft_split: float = 0.2,
    rm_split: float = 0.4,
    ppo_split: float = 0.4,
    seed: int = 42,
):
    """Unified dataset loader."""
    loaders = {
        "tldr": load_tldr_dataset,
        "hh-rlhf": load_hhrlhf_dataset,
        "webgpt": load_webgpt_dataset,
        "apps": load_apps_dataset,
    }
    if task not in loaders:
        raise ValueError(f"Unknown task: {task}. Choose from {list(loaders.keys())}")
    return loaders[task](tokenizer, sft_split, rm_split, ppo_split, seed)


def collate_for_rlhf(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for RLHF training."""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

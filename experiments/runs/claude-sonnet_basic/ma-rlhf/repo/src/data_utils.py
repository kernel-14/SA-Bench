"""
Data utilities for MA-RLHF experiments.

Handles dataset loading and preprocessing for:
- TL;DR summarization (OpenAI Summarization dataset)
- HH-RLHF dialogue generation (Anthropic HH-RLHF)
- WebGPT Comparisons (question answering)
- APPS (code generation)

Data splits follow the paper (Section B.2):
  - 20% for SFT
  - 40% for reward model training
  - 40% for PPO training
"""

import json
import random
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


def split_dataset(data: list, sft_ratio: float = 0.2, rm_ratio: float = 0.4) -> Tuple[list, list, list]:
    """
    Split dataset into SFT / RM / PPO portions.

    Args:
        data: Full dataset list.
        sft_ratio: Fraction for SFT (default 20%).
        rm_ratio: Fraction for RM (default 40%).

    Returns:
        (sft_data, rm_data, ppo_data)
    """
    n = len(data)
    n_sft = int(n * sft_ratio)
    n_rm = int(n * rm_ratio)

    sft_data = data[:n_sft]
    rm_data = data[n_sft:n_sft + n_rm]
    ppo_data = data[n_sft + n_rm:]
    return sft_data, rm_data, ppo_data


class TLDRDataset(Dataset):
    """
    TL;DR summarization dataset.

    Format: Reddit posts with human-annotated preference pairs.
    The policy is asked to generate summaries for Reddit posts.

    Prompt format follows Stiennon et al. (2020):
        SUBREDDIT: r/{subreddit}
        TITLE: {title}
        POST: {post}
        TL;DR:
    """

    PROMPT_TEMPLATE = "SUBREDDIT: r/{subreddit}\nTITLE: {title}\nPOST: {post}\nTL;DR:"

    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_prompt_len: int = 512,
        max_response_len: int = 128,
        mode: str = "ppo",  # 'sft', 'rm', 'ppo'
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len
        self.max_response_len = max_response_len
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        if "info" in item:
            # OpenAI summarization format
            post = item["info"].get("post", "")
            subreddit = item["info"].get("subreddit", "")
            title = item["info"].get("title", "")
        else:
            post = item.get("post", "")
            subreddit = item.get("subreddit", "")
            title = item.get("title", "")

        prompt = self.PROMPT_TEMPLATE.format(
            subreddit=subreddit, title=title, post=post
        )

        if self.mode == "ppo":
            enc = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_prompt_len,
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "prompt": prompt,
            }

        elif self.mode == "sft":
            chosen = item.get("chosen", item.get("summary", ""))
            full_text = prompt + " " + chosen
            enc = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_prompt_len + self.max_response_len,
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": enc["input_ids"].squeeze(0),
            }

        elif self.mode == "rm":
            chosen = item.get("chosen", "")
            rejected = item.get("rejected", "")
            chosen_text = prompt + " " + chosen
            rejected_text = prompt + " " + rejected

            chosen_enc = self.tokenizer(
                chosen_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_prompt_len + self.max_response_len,
            )
            rejected_enc = self.tokenizer(
                rejected_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_prompt_len + self.max_response_len,
            )
            return {
                "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
                "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
                "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
                "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
            }


class HHRLHFDataset(Dataset):
    """
    Anthropic HH-RLHF dialogue dataset.

    Format: Single-turn or multi-turn dialogues with preference labels.
    The policy generates helpful and harmless responses.

    Prompt format uses human-assistant chat template.
    """

    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_prompt_len: int = 512,
        max_response_len: int = 256,
        mode: str = "ppo",
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len
        self.max_response_len = max_response_len
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def _extract_prompt_and_response(self, text: str) -> Tuple[str, str]:
        """Extract prompt (human turns) and last assistant response."""
        # HH-RLHF format: alternating Human:/Assistant: turns
        if "\n\nAssistant:" in text:
            last_assistant_idx = text.rfind("\n\nAssistant:")
            prompt = text[:last_assistant_idx + len("\n\nAssistant:")]
            response = text[last_assistant_idx + len("\n\nAssistant:"):].strip()
        else:
            prompt = text
            response = ""
        return prompt, response

    def __getitem__(self, idx):
        item = self.data[idx]
        chosen_text = item.get("chosen", "")
        rejected_text = item.get("rejected", "")

        chosen_prompt, chosen_response = self._extract_prompt_and_response(chosen_text)

        if self.mode == "ppo":
            enc = self.tokenizer(
                chosen_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_prompt_len,
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "prompt": chosen_prompt,
            }

        elif self.mode == "sft":
            enc = self.tokenizer(
                chosen_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_prompt_len + self.max_response_len,
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": enc["input_ids"].squeeze(0),
            }

        elif self.mode == "rm":
            chosen_enc = self.tokenizer(
                chosen_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_prompt_len + self.max_response_len,
            )
            rejected_enc = self.tokenizer(
                rejected_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_prompt_len + self.max_response_len,
            )
            return {
                "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
                "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
                "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
                "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
            }


class WebGPTDataset(Dataset):
    """
    WebGPT Comparisons dataset for question answering.

    The policy generates responses that balance factual accuracy and coherence.
    5% of data is used for validation (no separate validation set provided).
    """

    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_prompt_len: int = 256,
        max_response_len: int = 512,
        mode: str = "ppo",
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len
        self.max_response_len = max_response_len
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item.get("question", {}).get("full_text", item.get("question", ""))

        # Determine which answer is preferred
        score_0 = item.get("score_0", 0)
        score_1 = item.get("score_1", 0)
        if score_0 >= score_1:
            chosen = item.get("answer_0", "")
            rejected = item.get("answer_1", "")
        else:
            chosen = item.get("answer_1", "")
            rejected = item.get("answer_0", "")

        prompt = f"Human: {question}\n\nAssistant:"

        if self.mode == "ppo":
            enc = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_prompt_len,
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "prompt": prompt,
            }

        elif self.mode == "sft":
            full_text = prompt + " " + chosen
            enc = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_prompt_len + self.max_response_len,
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": enc["input_ids"].squeeze(0),
            }

        elif self.mode == "rm":
            chosen_text = prompt + " " + chosen
            rejected_text = prompt + " " + rejected
            chosen_enc = self.tokenizer(
                chosen_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_prompt_len + self.max_response_len,
            )
            rejected_enc = self.tokenizer(
                rejected_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_prompt_len + self.max_response_len,
            )
            return {
                "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
                "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
                "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
                "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
            }


class APPSDataset(Dataset):
    """
    APPS (Automated Programming Progress Standard) dataset for code generation.

    The policy writes executable Python code based on natural language descriptions.
    Reward is based on compiler/test execution signal (Section B.5).
    """

    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_prompt_len: int = 600,
        max_response_len: int = 512,
        mode: str = "ppo",
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len
        self.max_response_len = max_response_len
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item.get("question", "")
        solutions = item.get("solutions", "[]")

        prompt = f"# Problem:\n{question}\n\n# Solution:\n"

        if self.mode == "ppo":
            enc = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_prompt_len,
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "prompt": prompt,
                "test_cases": item.get("input_output", "{}"),
            }

        elif self.mode == "sft":
            try:
                sol_list = json.loads(solutions) if isinstance(solutions, str) else solutions
                solution = sol_list[0] if sol_list else ""
            except (json.JSONDecodeError, IndexError):
                solution = ""
            full_text = prompt + solution
            enc = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_prompt_len + self.max_response_len,
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": enc["input_ids"].squeeze(0),
            }


def compute_code_reward(code: str, test_cases: dict) -> float:
    """
    Compute the adaptive compiler signal reward for code generation (Section B.5).

    R(x, y) = {
        -0.3 + 1.3 * N_pass / (N_pass + N_fail),  if y successfully compiled
        -0.6,                                        if y received runtime error
        -1.0,                                        if y received compile error
    }

    Args:
        code: Generated Python code string.
        test_cases: Dict with 'inputs' and 'outputs' lists.

    Returns:
        Scalar reward value.
    """
    import subprocess
    import tempfile
    import os

    # Check if code compiles
    try:
        compile(code, "<string>", "exec")
    except SyntaxError:
        return -1.0

    # Run test cases
    inputs = test_cases.get("inputs", [])
    outputs = test_cases.get("outputs", [])

    if not inputs:
        # No test cases: just check compilation
        return 0.0

    n_pass = 0
    n_fail = 0

    for inp, expected_out in zip(inputs, outputs):
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                tmp_path = f.name

            result = subprocess.run(
                ["python", tmp_path],
                input=str(inp),
                capture_output=True,
                text=True,
                timeout=5,
            )
            os.unlink(tmp_path)

            if result.returncode == 0:
                actual_out = result.stdout.strip()
                if actual_out == str(expected_out).strip():
                    n_pass += 1
                else:
                    n_fail += 1
            else:
                n_fail += 1
        except subprocess.TimeoutExpired:
            n_fail += 1
        except Exception:
            n_fail += 1

    if n_pass + n_fail == 0:
        return 0.0

    return -0.3 + 1.3 * n_pass / (n_pass + n_fail)


def collate_fn_pad(batch: List[Dict], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """
    Collate function that pads sequences to the same length within a batch.
    """
    keys = batch[0].keys()
    result = {}

    for key in keys:
        if key in ("prompt", "test_cases"):
            result[key] = [item[key] for item in batch]
            continue

        tensors = [item[key] for item in batch]
        if isinstance(tensors[0], torch.Tensor):
            max_len = max(t.size(0) for t in tensors)
            padded = []
            for t in tensors:
                pad_len = max_len - t.size(0)
                if pad_len > 0:
                    if "label" in key:
                        pad_val = -100  # ignore index for cross-entropy
                    elif "mask" in key:
                        pad_val = 0
                    else:
                        pad_val = pad_token_id
                    t = torch.cat([t, torch.full((pad_len,), pad_val, dtype=t.dtype)])
                padded.append(t)
            result[key] = torch.stack(padded)
        else:
            result[key] = tensors

    return result

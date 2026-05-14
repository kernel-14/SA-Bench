
import torch
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer
from typing import Dict, List, Optional
from torch.utils.data import DataLoader

from config import DataConfig, SFTConfig, RMConfig, PPOConfig

class DataCollatorForSFT:
    """
    Data collator for Supervised Fine-Tuning (SFT).
    Prepares input_ids and attention_mask for causal language modeling.
    """
    def __init__(self, tokenizer: AutoTokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer(
            [e["text"] for e in examples],
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch


class DataCollatorForRM:
    """
    Data collator for Reward Modeling (RM).
    Prepares pairs of chosen and rejected responses.
    The reward model loss function expects (input_ids, attention_mask) for chosen and rejected.
    """
    def __init__(self, tokenizer: AutoTokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        # Tokenize chosen responses
        chosen_batch = self.tokenizer(
            [e["chosen"] for e in examples],
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # Tokenize rejected responses
        rejected_batch = self.tokenizer(
            [e["rejected"] for e in examples],
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "chosen_input_ids": chosen_batch["input_ids"],
            "chosen_attention_mask": chosen_batch["attention_mask"],
            "rejected_input_ids": rejected_batch["input_ids"],
            "rejected_attention_mask": rejected_batch["attention_mask"],
        }


class DataCollatorForPPO:
    """
    Data collator for PPO (Reinforcement Learning from Human Feedback).
    Prepares prompts for the policy model to generate responses.
    """
    def __init__(self, tokenizer: AutoTokenizer, max_prompt_length: int, max_response_length: int):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        # Tokenize prompts
        prompt_batch = self.tokenizer(
            [e["prompt"] for e in examples],
            max_length=self.max_prompt_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "prompt_input_ids": prompt_batch["input_ids"],
            "prompt_attention_mask": prompt_batch["attention_mask"],
        }


class RLHFDataset:
    """
    Manages loading, splitting, and formatting datasets for SFT, RM, and PPO stages.
    """
    def __init__(self, data_config: DataConfig, tokenizer: AutoTokenizer):
        self.data_config = data_config
        self.tokenizer = tokenizer

    def load_and_prepare_tldr(self) -> Dict[str, DatasetDict]:
        """Loads and prepares the TL;DR summarization dataset."""
        # This is a placeholder; actual dataset loading might involve local files or Hugging Face hub.
        # For this reproduction, we assume the dataset can be loaded by name or from a path.
        # The paper refers to "TL;DR (Stiennon et al., 2020) dataset for text summarization"
        # and "Reddit TL;DR (Volske et al. ¨ , 2017)". A common HF dataset is `openai/tldr_summarize`.
        dataset = load_dataset("openai/tldr_summarize")

        # Split data into SFT, RM, PPO (20%, 40%, 40%) - Appendix B.2
        # Note: The paper mentions "93k human-annotated preference pairs and 86k pairs for validation"
        # for TL;DR. It seems the `chosen`/`rejected` fields are for RM, and `prompt` for PPO.
        # SFT uses "prompts and the chosen sentences as the instruction data".
        
        # SFT dataset preparation
        sft_data = dataset["train"].map(
            lambda x: {"text": self.tokenizer.bos_token + x["prompt"] + x["chosen"] + self.tokenizer.eos_token},
            remove_columns=["prompt", "chosen", "rejected", "policy", "label"]
        )
        sft_train = sft_data.select(range(int(len(sft_data) * self.data_config.sft_data_ratio)))

        # RM dataset preparation (using chosen/rejected pairs)
        rm_data = dataset["train"].map(
            lambda x: {
                "prompt": x["prompt"],
                "chosen": self.tokenizer.bos_token + x["prompt"] + x["chosen"] + self.tokenizer.eos_token,
                "rejected": self.tokenizer.bos_token + x["prompt"] + x["rejected"] + self.tokenizer.eos_token,
            }
        )
        rm_train_start_idx = len(sft_data) - len(sft_train) # Remaining data after SFT
        rm_train = rm_data.select(range(rm_train_start_idx, rm_train_start_idx + int(len(sft_data) * self.data_config.rm_data_ratio)))
        rm_eval = dataset["validation"].map(
            lambda x: {
                "prompt": x["prompt"],
                "chosen": self.tokenizer.bos_token + x["prompt"] + x["chosen"] + self.tokenizer.eos_token,
                "rejected": self.tokenizer.bos_token + x["prompt"] + x["rejected"] + self.tokenizer.eos_token,
            }
        )


        # PPO dataset preparation (using prompts)
        ppo_data = dataset["train"].map(
            lambda x: {"prompt": self.tokenizer.bos_token + x["prompt"]},
            remove_columns=["chosen", "rejected", "policy", "label"]
        )
        ppo_train_start_idx = len(sft_data) - len(sft_train) - len(rm_train)
        ppo_train = ppo_data.select(range(ppo_train_start_idx, ppo_train_start_idx + int(len(sft_data) * self.data_config.ppo_data_ratio)))
        ppo_eval = dataset["validation"].map(
            lambda x: {"prompt": self.tokenizer.bos_token + x["prompt"]},
            remove_columns=["chosen", "rejected", "policy", "label"]
        )

        return {
            "sft_train": sft_train,
            "rm_train": rm_train,
            "rm_eval": rm_eval,
            "ppo_train": ppo_train,
            "ppo_eval": ppo_eval,
        }

    def load_and_prepare_hh_rlhf(self) -> Dict[str, DatasetDict]:
        """Loads and prepares the Anthropic Helpful and Harmless (HH-RLHF) dataset."""
        # Using `Anthropic/hh-rlhf` from Hugging Face
        dataset = load_dataset("Anthropic/hh-rlhf")

        # SFT: "human-assistant chat template to format the instructions."
        # Use chosen responses for SFT.
        def format_chat_template_sft(example):
            return {"text": self.tokenizer.bos_token + example["chosen"] + self.tokenizer.eos_token}

        sft_data = dataset["train"].map(format_chat_template_sft, remove_columns=["chosen", "rejected"])
        sft_train = sft_data.select(range(int(len(sft_data) * self.data_config.sft_data_ratio)))

        # RM: "human-assistant chat template to format the instructions."
        def format_chat_template_rm(example):
            return {
                "prompt": self.tokenizer.bos_token + example["chosen"].split("Assistant:")[0] + "Assistant:",
                "chosen": self.tokenizer.bos_token + example["chosen"] + self.tokenizer.eos_token,
                "rejected": self.tokenizer.bos_token + example["rejected"] + self.tokenizer.eos_token,
            }

        rm_data = dataset["train"].map(format_chat_template_rm)
        rm_train_start_idx = len(sft_data) - len(sft_train)
        rm_train = rm_data.select(range(rm_train_start_idx, rm_train_start_idx + int(len(sft_data) * self.data_config.rm_data_ratio)))
        rm_eval = dataset["test"].map(format_chat_template_rm) # Using test set for eval

        # PPO: "human-assistant chat template to format the instructions."
        def format_chat_template_ppo(example):
            return {"prompt": self.tokenizer.bos_token + example["chosen"].split("Assistant:")[0] + "Assistant:"}

        ppo_data = dataset["train"].map(format_chat_template_ppo, remove_columns=["chosen", "rejected"])
        ppo_train_start_idx = len(sft_data) - len(sft_train) - len(rm_train)
        ppo_train = ppo_data.select(range(ppo_train_start_idx, ppo_train_start_idx + int(len(sft_data) * self.data_config.ppo_data_ratio)))
        ppo_eval = dataset["test"].map(format_chat_template_ppo, remove_columns=["chosen", "rejected"])

        return {
            "sft_train": sft_train,
            "rm_train": rm_train,
            "rm_eval": rm_eval,
            "ppo_train": ppo_train,
            "ppo_eval": ppo_eval,
        }

    def load_and_prepare_webgpt(self) -> Dict[str, DatasetDict]:
        """Loads and prepares the WebGPT Comparisons dataset."""
        # This dataset is not directly available on Hugging Face as `webgpt_comparisons`.
        # The paper references "WebGPT Comparison (Nakano et al., 2021)".
        # It says "default validation set of the WebGPT dataset" and "19.6k instances for training.
        # We split 5% instances for validation, as no separate validation set is provided."
        # For now, we will create a dummy dataset structure. In a real scenario, this would
        # involve specific loading logic for the WebGPT dataset files.
        print("WARNING: WebGPT Comparisons dataset is a placeholder. Please replace with actual loading logic.")
        # Create a dummy dataset for demonstration
        dummy_data = {
            "prompt": ["What is the capital of France?", "Who invented the lightbulb?"],
            "chosen": ["The capital of France is Paris.", "Thomas Edison invented the lightbulb."],
            "rejected": ["France's capital is Berlin.", "Nikola Tesla invented the lightbulb."],
        }
        dataset = DatasetDict({
            "train": Dataset.from_dict(dummy_data),
            "validation": Dataset.from_dict(dummy_data) # Using same for validation
        })

        # SFT: concatenate prompt and chosen
        sft_data = dataset["train"].map(
            lambda x: {"text": self.tokenizer.bos_token + x["prompt"] + x["chosen"] + self.tokenizer.eos_token},
            remove_columns=["prompt", "chosen", "rejected"]
        )
        sft_train = sft_data.select(range(int(len(sft_data) * self.data_config.sft_data_ratio)))

        # RM: chosen over rejected
        rm_data = dataset["train"].map(
            lambda x: {
                "prompt": x["prompt"],
                "chosen": self.tokenizer.bos_token + x["prompt"] + x["chosen"] + self.tokenizer.eos_token,
                "rejected": self.tokenizer.bos_token + x["prompt"] + x["rejected"] + self.tokenizer.eos_token,
            }
        )
        rm_train_start_idx = len(sft_data) - len(sft_train)
        rm_train = rm_data.select(range(rm_train_start_idx, rm_train_start_idx + int(len(sft_data) * self.data_config.rm_data_ratio)))
        
        # "We split 5% instances for validation" for RM eval
        webgpt_rm_eval_ratio = 0.05
        rm_eval_data = dataset["train"].select(range(len(dataset["train"]) - int(len(dataset["train"]) * webgpt_rm_eval_ratio), len(dataset["train"]))).map(
            lambda x: {
                "prompt": x["prompt"],
                "chosen": self.tokenizer.bos_token + x["prompt"] + x["chosen"] + self.tokenizer.eos_token,
                "rejected": self.tokenizer.bos_token + x["prompt"] + x["rejected"] + self.tokenizer.eos_token,
            }
        )


        # PPO: only prompt
        ppo_data = dataset["train"].map(
            lambda x: {"prompt": self.tokenizer.bos_token + x["prompt"]},
            remove_columns=["chosen", "rejected"]
        )
        ppo_train_start_idx = len(sft_data) - len(sft_train) - len(rm_train)
        ppo_train = ppo_data.select(range(ppo_train_start_idx, ppo_train_start_idx + int(len(sft_data) * self.data_config.ppo_data_ratio)))
        
        # PPO eval also from the 5% split
        ppo_eval = dataset["train"].select(range(len(dataset["train"]) - int(len(dataset["train"]) * webgpt_rm_eval_ratio), len(dataset["train"]))).map(
            lambda x: {"prompt": self.tokenizer.bos_token + x["prompt"]},
            remove_columns=["chosen", "rejected"]
        )

        return {
            "sft_train": sft_train,
            "rm_train": rm_train,
            "rm_eval": rm_eval_data,
            "ppo_train": ppo_train,
            "ppo_eval": ppo_eval,
        }

    def load_and_prepare_apps(self) -> Dict[str, DatasetDict]:
        """Loads and prepares the APPS dataset for code generation."""
        # "The APPS (Hendrycks et al., 2021) dataset. ... 5k training and 5k validation instances."
        # "format the instruction data in line with Hendrycks et al. (2021)."
        # "RM stage is omitted for this task."
        # "For the program synthesis dataset, 80% of the data is used in this stage, with both the
        #  policy and critic models initialized using the SFT model." (PPO stage)
        # "The pass@1 metric serves as the reward signal for program synthesis, compensating
        #  for the absence of a reward model."
        
        print("WARNING: APPS dataset loading is a placeholder. Please replace with actual loading logic.")
        # Dummy data for APPS
        dummy_apps_data = {
            "problem_id": [0, 1],
            "question": ["Write a function to add two numbers.", "Implement a sorting algorithm."],
            "solution": ["def add(a, b):\\n    return a + b", "def bubble_sort(arr):\\n    # ..."],
            "starter_code": ["", ""],
            "test_cases": ["", ""],
            "input_output": ["", ""],
            "difficulty": ["", ""]
        }
        dataset = DatasetDict({
            "train": Dataset.from_dict(dummy_apps_data),
            "test": Dataset.from_dict(dummy_apps_data) # Using test as eval as per paper
        })

        # SFT: "format the instruction data in line with Hendrycks et al. (2021)."
        # Usually: problem description + starter code + solution
        def format_apps_sft(example):
            # Simplified formatting, actual might be more complex
            instruction = example["question"]
            code = example["solution"]
            return {"text": self.tokenizer.bos_token + instruction + "\\n```python\\n" + code + "\\n```" + self.tokenizer.eos_token}

        sft_data = dataset["train"].map(format_apps_sft, remove_columns=dataset["train"].column_names)
        sft_train = sft_data.select(range(int(len(sft_data) * self.data_config.apps_sft_data_ratio)))

        # RM stage is omitted.

        # PPO: only prompt (problem description + starter code)
        def format_apps_ppo(example):
            instruction = example["question"]
            # For generation, we provide the prompt and expect the solution
            return {"prompt": self.tokenizer.bos_token + instruction + "\\n```python\\n"}

        ppo_data = dataset["train"].map(format_apps_ppo, remove_columns=dataset["train"].column_names)
        ppo_train_start_idx = len(sft_data) - len(sft_train)
        ppo_train = ppo_data.select(range(ppo_train_start_idx, ppo_train_start_idx + int(len(sft_data) * self.data_config.apps_ppo_data_ratio)))
        
        ppo_eval = dataset["test"].map(format_apps_ppo, remove_columns=dataset["test"].column_names)

        return {
            "sft_train": sft_train,
            # "rm_train": None, # No RM for APPS
            # "rm_eval": None,
            "ppo_train": ppo_train,
            "ppo_eval": ppo_eval,
        }


def get_dataloaders(
    data_config: DataConfig,
    sft_config: SFTConfig,
    rm_config: RMConfig,
    ppo_config: PPOConfig,
    tokenizer: AutoTokenizer,
    task_name: str,
) -> Dict[str, DataLoader]:
    """
    Main function to get all data loaders for the different stages.
    """
    dataset_manager = RLHFDataset(data_config, tokenizer)
    
    if task_name == "tldr":
        datasets = dataset_manager.load_and_prepare_tldr()
        sft_max_seq_len = sft_config.max_seq_length
        rm_max_seq_len = rm_config.max_seq_length
        ppo_max_prompt_len = ppo_config.max_prompt_length
        ppo_max_response_len = ppo_config.max_response_length
    elif task_name == "hh_rlhf":
        datasets = dataset_manager.load_and_prepare_hh_rlhf()
        sft_max_seq_len = sft_config.max_seq_length
        rm_max_seq_len = rm_config.max_seq_length
        ppo_max_prompt_len = ppo_config.max_prompt_length
        ppo_max_response_len = ppo_config.max_response_length
    elif task_name == "webgpt":
        datasets = dataset_manager.load_and_prepare_webgpt()
        sft_max_seq_len = sft_config.max_seq_length
        rm_max_seq_len = rm_config.max_seq_length
        ppo_max_prompt_len = ppo_config.max_prompt_length
        ppo_max_response_len = ppo_config.max_response_length
    elif task_name == "apps":
        datasets = dataset_manager.load_and_prepare_apps()
        sft_max_seq_len = sft_config.max_seq_length
        rm_max_seq_len = rm_config.max_seq_length # Not used, but keep for type consistency
        ppo_max_prompt_len = ppo_config.code_max_prompt_length # Use code specific lengths
        ppo_max_response_len = ppo_config.code_max_response_length
    else:
        raise ValueError(f"Unknown task name: {task_name}")

    sft_dataloader = DataLoader(
        datasets["sft_train"],
        shuffle=True,
        batch_size=sft_config.batch_size if task_name != "apps" else sft_config.code_batch_size,
        collate_fn=DataCollatorForSFT(tokenizer, sft_max_seq_len),
    )

    rm_train_dataloader = None
    rm_eval_dataloader = None
    if "rm_train" in datasets and datasets["rm_train"] is not None:
        rm_train_dataloader = DataLoader(
            datasets["rm_train"],
            shuffle=True,
            batch_size=rm_config.batch_size,
            collate_fn=DataCollatorForRM(tokenizer, rm_max_seq_len),
        )
        rm_eval_dataloader = DataLoader(
            datasets["rm_eval"],
            shuffle=False,
            batch_size=rm_config.batch_size,
            collate_fn=DataCollatorForRM(tokenizer, rm_max_seq_len),
        )

    ppo_dataloader = DataLoader(
        datasets["ppo_train"],
        shuffle=True,
        batch_size=ppo_config.batch_size if task_name != "apps" else ppo_config.code_batch_size,
        collate_fn=DataCollatorForPPO(tokenizer, ppo_max_prompt_len, ppo_max_response_len),
    )
    ppo_eval_dataloader = DataLoader(
        datasets["ppo_eval"],
        shuffle=False,
        batch_size=ppo_config.batch_size if task_name != "apps" else ppo_config.code_batch_size,
        collate_fn=DataCollatorForPPO(tokenizer, ppo_max_prompt_len, ppo_max_response_len),
    )

    return {
        "sft_train": sft_dataloader,
        "rm_train": rm_train_dataloader,
        "rm_eval": rm_eval_dataloader,
        "ppo_train": ppo_dataloader,
        "ppo_eval": ppo_eval_dataloader,
    }


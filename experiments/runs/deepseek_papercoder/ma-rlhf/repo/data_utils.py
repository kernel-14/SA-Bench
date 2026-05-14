# data_utils.py
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Prepares and tokenizes datasets for SFT, RM, and PPO stages.
    Implements all dataset-specific formatting and splitting.
    """

    def __init__(self, config: dict, tokenizer: AutoTokenizer):
        """
        Args:
            config: experiment-specific configuration dictionary (from config.yaml).
                    Must include 'dataset_name', 'max_prompt_length', 'max_response_length'.
                    Optionally includes 'data_splits' (dict with 'sft_ratio', 'rm_ratio', 'ppo_ratio').
            tokenizer: HuggingFace tokenizer used for all tokenization.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.dataset_name = config.get("dataset_name")
        if not self.dataset_name:
            raise ValueError("config must contain 'dataset_name'")

        # Default split ratios (can be overridden per experiment)
        splits = config.get("data_splits", {})
        self.sft_ratio = splits.get("sft_ratio", 0.2)
        self.rm_ratio = splits.get("rm_ratio", 0.4)
        self.ppo_ratio = splits.get("ppo_ratio", 0.4)

        # Ensure tokenizer has proper settings
        self.tokenizer.padding_side = "right"

    def _load_raw_dataset(self, dataset_name: str) -> DatasetDict:
        """Load the raw dataset from HuggingFace Hub or local path."""
        try:
            if dataset_name == "tldr":
                # OpenAI summarize_from_feedback
                ds = load_dataset("openai/summarize_from_feedback", "comparisons")
                # The dataset has 'train' and 'validation' splits; we only use 'train' for training splits,
                # and keep 'validation' for final RM evaluation.
                return ds
            elif dataset_name == "hhrlhf":
                # Anthropic HH-RLHF (helpful-base, single-turn)
                ds = load_dataset("Anthropic/hh-rlhf", "helpful-base")
                # Contains 'train' and 'test'; we rename 'test' to 'validation' internally.
                return DatasetDict({"train": ds["train"], "validation": ds["test"]})
            elif dataset_name == "webgpt":
                ds = load_dataset("openai/webgpt_comparisons")
                # No default split; we'll split later
                return ds
            elif dataset_name == "apps":
                ds = load_dataset("codeparrot/apps", "all")
                return ds
            else:
                raise ValueError(f"Unknown dataset name: {dataset_name}")
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            raise

    def _split_train_set(
        self, dataset: Dataset, ratios: List[float], seed: int = 42
    ) -> Dict[str, Dataset]:
        """
        Split a HuggingFace Dataset into multiple parts according to given ratios.
        Returns a dict with keys 'sft', 'rm', 'ppo' containing the respective splits.
        """
        total = len(dataset)
        sft_size = int(total * ratios[0])
        rm_size = int(total * ratios[1])
        # The rest goes to ppo
        # Use train_test_split to ensure deterministic and reproducible splits.
        split1 = dataset.train_test_split(test_size=rm_size + (total - sft_size - rm_size), seed=seed)
        sft_ds = split1["train"]
        remaining_ds = split1["test"]
        # Further split remaining into rm and ppo
        remaining_len = len(remaining_ds)
        rm_len = rm_size
        ppo_len = remaining_len - rm_len
        split2 = remaining_ds.train_test_split(test_size=ppo_len, seed=seed)
        rm_ds = split2["train"]
        ppo_ds = split2["test"]
        return {"sft": sft_ds, "rm": rm_ds, "ppo": ppo_ds}

    def _parse_hhrlhf_sample(self, text: str) -> Tuple[str, str]:
        """
        Extract prompt (everything before the last assistant turn) and final assistant response.
        For single-turn HH-RLHF, the text is a dialogue ending with Assistant: ...
        """
        # The text is formatted with lines like "Human: ... Assistant: ..."
        # Find the last occurrence of "Assistant:"
        split_marker = "Assistant:"
        last_idx = text.rfind(split_marker)
        if last_idx == -1:
            # Fallback: assume the whole text is the assistant response (should not happen)
            return "", text
        prompt = text[:last_idx].strip()
        # Remove the trailing "Assistant:" portion; the response starts after that and may include a newline.
        response = text[last_idx + len(split_marker):].strip()
        return prompt, response

    def _format_and_tokenize_sft(self, examples: Dict[str, List], mode: str) -> Dict[str, List]:
        """
        Tokenize for SFT. mode can be 'tldr' (summarization) or 'chat' (dialogue/QA).
        The input examples should contain at least 'prompt' and 'response' keys.
        For code generation, mode='code'.
        """
        batch_texts = []
        prompts = examples.get("prompt")
        responses = examples.get("response")
        if prompts is None or responses is None:
            raise KeyError("Batch must contain 'prompt' and 'response' keys.")

        for prompt, resp in zip(prompts, responses):
            if mode == "tldr":
                text = f"{prompt}\n\nTL;DR:{resp}"
            elif mode == "chat":
                text = f"Human: {prompt}\n\nAssistant: {resp}"
            elif mode == "code":
                # Format as instruction
                text = f"Write a Python program to solve the following problem:\n\n{prompt}\n\nSolution:\n{resp}"
            else:
                raise ValueError(f"Unknown mode {mode}")
            batch_texts.append(text)

        # Tokenize with left truncation for prompts and right truncation for responses
        # Since we are concatenating, we need to truncate appropriately.
        # We'll tokenize the whole text and ensure that the prompt part is not cut too much.
        # However, for simplicity, we can tokenize with max length = max_prompt_length + max_response_length,
        # and set labels to ignore the first max_prompt_length tokens.
        max_total = self.config["max_prompt_length"] + self.config["max_response_length"]
        encodings = self.tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_total,
            padding=False,   # will pad later in trainer
            return_tensors=None,
        )
        # Build labels: ignore prompt part
        # Since truncation may cut from left, we need to identify prompt tokens.
        # A robust approach: we tokenize prompt separately to find its length,
        # then after concatenation, set labels to -100 for the first prompt_length tokens.
        labels_list = []
        for i, text in enumerate(batch_texts):
            # We need to know where prompt ends; we can tokenize prompt+response with the template.
            # But it's easier: tokenize the full text, then tokenize just the prompt,
            # count tokens.
            if mode in ("tldr", "chat"):
                prompt_text = prompts[i]
                if mode == "tldr":
                    prompt_full = f"{prompt_text}\n\nTL;DR:"
                elif mode == "chat":
                    prompt_full = f"Human: {prompt_text}\n\nAssistant:"
                enc_prompt = self.tokenizer(prompt_full, add_special_tokens=True)["input_ids"]
                prompt_len = len(enc_prompt)
            elif mode == "code":
                prompt_full = f"Write a Python program to solve the following problem:\n\n{prompts[i]}\n\nSolution:\n"
                enc_prompt = self.tokenizer(prompt_full, add_special_tokens=True)["input_ids"]
                prompt_len = len(enc_prompt)
            else:
                prompt_len = 0
            # Truncation may have cut some tokens from the beginning; we need to adjust prompt_len.
            # The tokenized ids will have length <= max_total. We can find the actual prompt part in the truncated ids.
            # Better: use tokenizer with return_offsets_mapping, but not all tokenizers support it.
            # Simpler: we bypass exact label alignment and just set labels to -100 for the first prompt_len tokens,
            # assuming truncation only cuts from left and the prompt is not extremely long beyond max_prompt_length.
            # This is acceptable.
            labels = [-100] * prompt_len + encodings["input_ids"][i][prompt_len:]
            # In case truncation removed some tokens, the total length might be less than prompt_len + response_len.
            # We'll adjust: if prompt_len > len(encodings["input_ids"][i]):
            # then the entire sequence is part of the prompt (should not happen with proper truncation).
            if prompt_len >= len(encodings["input_ids"][i]):
                # This would mean the prompt alone fills or exceeds the max_total; fallback: label everything as prompt
                labels = [-100] * len(encodings["input_ids"][i])
            else:
                # Ensure labels length matches input_ids length
                labels = labels[: len(encodings["input_ids"][i])]
                # Pad labels to match if shorter (should not happen)
                while len(labels) < len(encodings["input_ids"][i]):
                    labels.append(-100)
            labels_list.append(labels)

        # Build attention_mask
        attention_mask = [
            [1] * len(ids) for ids in encodings["input_ids"]
        ]

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": attention_mask,
            "labels": labels_list,
        }

    def _format_and_tokenize_rm(self, examples: Dict[str, List], mode: str) -> Dict[str, List]:
        """
        Tokenize for reward model: returns 'input_ids_chosen', 'attention_mask_chosen',
        'input_ids_rejected', 'attention_mask_rejected'.
        The prompt and response are concatenated as a single sequence.
        """
        prompts = examples["prompt"]
        chosen_responses = examples["chosen"]
        rejected_responses = examples["rejected"]

        chosen_texts = []
        rejected_texts = []
        for prompt, chosen, rejected in zip(prompts, chosen_responses, rejected_responses):
            if mode == "tldr":
                chosen_texts.append(f"{prompt}\n\nTL;DR:{chosen}")
                rejected_texts.append(f"{prompt}\n\nTL;DR:{rejected}")
            elif mode == "chat":
                chosen_texts.append(f"Human: {prompt}\n\nAssistant: {chosen}")
                rejected_texts.append(f"Human: {prompt}\n\nAssistant: {rejected}")
            else:
                raise ValueError(f"Unknown mode {mode}")

        max_total = self.config["max_prompt_length"] + self.config["max_response_length"]
        chosen_enc = self.tokenizer(chosen_texts, truncation=True, max_length=max_total, padding=False)
        rejected_enc = self.tokenizer(rejected_texts, truncation=True, max_length=max_total, padding=False)

        return {
            "input_ids_chosen": chosen_enc["input_ids"],
            "attention_mask_chosen": [[1]*len(ids) for ids in chosen_enc["input_ids"]],
            "input_ids_rejected": rejected_enc["input_ids"],
            "attention_mask_rejected": [[1]*len(ids) for ids in rejected_enc["input_ids"]],
        }

    def _format_and_tokenize_ppo(self, examples: Dict[str, List], mode: str) -> Dict[str, List]:
        """Tokenize only prompt for PPO (online rollout)."""
        prompts = examples["prompt"]
        if mode == "tldr":
            texts = [f"{p}\n\nTL;DR:" for p in prompts]
        elif mode == "chat":
            texts = [f"Human: {p}\n\nAssistant:" for p in prompts]
        elif mode == "code":
            texts = [f"Write a Python program to solve the following problem:\n\n{p}\n\nSolution:\n" for p in prompts]
        else:
            raise ValueError(f"Unknown mode {mode}")

        # Only left truncation to max_prompt_length
        max_len = self.config["max_prompt_length"]
        encodings = self.tokenizer(texts, truncation=True, max_length=max_len, padding=False)
        # Save prompt length for later slicing during generation
        prompt_lens = [len(ids) for ids in encodings["input_ids"]]
        # Add prompt_length field
        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": [[1]*l for l in prompt_lens],
            "prompt_length": prompt_lens,
        }

    # --------------------------------------------------------------
    # Public methods
    # --------------------------------------------------------------

    def load_sft_data(self, dataset_name: Optional[str] = None, split: str = "train") -> Dataset:
        """
        Returns a tokenized SFT dataset for the specified split.
        If split='train', returns the training portion (SFT split).
        If split='validation', returns the original validation set (if exists) without internal splits.
        """
        if dataset_name is None:
            dataset_name = self.dataset_name

        if dataset_name == "apps":
            return self._load_apps_sft()

        raw = self._load_raw_dataset(dataset_name)
        if split == "validation":
            if "validation" in raw:
                # Use original validation split (e.g., for TL;DR, HH-RLHF)
                val_ds = raw["validation"]
            else:
                # For WebGPT, we'll produce a validation set from the whole dataset later.
                # But calling load_sft_data with validation only makes sense for final eval,
                # where we don't use the internal splits. So we won't spit further.
                # For now, we'll handle this in evaluate.py by loading raw.
                logger.warning("Validation split requested but dataset does not have a predefined validation split. Returning empty.")
                return Dataset.from_dict({})
            # Format and tokenize
            mode = "tldr" if dataset_name == "tldr" else "chat"
            # Preprocess: extract prompt and response
            val_samples = {"prompt": [], "response": []}
            for sample in val_ds:
                if dataset_name == "tldr":
                    info = sample.get("info")
                    if info and "post" in info:
                        prompt = info["post"]
                    else:
                        continue
                    # For validation, we don't have chosen/rejected; we just have summaries? Actually validation set
                    # also contains comparisons but we can treat the "chosen" as response for SFT? Not needed for SFT eval.
                    # For consistency, we'll just return prompts? But SFT needs response.
                    # The paper says "We compute RM score on 2k validation instances". They used RM score, not SFT.
                    # So SFT validation is not used for final metrics; we can ignore it.
                elif dataset_name == "hhrlhf":
                    # HH-RLHF validation: each sample has 'chosen' and 'rejected' but for SFT we only need chosen.
                    chosen_text = sample.get("chosen", "")
                    prompt, response = self._parse_hhrlhf_sample(chosen_text)
                    val_samples["prompt"].append(prompt)
                    val_samples["response"].append(response)
            val_ds_proc = Dataset.from_dict(val_samples)
            return self._format_and_tokenize_sft_in_ds(val_ds_proc, mode)

        # For split='train', derive from raw train set
        train_raw = raw["train"]
        # Perform internal split
        splits = self._split_train_set(train_raw, [self.sft_ratio, self.rm_ratio, self.ppo_ratio])
        sft_raw = splits["sft"]

        mode = "tldr" if dataset_name == "tldr" else "chat"
        # Prepare prompts and responses list
        samples = {"prompt": [], "response": []}
        for sample in sft_raw:
            if dataset_name == "tldr":
                info = sample.get("info")
                summaries = sample.get("summaries")
                choice = sample.get("choice")
                if not (info and summaries and choice is not None):
                    continue
                prompt = info.get("post", "")
                response = summaries[choice]["text"]  # chosen
                samples["prompt"].append(prompt)
                samples["response"].append(response)
            elif dataset_name == "hhrlhf":
                chosen_text = sample.get("chosen", "")
                prompt, response = self._parse_hhrlhf_sample(chosen_text)
                samples["prompt"].append(prompt)
                samples["response"].append(response)
            elif dataset_name == "webgpt":
                question = sample.get("question")
                answer_0 = sample.get("answer_0")
                answer_1 = sample.get("answer_1")
                score_0 = sample.get("score_0")
                score_1 = sample.get("score_1")
                if question is None or answer_0 is None or answer_1 is None:
                    continue
                # Determine chosen by score; if equal, pick first
                if score_0 >= score_1:
                    chosen = answer_0
                else:
                    chosen = answer_1
                samples["prompt"].append(question)
                samples["response"].append(chosen)

        sft_dataset = Dataset.from_dict(samples)
        tokenized = sft_dataset.map(
            lambda x: self._format_and_tokenize_sft(x, mode=mode),
            batched=True,
            remove_columns=sft_dataset.column_names,
        )
        return tokenized

    def _load_apps_sft(self) -> Dataset:
        """Special SFT loading for APPS."""
        raw = self._load_raw_dataset("apps")["train"]
        samples = {"prompt": [], "response": []}
        for sample in raw:
            problem = sample.get("problem", "")
            solutions_str = sample.get("solutions", "[]")
            # Parse solutions JSON
            try:
                solutions = json.loads(solutions_str)
            except (json.JSONDecodeError, TypeError):
                solutions = []
            if solutions:
                # Use first solution as target
                sol = solutions[0]
            else:
                continue  # skip empty
            samples["prompt"].append(problem)
            samples["response"].append(sol)
        if len(samples["prompt"]) == 0:
            raise ValueError("No valid APPS training samples found.")
        ds = Dataset.from_dict(samples)
        tokenized = ds.map(
            lambda x: self._format_and_tokenize_sft(x, mode="code"),
            batched=True,
            remove_columns=ds.column_names,
        )
        return tokenized

    def load_rm_data(self, dataset_name: Optional[str] = None, split: str = "train") -> Dataset:
        """Load preference pairs for reward model training."""
        if dataset_name is None:
            dataset_name = self.dataset_name
        if dataset_name == "apps":
            raise RuntimeError("APPS dataset does not have RM stage.")

        raw = self._load_raw_dataset(dataset_name)
        if split == "validation":
            # For RM validation, we need a held-out set of preference pairs.
            # If dataset has predefined validation, use it.
            if "validation" in raw:
                val_raw = raw["validation"]
                # Extract pairs
                samples = {"prompt": [], "chosen": [], "rejected": []}
                for sample in val_raw:
                    if dataset_name == "tldr":
                        info = sample.get("info")
                        summaries = sample.get("summaries")
                        choice = sample.get("choice")
                        if not info or not summaries or choice is None:
                            continue
                        prompt = info.get("post", "")
                        chosen = summaries[choice]["text"]
                        rejected = summaries[1 - choice]["text"]
                        samples["prompt"].append(prompt)
                        samples["chosen"].append(chosen)
                        samples["rejected"].append(rejected)
                    elif dataset_name == "hhrlhf":
                        chosen_text = sample.get("chosen", "")
                        rejected_text = sample.get("rejected", "")
                        prompt_chosen, chosen_resp = self._parse_hhrlhf_sample(chosen_text)
                        # Use prompt from chosen (both should share same prompt)
                        prompt = prompt_chosen
                        _, rejected_resp = self._parse_hhrlhf_sample(rejected_text)
                        samples["prompt"].append(prompt)
                        samples["chosen"].append(chosen_resp)
                        samples["rejected"].append(rejected_resp)
                val_ds = Dataset.from_dict(samples)
                mode = "tldr" if dataset_name == "tldr" else "chat"
                tokenized_val = val_ds.map(
                    lambda x: self._format_and_tokenize_rm(x, mode=mode),
                    batched=True,
                    remove_columns=val_ds.column_names,
                )
                return tokenized_val
            else:
                # For WebGPT, we need to create validation from training data (5% of RM set)
                logger.info("Creating RM validation split from training data (5% of RM portion).")
                train_raw = raw["train"]
                # We'll first split the training set into SFT/RM/PPO, then take 5% of RM as val.
                splits = self._split_train_set(train_raw, [self.sft_ratio, self.rm_ratio, self.ppo_ratio])
                rm_raw = splits["rm"]
                # Split rm_raw into train and validation (5% validation)
                rm_split = rm_raw.train_test_split(test_size=0.05, seed=42)
                rm_train = rm_split["train"]
                rm_val = rm_split["test"]
                # For val, process as above; but we'll cache the split for later RM train usage.
                # We'll rely on the caller to use split='train' for RM training data.
                # We'll process val here.
                samples = self._extract_rm_samples(rm_val, dataset_name)
                mode = "tldr" if dataset_name == "tldr" else "chat"
                val_ds = Dataset.from_dict(samples)
                tokenized_val = val_ds.map(
                    lambda x: self._format_and_tokenize_rm(x, mode=mode),
                    batched=True,
                    remove_columns=val_ds.column_names,
                )
                return tokenized_val

        # For split='train', process RM portion of training data
        train_raw = raw["train"]
        splits = self._split_train_set(train_raw, [self.sft_ratio, self.rm_ratio, self.ppo_ratio])
        rm_raw = splits["rm"]
        # If validation split will be extracted later, we should reserve 5% here for WebGPT.
        # To avoid data leakage, when loading RM training data, we should exclude the validation set.
        # We'll handle it: for WebGPT, we'll do an explicit 95/5 split of rm_raw and return the 95% portion.
        if dataset_name == "webgpt":
            rm_split = rm_raw.train_test_split(test_size=0.05, seed=42)
            rm_train = rm_split["train"]
        else:
            rm_train = rm_raw

        samples = self._extract_rm_samples(rm_train, dataset_name)
        mode = "tldr" if dataset_name == "tldr" else "chat"
        rm_ds = Dataset.from_dict(samples)
        tokenized = rm_ds.map(
            lambda x: self._format_and_tokenize_rm(x, mode=mode),
            batched=True,
            remove_columns=rm_ds.column_names,
        )
        return tokenized

    def _extract_rm_samples(self, dataset, dataset_name):
        """Helper to build RM samples dict from dataset."""
        samples = {"prompt": [], "chosen": [], "rejected": []}
        for sample in dataset:
            if dataset_name == "tldr":
                info = sample.get("info")
                summaries = sample.get("summaries")
                choice = sample.get("choice")
                if not info or not summaries or choice is None:
                    continue
                prompt = info.get("post", "")
                chosen = summaries[choice]["text"]
                rejected = summaries[1 - choice]["text"]
                samples["prompt"].append(prompt)
                samples["chosen"].append(chosen)
                samples["rejected"].append(rejected)
            elif dataset_name == "hhrlhf":
                chosen_text = sample.get("chosen", "")
                rejected_text = sample.get("rejected", "")
                prompt_chosen, chosen_resp = self._parse_hhrlhf_sample(chosen_text)
                prompt_rejected, rejected_resp = self._parse_hhrlhf_sample(rejected_text)
                # Use prompt from chosen; both should be identical.
                samples["prompt"].append(prompt_chosen)
                samples["chosen"].append(chosen_resp)
                samples["rejected"].append(rejected_resp)
            elif dataset_name == "webgpt":
                question = sample.get("question")
                answer_0 = sample.get("answer_0")
                answer_1 = sample.get("answer_1")
                score_0 = sample.get("score_0")
                score_1 = sample.get("score_1")
                if None in (question, answer_0, answer_1, score_0, score_1):
                    continue
                if score_0 >= score_1:
                    chosen = answer_0
                    rejected = answer_1
                else:
                    chosen = answer_1
                    rejected = answer_0
                samples["prompt"].append(question)
                samples["chosen"].append(chosen)
                samples["rejected"].append(rejected)
        return samples

    def load_ppo_data(self, dataset_name: Optional[str] = None, split: str = "train") -> Dataset:
        """Returns tokenized prompts for PPO training."""
        if dataset_name is None:
            dataset_name = self.dataset_name
        if dataset_name == "apps":
            return self._load_apps_ppo_prompts()
        raw = self._load_raw_dataset(dataset_name)
        train_raw = raw["train"]
        splits = self._split_train_set(train_raw, [self.sft_ratio, self.rm_ratio, self.ppo_ratio])
        ppo_raw = splits["ppo"]

        mode = "tldr" if dataset_name == "tldr" else "chat"
        prompts = []
        for sample in ppo_raw:
            if dataset_name == "tldr":
                info = sample.get("info")
                if info and "post" in info:
                    prompts.append(info["post"])
            elif dataset_name == "hhrlhf":
                chosen_text = sample.get("chosen", "")
                prompt, _ = self._parse_hhrlhf_sample(chosen_text)
                prompts.append(prompt)
            elif dataset_name == "webgpt":
                question = sample.get("question")
                if question:
                    prompts.append(question)
        if not prompts:
            raise ValueError("No prompts found for PPO.")
        ppo_ds = Dataset.from_dict({"prompt": prompts})
        tokenized = ppo_ds.map(
            lambda x: self._format_and_tokenize_ppo(x, mode=mode),
            batched=True,
            remove_columns=ppo_ds.column_names,
        )
        return tokenized

    def _load_apps_ppo_prompts(self) -> Dataset:
        """All APPS training prompts (entire train set)."""
        raw = self._load_raw_dataset("apps")["train"]
        prompts = [sample["problem"] for sample in raw]
        ds = Dataset.from_dict({"prompt": prompts})
        tokenized = ds.map(
            lambda x: self._format_and_tokenize_ppo(x, mode="code"),
            batched=True,
            remove_columns=ds.column_names,
        )
        return tokenized

    def prepare_apps_data(self) -> Dict[str, Dataset]:
        """
        For APPS, returns a dict with:
          - 'sft_train': tokenized SFT dataset (ground truth solutions)
          - 'ppo_prompts': tokenized PPO prompts (same train set)
          - 'eval_test': the raw test dataset (for pass@k evaluation)
        """
        raw = self._load_raw_dataset("apps")
        train_raw = raw["train"]
        test_raw = raw["test"]

        # SFT train
        sft_train = self._load_apps_sft()  # already tokenized

        # PPO prompts
        ppo_prompts = self._load_apps_ppo_prompts()

        # Eval test set: keep original columns (problem, test_list, etc.)
        # We'll store it as a regular Dataset; evaluator will use it directly.
        return {
            "sft_train": sft_train,
            "ppo_prompts": ppo_prompts,
            "eval_test": test_raw,
        }

import os
import random
import torch
import datasets
from torch.utils.data import DataLoader as PyTorchDataLoader
from transformers import DataCollatorWithPadding
from typing import Dict, Any, List, Tuple, Union, Optional
from loguru import logger
from omegaconf import DictConfig

# To avoid circular import with config.py, we define Config as DictConfig directly.
# In a real project with a 'config' module, this would typically be 'from config import Config'
Config = DictConfig

# Assuming TokenizerWrapper is imported from utils.py
# To avoid circular import when testing this file independently, we define a mock for type hinting.
# In the actual project, this would be: `from utils import TokenizerWrapper`
class TokenizerWrapper:
    """Mock TokenizerWrapper for type hinting and to satisfy DataLoader dependencies."""
    tokenizer: Any # AutoTokenizer instance

    def __init__(self, model_name: str):
        # Placeholder for actual tokenizer initialization
        pass

    def encode(self, text: Union[str, List[str]], add_special_tokens: bool = True, max_length: Optional[int] = None, truncation: bool = True, padding: Union[str, bool] = 'max_length', return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Mock method should not be called directly.")

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> Union[str, List[str]]:
        raise NotImplementedError("Mock method should not be called directly.")


class DataLoader:
    """
    Handles loading, preprocessing, and formatting datasets for Supervised Fine-Tuning (SFT),
    Reward Model (RM) training, Proximal Policy Optimization (PPO), and evaluation stages
    of the MA-RLHF pipeline.
    """

    def __init__(self, config: Config, tokenizer_wrapper: TokenizerWrapper):
        """
        Initializes the DataLoader instance.

        Args:
            config: A DictConfig object containing the global and stage-specific configurations.
            tokenizer_wrapper: An instance of TokenizerWrapper for tokenization operations.
        """
        self.config: Config = config
        self.tokenizer_wrapper: TokenizerWrapper = tokenizer_wrapper
        self.datasets: Dict[Tuple[str, str], datasets.Dataset] = {}
        logger.info("DataLoader initialized.")

    def _load_dataset(self, task_name: str, split: str) -> datasets.Dataset:
        """
        Internal helper method to load a specific dataset split for a given task.
        Handles WebGPT's special 95/5 train/validation split as described in the paper.

        Args:
            task_name: The name of the task (e.g., 'tldr_summarization', 'webgpt_comparison').
            split: The dataset split to load ('train' for training data, 'eval' for evaluation data).

        Returns:
            A datasets.Dataset object.

        Raises:
            ValueError: If the task_name is not recognized or split is invalid.
            FileNotFoundError: If the specified data file does not exist.
        """
        cache_key = (task_name, split)
        if cache_key in self.datasets:
            logger.debug(f"Returning cached dataset for {task_name}, split {split}.")
            return self.datasets[cache_key]

        task_data_cfg = self.config.data_configs.get(task_name)
        if not task_data_cfg:
            raise ValueError(f"Unknown task_name: '{task_name}' in data_configs.")

        base_data_path = self.config.data_configs.base_data_path

        if task_name == "webgpt_comparison":
            # WebGPT special handling: "We split 5% instances for validation, as no separate validation set is provided."
            train_split_cache_key = ("webgpt_comparison", "train_split_95")
            eval_split_cache_key = ("webgpt_comparison", "eval_split_5")

            if train_split_cache_key not in self.datasets or eval_split_cache_key not in self.datasets:
                full_train_file_path = os.path.join(base_data_path, task_data_cfg.train_file)
                logger.info(f"Loading full WebGPT train data from: {full_train_file_path} for 95/5 split.")
                if not os.path.exists(full_train_file_path):
                     raise FileNotFoundError(f"WebGPT train file not found: {full_train_file_path}")
                
                full_dataset = datasets.load_dataset('json', data_files={'data': full_train_file_path})['data']
                
                # Perform the 95/5 split
                split_datasets = full_dataset.train_test_split(
                    test_size=0.05, seed=self.config.global.seed, shuffle=True
                )
                self.datasets[train_split_cache_key] = split_datasets['train']
                self.datasets[eval_split_cache_key] = split_datasets['test']
                logger.info(f"WebGPT dataset split: Train={len(split_datasets['train'])}, Eval={len(split_datasets['test'])}")
            
            if split == 'train':
                return self.datasets[train_split_cache_key]
            elif split == 'eval':
                return self.datasets[eval_split_cache_key]
            else:
                raise ValueError(f"Invalid split '{split}' for WebGPT task. Expected 'train' or 'eval'.")
        
        # For other datasets, load directly based on 'train_file' or 'eval_file' keys
        file_key = f"{'train' if split == 'train' else 'eval'}_file"
        file_path = os.path.join(base_data_path, task_data_cfg[file_key])

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found for {task_name} ({split}): {file_path}")

        logger.info(f"Loading dataset from: {file_path}")
        loaded_dataset = datasets.load_dataset('json', data_files={'data': file_path})['data']
        self.datasets[cache_key] = loaded_dataset
        return loaded_dataset

    def _get_prompt_text(self, example: Dict[str, Any], task_name: str) -> str:
        """
        Constructs the prompt text from an example based on the task and template.

        Args:
            example: A dictionary containing raw data fields for a single example.
            task_name: The name of the task to determine the formatting logic.

        Returns:
            A string representing the formatted prompt.

        Raises:
            ValueError: If the task_name is not recognized for prompt construction.
        """
        prompt_template = self.config.data_configs[task_name].prompt_template
        
        # Based on config.yaml prompt_template examples
        if task_name == 'tldr_summarization':
            return prompt_template.format(post=example['post'])
        elif task_name == 'hh_rlhf':
            return prompt_template.format(query=example['query'])
        elif task_name == 'webgpt_comparison':
            return prompt_template.format(question=example['question'])
        elif task_name == 'apps_code_gen':
            return prompt_template.format(problem=example['problem'])
        else:
            raise ValueError(f"Unknown task_name '{task_name}' for prompt construction.")

    def _format_sft_example(self, example: Dict[str, Any], task_name: str) -> Dict[str, torch.Tensor]:
        """
        Formats a raw data example into input_ids, labels, and attention_mask for SFT.
        As per Appendix B.2, for SFT, we concatenate prompt and chosen response.

        Args:
            example: A dictionary containing raw data fields for a single example.
            task_name: The name of the task to determine the formatting logic.

        Returns:
            A dictionary with 'input_ids', 'attention_mask', and 'labels' as torch.Tensor.
        """
        # Determine the response text based on task
        response_text: str
        if task_name == 'tldr_summarization':
            response_text = example['summary']
        elif task_name == 'hh_rlhf':
            response_text = example['chosen_response']
        elif task_name == 'webgpt_comparison':
            response_text = example['response']
        elif task_name == 'apps_code_gen':
            response_text = example['code']
        else:
            raise ValueError(f"Unknown task_name '{task_name}' for SFT response extraction.")

        prompt_text = self._get_prompt_text(example, task_name)
        full_text = prompt_text + response_text # Concatenate prompt and response

        encoded = self.tokenizer_wrapper.encode(
            full_text,
            max_length=self.config.sft_config.max_seq_length,
            truncation=True,
            padding='do_not_pad', # Padding will be handled by collate_fn
            return_tensors="pt"
        )
        
        input_ids = encoded['input_ids'].squeeze(0) # Remove batch dimension for single example
        attention_mask = encoded['attention_mask'].squeeze(0)

        # Labels are generally the same as input_ids for causal LM fine-tuning on full sequence.
        # DataCollatorWithPadding will pad labels with tokenizer.pad_token_id, then _sft_collate_fn
        # will convert these to -100 for loss calculation.
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    def load_sft_data(self, task_name: str) -> PyTorchDataLoader:
        """
        Loads, filters, formats, and returns a DataLoader for SFT training.

        Args:
            task_name: The name of the task (e.g., 'tldr_summarization').

        Returns:
            A torch.utils.data.DataLoader object for SFT. Returns None if 0 samples are selected.
        """
        full_dataset = self._load_dataset(task_name, 'train')
        sft_ratio = self.config.data_configs.sft_data_ratio
        
        # Select a random subset based on ratio
        num_samples = int(len(full_dataset) * sft_ratio)
        if num_samples == 0:
            logger.warning(f"SFT data ratio ({sft_ratio}) resulted in 0 samples for task '{task_name}'. Skipping SFT data loading.")
            return None
        
        # Use global seed for reproducibility
        current_random_state = random.getstate()
        random.seed(self.config.global.seed)
        subset_indices = random.sample(range(len(full_dataset)), num_samples)
        random.setstate(current_random_state) # Restore random state
        
        sft_dataset_subset = full_dataset.select(subset_indices)
        logger.info(f"Loaded {len(sft_dataset_subset)} samples for SFT training for task '{task_name}'.")

        # Apply formatting using .map()
        formatted_dataset = sft_dataset_subset.map(
            lambda example: self._format_sft_example(example, task_name),
            batched=False,
            # Remove original columns to keep only the processed tensor columns
            remove_columns=full_dataset.column_names, 
            desc=f"Formatting SFT data for {task_name}"
        )

        batch_size = self.config.sft_config.batch_size
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer_wrapper.tokenizer,
            padding="longest",
            return_tensors="pt",
            # label_pad_token_id is typically used when labels are also token IDs
            # and you want to explicitly pad them differently from input_ids.
            # Here, we set -100 for labels' padding within our custom collate_fn.
            label_pad_token_id=self.tokenizer_wrapper.tokenizer.pad_token_id
        )

        def _sft_collate_fn(batch_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
            """
            Custom collate_fn for SFT to handle padding and set labels' padding tokens to -100.
            """
            collated_batch = data_collator(batch_list)
            # Set labels' padding tokens to -100 for ignore_index in CrossEntropyLoss
            collated_batch['labels'][collated_batch['labels'] == self.tokenizer_wrapper.tokenizer.pad_token_id] = -100
            return collated_batch
        
        return PyTorchDataLoader(
            formatted_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=_sft_collate_fn,
            drop_last=True # Ensure full batches
        )

    def _format_rm_example(self, example: Dict[str, Any], task_name: str) -> Dict[str, torch.Tensor]:
        """
        Formats a raw preference example into tokenized prompt, chosen, and rejected responses
        for Reward Model training.

        Args:
            example: A dictionary containing raw data fields for a single example,
                     including 'chosen_response' and 'rejected_response'.
            task_name: The name of the task.

        Returns:
            A dictionary with 'prompt_ids', 'prompt_attention_mask',
            'chosen_ids', 'chosen_attention_mask', 'rejected_ids', 'rejected_attention_mask'
            as torch.Tensor.

        Raises:
            ValueError: If task_name is 'apps_code_gen' as RM is omitted for this task.
        """
        if task_name == 'apps_code_gen':
            raise ValueError("Reward Model stage is omitted for APPS dataset. Cannot format RM example.")

        prompt_text = self._get_prompt_text(example, task_name)
        
        # Tokenize prompt
        prompt_encoded = self.tokenizer_wrapper.encode(
            prompt_text,
            max_length=self.config.ppo_config.max_prompt_length, # Reusing PPO max_prompt_length from config
            truncation=True,
            padding='do_not_pad',
            return_tensors="pt"
        )
        prompt_ids = prompt_encoded['input_ids'].squeeze(0)
        prompt_attention_mask = prompt_encoded['attention_mask'].squeeze(0)

        # Tokenize chosen response
        chosen_encoded = self.tokenizer_wrapper.encode(
            example['chosen_response'],
            max_length=self.config.ppo_config.max_response_length, # Reusing PPO max_response_length from config
            truncation=True,
            padding='do_not_pad',
            return_tensors="pt"
        )
        chosen_ids = chosen_encoded['input_ids'].squeeze(0)
        chosen_attention_mask = chosen_encoded['attention_mask'].squeeze(0)

        # Tokenize rejected response
        rejected_encoded = self.tokenizer_wrapper.encode(
            example['rejected_response'],
            max_length=self.config.ppo_config.max_response_length, # Reusing PPO max_response_length from config
            truncation=True,
            padding='do_not_pad',
            return_tensors="pt"
        )
        rejected_ids = rejected_encoded['input_ids'].squeeze(0)
        rejected_attention_mask = rejected_encoded['attention_mask'].squeeze(0)

        return {
            'prompt_ids': prompt_ids,
            'prompt_attention_mask': prompt_attention_mask,
            'chosen_ids': chosen_ids,
            'chosen_attention_mask': chosen_attention_mask,
            'rejected_ids': rejected_ids,
            'rejected_attention_mask': rejected_attention_mask,
        }

    def load_rm_data(self, task_name: str) -> Optional[PyTorchDataLoader]:
        """
        Loads, filters, formats, and returns a DataLoader for Reward Model training.

        Args:
            task_name: The name of the task.

        Returns:
            A torch.utils.data.DataLoader object for RM, or None if RM is skipped as per config.
        """
        if self.config.rm_config.skip_training:
            logger.info(f"Skipping RM data loading for task '{task_name}' as per config.")
            return None

        full_dataset = self._load_dataset(task_name, 'train')
        rm_ratio = self.config.data_configs.rm_data_ratio

        num_samples = int(len(full_dataset) * rm_ratio)
        if num_samples == 0:
            logger.warning(f"RM data ratio ({rm_ratio}) resulted in 0 samples for task '{task_name}'. Skipping RM data loading.")
            return None
        
        # Use global seed for reproducibility
        current_random_state = random.getstate()
        random.seed(self.config.global.seed)
        subset_indices = random.sample(range(len(full_dataset)), num_samples)
        random.setstate(current_random_state) # Restore random state

        rm_dataset_subset = full_dataset.select(subset_indices)
        logger.info(f"Loaded {len(rm_dataset_subset)} samples for RM training for task '{task_name}'.")
        
        formatted_dataset = rm_dataset_subset.map(
            lambda example: self._format_rm_example(example, task_name),
            batched=False,
            remove_columns=full_dataset.column_names,
            desc=f"Formatting RM data for {task_name}"
        )

        batch_size = self.config.rm_config.batch_size
        
        def _rm_collate_fn(batch_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
            """
            Custom collate_fn for RM data to handle padding for prompt, chosen, and rejected parts.
            """
            collated = {}
            
            # Helper to pad a list of tensor dictionaries
            def _pad_batch(items: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
                return DataCollatorWithPadding(
                    self.tokenizer_wrapper.tokenizer, padding="longest", return_tensors="pt"
                )(items)

            # Group and pad prompt-related tensors
            prompt_parts = [{'input_ids': item['prompt_ids'], 'attention_mask': item['prompt_attention_mask']} for item in batch_list]
            padded_prompt = _pad_batch(prompt_parts)
            collated['prompt_ids'] = padded_prompt['input_ids']
            collated['prompt_attention_mask'] = padded_prompt['attention_mask']

            # Group and pad chosen response-related tensors
            chosen_parts = [{'input_ids': item['chosen_ids'], 'attention_mask': item['chosen_attention_mask']} for item in batch_list]
            padded_chosen = _pad_batch(chosen_parts)
            collated['chosen_ids'] = padded_chosen['input_ids']
            collated['chosen_attention_mask'] = padded_chosen['attention_mask']

            # Group and pad rejected response-related tensors
            rejected_parts = [{'input_ids': item['rejected_ids'], 'attention_mask': item['rejected_attention_mask']} for item in batch_list]
            padded_rejected = _pad_batch(rejected_parts)
            collated['rejected_ids'] = padded_rejected['input_ids']
            collated['rejected_attention_mask'] = padded_rejected['attention_mask']

            return collated

        return PyTorchDataLoader(
            formatted_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=_rm_collate_fn,
            drop_last=True
        )

    def _format_ppo_example(self, example: Dict[str, Any], task_name: str, is_eval_apps: bool = False) -> Union[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Formats a raw prompt example into tokenized prompt_ids and attention_mask for PPO training or evaluation.
        For APPS evaluation (`is_eval_apps=True`), it returns the raw example dict for further processing by CodeExecutor.
        
        Args:
            example: A dictionary containing raw data fields for a single example.
            task_name: The name of the task.
            is_eval_apps: If True, indicates this is for APPS evaluation, and the raw example should be returned.

        Returns:
            A dictionary with 'prompt_ids' and 'attention_mask' as torch.Tensor,
            or the raw example dictionary if is_eval_apps is True.
        """
        if is_eval_apps and task_name == 'apps_code_gen':
            # For APPS evaluation, we need the full original example to pass to CodeExecutor
            # which will extract problem and test_cases.
            return example

        prompt_text = self._get_prompt_text(example, task_name)
        
        encoded = self.tokenizer_wrapper.encode(
            prompt_text,
            max_length=self.config.ppo_config.max_prompt_length,
            truncation=True,
            padding='do_not_pad',
            return_tensors="pt"
        )
        prompt_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        return {
            'prompt_ids': prompt_ids,
            'attention_mask': attention_mask,
            'prompt_text': prompt_text # Include original text for potential debugging/logging
        }

    def load_ppo_data(self, task_name: str) -> PyTorchDataLoader:
        """
        Loads, filters, formats, and returns a DataLoader for PPO training.

        Args:
            task_name: The name of the task.

        Returns:
            A torch.utils.data.DataLoader object for PPO. Returns None if 0 samples are selected.
        """
        full_dataset = self._load_dataset(task_name, 'train')
        
        ppo_ratio: float
        if task_name == 'apps_code_gen':
            # Appendix B.2: For APPS, 80% of data for PPO (20% SFT, 0% RM, 80% PPO)
            ppo_ratio = 0.8 
        else:
            ppo_ratio = self.config.data_configs.ppo_data_ratio # Default 0.4 for others

        num_samples = int(len(full_dataset) * ppo_ratio)
        if num_samples == 0:
            logger.warning(f"PPO data ratio ({ppo_ratio}) resulted in 0 samples for task '{task_name}'. Skipping PPO data loading.")
            return None

        # Use global seed for reproducibility
        current_random_state = random.getstate()
        random.seed(self.config.global.seed)
        subset_indices = random.sample(range(len(full_dataset)), num_samples)
        random.setstate(current_random_state) # Restore random state

        ppo_dataset_subset = full_dataset.select(subset_indices)
        logger.info(f"Loaded {len(ppo_dataset_subset)} samples for PPO training for task '{task_name}'.")

        formatted_dataset = ppo_dataset_subset.map(
            lambda example: self._format_ppo_example(example, task_name),
            batched=False,
            remove_columns=full_dataset.column_names,
            desc=f"Formatting PPO data for {task_name}"
        )
        
        batch_size = self.config.ppo_config.batch_size

        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer_wrapper.tokenizer,
            padding="longest",
            return_tensors="pt"
        )
        
        return PyTorchDataLoader(
            formatted_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=data_collator,
            drop_last=True
        )

    def load_eval_data(self, task_name: str) -> PyTorchDataLoader:
        """
        Loads, samples, formats, and returns a DataLoader for various evaluation scenarios.

        Args:
            task_name: The name of the task.

        Returns:
            A torch.utils.data.DataLoader object for evaluation.
        """
        full_eval_dataset = self._load_dataset(task_name, 'eval')
        eval_dataset_subset = full_eval_dataset

        # Apply sampling based on evaluation type and config.
        # Paper Appendix 4.1: "2k validation instances for the TL;DR and HH-RLHF datasets"
        # "default validation set of the WebGPT dataset"
        # "50 instances that are drawn from the instances used in the RM evaluation" for GPT-4/Human
        # For APPS, "evaluated on the provided 5k test set" (which is the eval split)

        if task_name == 'apps_code_gen':
            logger.info(f"Using full APPS eval set ({len(full_eval_dataset)}) for pass@k evaluation.")
            # For APPS, the formatted example is the raw example dict itself
            formatted_dataset = eval_dataset_subset.map(
                lambda example: self._format_ppo_example(example, task_name, is_eval_apps=True),
                batched=False,
                desc=f"Preparing APPS eval data for {task_name}"
            )
            return PyTorchDataLoader(
                formatted_dataset,
                batch_size=self.config.evaluation_config.eval_batch_size,
                shuffle=False, # Evaluation data should not be shuffled
                collate_fn=lambda batch: batch, # For APPS, return raw list of examples
                drop_last=False
            )
        
        # For other tasks, determine sample size
        num_samples_config = self.config.evaluation_config.rm_eval_samples # Default for RM scores

        # Override for GPT-4/Human evaluations if specific config is there
        # This DataLoader is general, specific eval managers might sample further or use subset of this.
        # For now, we follow the RM evaluation sample size if available, otherwise use full.
        if task_name in ['tldr_summarization', 'hh_rlhf'] and num_samples_config < len(full_eval_dataset):
            # Only sample if the dataset is larger than the configured sample size
            random.seed(self.config.global.seed)
            subset_indices = random.sample(range(len(full_eval_dataset)), num_samples_config)
            eval_dataset_subset = full_eval_dataset.select(subset_indices)
            logger.info(f"Sampled {num_samples_config} instances for RM score evaluation for task '{task_name}'.")
        elif task_name == 'webgpt_comparison':
            # For WebGPT, the _load_dataset already handles providing the 5% validation split,
            # which is then used in full for RM evaluation (as per paper).
            logger.info(f"Using full WebGPT eval set ({len(full_eval_dataset)}) for RM score evaluation.")
        else:
            logger.info(f"No specific sampling rule for eval data found for task '{task_name}'. Using full eval set ({len(full_eval_dataset)}).")
            
        formatted_dataset = eval_dataset_subset.map(
            lambda example: self._format_ppo_example(example, task_name),
            batched=False,
            remove_columns=full_eval_dataset.column_names,
            desc=f"Formatting Eval data for {task_name}"
        )

        batch_size = self.config.evaluation_config.eval_batch_size
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer_wrapper.tokenizer,
            padding="longest",
            return_tensors="pt"
        )

        return PyTorchDataLoader(
            formatted_dataset,
            batch_size=batch_size,
            shuffle=False, # Evaluation data should not be shuffled
            collate_fn=data_collator,
            drop_last=False # Do not drop last batch for evaluation
        )


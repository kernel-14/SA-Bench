import argparse
import os
import yaml
import torch
import numpy as np
from typing import Dict, List, Tuple, Union

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    set_seed,
    TrainingArguments,
)
from datasets import DatasetDict

# Import custom modules
from config import Config, LoRASBConfig
from dataset_utils import DatasetLoader
from lora_sb import LoRASBModel, LoRASBInitializer
from trainer import LoRASBTrainer
from evaluation import Evaluator

# Define the path to the config file
DEFAULT_CONFIG_PATH = "config.yaml"

def main() -> None:
    """
    Main function to orchestrate the LoRA-SB experiment.
    Handles argument parsing, configuration, setup, initialization, training, and evaluation.
    """
    parser = argparse.ArgumentParser(description="LoRA-SB Reproduction Experiment")

    # Command-line arguments for overriding config.yaml settings
    parser.add_argument("--config_path", type=str, default=DEFAULT_CONFIG_PATH,
                        help=f"Path to the YAML configuration file. Default: {DEFAULT_CONFIG_PATH}")
    parser.add_argument("--model_name", type=str, help="HuggingFace model name (e.g., 'mistralai/Mistral-7B-v0.1')")
    parser.add_argument("--task_name", type=str, help="Name of the task (e.g., 'MetaMathQA', 'glue_cola')")
    parser.add_argument("--rank", type=int, help="LoRA rank (r)")
    parser.add_argument("--learning_rate", type=float, help="Learning rate for training")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Per device training batch size")
    parser.add_argument("--max_seq_len", type=int, help="Maximum sequence length for tokenization")
    parser.add_argument("--grad_acc_steps", type=int, help="Gradient accumulation steps")
    parser.add_argument("--output_dir", type=str, help="Output directory for logs and checkpoints")
    parser.add_argument("--random_seed", type=int, help="Base random seed for reproducibility")
    parser.add_argument("--num_runs", type=int, help="Number of experimental runs for averaging results")
    parser.add_argument("--init_sample_ratio", type=float, help="Ratio of dataset for LoRA-SB initialization")
    parser.add_argument("--lora_sb_dropout", type=float, help="Dropout for LoRA-SB layers")
    parser.add_argument("--lr_scheduler_type", type=str, help="Learning rate scheduler type")
    parser.add_argument("--warmup_ratio", type=float, help="Warmup ratio for learning rate scheduler")


    args = parser.parse_args()

    # Load configuration from YAML and override with command-line arguments
    cfg = Config()
    cfg.load_from_args(args)

    print("\n--- Experiment Configuration ---")
    for attr, value in vars(cfg).items():
        print(f"  {attr}: {value}")
    print("--------------------------------\n")

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare for storing results across multiple runs
    all_run_metrics: Dict[str, List[float]] = {}

    # Main loop for multiple experimental runs
    for run_idx in range(cfg.num_runs):
        current_random_seed = cfg.random_seed + run_idx
        set_seed(current_random_seed)
        print(f"\n--- Starting Run {run_idx + 1}/{cfg.num_runs} with seed {current_random_seed} ---")

        # 1. Load Tokenizer
        print(f"Loading tokenizer: {cfg.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        # Some models don't have a pad token, set it to EOS token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 2. Determine Model Type and Load Base Model
        model_class = None
        is_causal_lm = False
        if cfg.task_name in ["MetaMathQA", "COMMONSENSE170K"]:
            model_class = AutoModelForCausalLM
            is_causal_lm = True
        elif cfg.task_name.startswith("glue_"):
            model_class = AutoModelForSequenceClassification
            is_causal_lm = False
            # For GLUE tasks, need to explicitly pass num_labels
            # Assume 2 for most, but get from dataset later if needed or a mapping.
            # For now, default to 2.
            num_labels_map = {
                "cola": 2, "mrpc": 2, "rte": 2, "sst2": 2,
                "qnli": 2, "stsb": 1 # STS-B is regression, treated as sequence classification with 1 label
            }
            num_labels = num_labels_map.get(cfg.task_name.split("_")[1], 2)
            print(f"Loading {model_class.__name__} for {cfg.task_name} with {num_labels} labels.")
            base_model = model_class.from_pretrained(cfg.model_name, torch_dtype=torch.bfloat16, num_labels=num_labels)
        else:
            # Default to CausalLM if task type is unknown
            print(f"Warning: Task '{cfg.task_name}' not explicitly mapped to model type. Defaulting to AutoModelForCausalLM.")
            model_class = AutoModelForCausalLM
            is_causal_lm = True

        print(f"Loading base model: {cfg.model_name} in torch.bfloat16")
        if is_causal_lm:
            base_model = model_class.from_pretrained(cfg.model_name, torch_dtype=torch.bfloat16)
        else: # SequenceClassification case already handled above with num_labels
             # If it was not GLUE and still not causal_lm, then this is an error
            if model_class is AutoModelForCausalLM: # Re-check if it defaulted
                 base_model = model_class.from_pretrained(cfg.model_name, torch_dtype=torch.bfloat16)


        # Handle Llama-3.2 3B ambiguity:
        # The paper mentions 'Llama-3.2 3B'. We need to be careful with exact model names.
        # If the exact model name isn't found, we can try a known similar one.
        # Here, we assume `cfg.model_name` will be set correctly in config.yaml,
        # or substituted by the user via CLI.
        # If, for instance, `meta-llama/Llama-3.2-3B-Instruct` is specified and not found,
        # an error will be raised by `from_pretrained`. For a robust solution,
        # one might add a try-except block here.
        # Example for robust Llama-3.2 3B handling (conceptual):
        # try:
        #     base_model = model_class.from_pretrained(cfg.model_name, torch_dtype=torch.bfloat16, **(num_labels_kwarg if not is_causal_lm else {}))
        # except OSError as e:
        #     print(f"Model '{cfg.model_name}' not found, trying 'meta-llama/Llama-3-8B-Instruct' as fallback. Error: {e}")
        #     cfg.model_name = "meta-llama/Llama-3-8B-Instruct"
        #     tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        #     if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        #     base_model = model_class.from_pretrained(cfg.model_name, torch_dtype=torch.bfloat16, **(num_labels_kwarg if not is_causal_lm else {}))
        
        base_model.to(device)

        # 3. Load and Preprocess Dataset
        print(f"Loading and preprocessing dataset for task: {cfg.task_name}")
        dataset_loader = DatasetLoader(tokenizer, cfg)
        dataset_dict: DatasetDict = dataset_loader.load_and_preprocess_dataset(cfg.task_name)
        
        train_dataset = dataset_dict["train"]
        eval_dataset = dataset_dict["validation"] if "validation" in dataset_dict else dataset_dict["test"]

        # For generative tasks, ensure original labels are stored for evaluation
        if is_causal_lm:
            # If the task is COMMONSENSE170K, the 'text' field already contains the answer.
            # For MetaMathQA using 'math_qa', we need to extract the answer part.
            if cfg.task_name == "COMMONSENSE170K":
                def extract_answer_from_text(example):
                    match = re.search(r"Answer:\s*(.+)", example['text'])
                    return {"original_labels": match.group(1).strip()} if match else {"original_labels": ""}
                eval_dataset = eval_dataset.map(extract_answer_from_text)
            elif cfg.task_name == "MetaMathQA":
                # Assuming the answer is present in 'answer' column of original 'math_qa' dataset
                # This should be mapped during preprocessing already, check `_preprocess_function_causal_lm`.
                # If not, need to add it here. For simplicity, assume it's in original format.
                if "answer" in dataset_dict["test"].column_names:
                    eval_dataset = eval_dataset.add_column("original_labels", dataset_dict["test"]["answer"])
                else:
                     raise ValueError("For MetaMathQA, original 'answer' column not found in dataset for evaluation.")

        # Instantiate data collator
        data_collator = dataset_loader.get_data_collator(cfg.task_name, "causal_lm" if is_causal_lm else "sequence_classification")

        # 4. LoRA-SB Configuration
        lora_sb_cfg = LoRASBConfig(
            r=cfg.rank,
            target_modules=cfg.target_modules,
            lora_dropout=cfg.lora_sb_dropout,
            s=cfg.lora_sb_s,
            task_type="CAUSAL_LM" if is_causal_lm else "SEQ_CLS" # peft config needs task_type
        )

        # 5. LoRA-SB Initialization
        print("Initializing LoRA-SB matrices...")
        initializer = LoRASBInitializer(
            base_model=base_model,
            tokenizer=tokenizer,
            config=cfg,
            lora_sb_config=lora_sb_cfg,
            learning_rate_for_avg_grad=cfg.learning_rate # As per "Anything UNCLEAR" point 3
        )

        # Get subset of training data for initialization
        # The minimum samples logic is handled inside dataset_loader.get_init_subset
        init_dataset = dataset_loader.get_init_subset(train_dataset)

        # Estimate average gradients (ΔW_avg)
        avg_gradients = initializer.estimate_avg_gradient(init_dataset)

        # Initialize LoRA-SB matrices (B, R, A)
        initialized_lora_matrices = initializer.initialize_lora_matrices(avg_gradients)

        # 6. Create LoRA-SB Model
        print("Creating LoRA-SB adapted model...")
        lora_sb_model = LoRASBModel(base_model, lora_sb_cfg, initialized_lora_matrices)
        lora_sb_model.print_trainable_parameters()
        
        # 7. Training Setup
        output_dir_run = os.path.join(cfg.output_dir, f"run_{run_idx+1}_seed_{current_random_seed}")
        os.makedirs(output_dir_run, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=output_dir_run,
            overwrite_output_dir=True,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.grad_acc_steps,
            num_train_epochs=cfg.epochs,
            learning_rate=cfg.learning_rate,
            lr_scheduler_type=cfg.lr_scheduler_type,
            warmup_ratio=cfg.warmup_ratio,
            logging_dir=os.path.join(output_dir_run, "logs"),
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="no", # Eval happens manually after training
            do_train=True,
            do_eval=False,
            seed=current_random_seed,
            fp16=False, # Use bf16 if model is loaded as such, or fp32 otherwise.
            bf16=True if base_model.dtype == torch.bfloat16 else False,
            report_to="none",
            remove_unused_columns=False, # Necessary when custom inputs/outputs for models
        )

        print("Initializing LoRASBTrainer...")
        lora_sb_trainer = LoRASBTrainer(
            model=lora_sb_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset, # Pass eval_dataset for potential data_collator use, not for trainer's eval loop
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        # 8. Train the Model
        print("Starting training...")
        train_result = lora_sb_trainer.train()
        print("Training complete.")

        # 9. Evaluate the Model
        print("Starting evaluation...")
        evaluator = Evaluator(tokenizer, cfg)
        current_run_metrics = evaluator.evaluate(lora_sb_model, eval_dataset, cfg.task_name)
        
        # Store metrics for aggregation
        for metric_name, value in current_run_metrics.items():
            if metric_name not in all_run_metrics:
                all_run_metrics[metric_name] = []
            all_run_metrics[metric_name].append(value)

        # Clean up to free memory for the next run
        del base_model
        del lora_sb_model
        del lora_sb_trainer
        torch.cuda.empty_cache()

    # 10. Report Final Aggregated Results
    print("\n--- Aggregated Results Across All Runs ---")
    for metric_name, values in all_run_metrics.items():
        avg_value = np.mean(values)
        std_value = np.std(values)
        print(f"  {metric_name}: Average = {avg_value:.4f}, Std Dev = {std_value:.4f}")
    print("------------------------------------------\n")

if __name__ == "__main__":
    main()


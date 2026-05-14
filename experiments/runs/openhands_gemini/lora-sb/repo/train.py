
import torch
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import load_metric
import numpy as np

from config import Config
from model import load_base_model, LoRASBModel
from data import (
    get_tokenizer,
    get_dataloader_for_initialization,
    get_glue_datasets,
    get_metamath_datasets,
    get_data_collator,
)
from utils import estimate_first_step_gradients, lora_sb_initialization
from lora_sb_layers import LoRASBLayer

# TODO: Add specific dataset loading for COMMONSENSE170K and evaluation metrics for all.

def main():
    config = Config()

    # --- Setup Task-Specific Configurations ---
    # This example will focus on one task type for simplicity,
    # but in a full reproduction, you'd loop through or select tasks.
    
    # For demonstration, let's pick an arithmetic task config
    task_config = config.TASK_CONFIGS["arithmetic"]
    config.model_name = task_config["model_name"]
    config.rank = task_config["ranks"][0] # Use the first rank for now
    config.learning_rate = task_config["learning_rate"]
    config.batch_size = task_config["batch_size"]
    config.max_seq_len = task_config["max_seq_len"]
    config.gradient_accumulation_steps = task_config["gradient_accumulation_steps"]
    config.epochs = task_config["epochs"]
    config.dropout = task_config["dropout"]
    config.lr_scheduler_type = task_config["lr_scheduler_type"]
    config.target_modules_llm = task_config["target_modules"] # Ensure correct target modules are used

    # Assume causal_lm task type for arithmetic for now
    task_type = "causal_lm" 
    if "nlu" in task_config: # Simple check to differentiate NLU from LLM tasks
        task_type = "sequence_classification"
        config.target_modules_roberta = task_config["target_modules"]
    
    print(f"--- Running with Config for {task_type} task ---")
    for key, value in task_config.items():
        print(f"{key}: {value}")
    print(f"LoRA-SB Rank: {config.rank}, Scaling Factor: {config.scaling_factor}")

    tokenizer = get_tokenizer(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # Or other suitable token
    
    # --- Base Model Loading ---
    base_model = load_base_model(config.model_name, task_type)
    base_model.to(config.device)

    # --- LoRA-SB Initialization ---
    print("Starting LoRA-SB initialization...")
    init_dataloader = get_dataloader_for_initialization(
        tokenizer=tokenizer,
        config=config,
        task_name="arithmetic" if task_type == "causal_lm" else "nlu", # TODO: Refine task_name for init
        model_name=config.model_name,
        num_samples=config.num_initialization_samples,
        max_seq_len=config.max_seq_len,
        batch_size=config.batch_size,
        seed=config.seed,
    )

    # Temporarily make the original weights trainable to compute gradients
    # Note: this is a simplified approach; `utils.estimate_first_step_gradients`
    # handles setting `requires_grad` for target modules internally.
    # We pass a 'dummy' model to it to get the ΔW_avg for original modules
    # before they are wrapped by LoRASBModel.
    
    # Create a *temporary* copy of the relevant parts of the base_model
    # or ensure estimate_first_step_gradients handles `requires_grad` changes
    # robustly on the original base_model.
    # The current `estimate_first_step_gradients` will modify `requires_grad`
    # on `base_model` directly, then restore it. This is fine.

    delta_w_avg_dict = estimate_first_step_gradients(
        base_model, init_dataloader, config.num_initialization_samples,
        config.device, config.target_modules_llm if task_type == "causal_lm" else config.target_modules_roberta
    )
    print("Finished estimating first-step gradients.")

    # --- Wrap base model with LoRA-SB layers ---
    lora_sb_model = LoRASBModel(base_model, config, task_type)
    lora_sb_model.to(config.device)
    
    # Apply LoRA-SB initialization to the new LoRASBLayers
    for name, module in lora_sb_model.lora_sb_layers.items():
        if isinstance(module, LoRASBLayer):
            original_module_name = name.replace(".lora_sb_layers", "") # Adjust name for delta_w_avg_dict lookup
            if original_module_name in delta_w_avg_dict:
                lora_sb_initialization(
                    module,
                    delta_w_avg_dict[original_module_name],
                    config.rank,
                    config.scaling_factor
                )
            else:
                print(f"Warning: No delta_w_avg found for {original_module_name}. Initializing with zeros.")
    
    lora_sb_model.print_trainable_parameters()
    
    # --- Load Full Datasets for Training ---
    # For arithmetic (MetaMathQA)
    if task_type == "causal_lm":
        train_dataset = get_metamath_datasets(tokenizer, config.max_seq_len)["train"]
        # For evaluation, the paper uses GSM8K and MATH, which are external benchmarks.
        # This setup would typically involve a separate evaluation script or custom Trainer logic.
        # For now, we'll just train.
        eval_dataset = None # Placeholder
        metric = None
    elif task_type == "sequence_classification":
        glue_task_name = "sst2" # Example GLUE task
        tokenized_glue_datasets = get_glue_datasets(tokenizer, glue_task_name, config.max_seq_len)
        train_dataset = tokenized_glue_datasets["train"]
        eval_dataset = tokenized_glue_datasets["validation"]
        metric = load_metric("glue", glue_task_name)

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            if glue_task_name != "stsb":
                predictions = np.argmax(predictions, axis=1)
            else:
                predictions = predictions[:, 0]
            return metric.compute(predictions=predictions, references=labels)
    else:
        raise ValueError("Unsupported task type for full dataset loading.")


    data_collator = get_data_collator(tokenizer, task_type)

    # --- Training Arguments ---
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=500,
        evaluation_strategy="epoch" if eval_dataset else "no",
        save_total_limit=1,
        fp16=torch.cuda.is_available(), # Use mixed precision if GPU available
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        seed=config.seed,
    )

    # --- Trainer ---
    trainer = Trainer(
        model=lora_sb_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if task_type == "sequence_classification" else None,
    )

    # --- Train ---
    print("Starting training...")
    trainer.train()
    print("Training complete.")

    # --- Evaluation (simplified, actual evaluation would be more complex as per paper) ---
    if eval_dataset:
        print("Starting evaluation...")
        results = trainer.evaluate()
        print(f"Evaluation Results: {results}")

if __name__ == "__main__":
    main()


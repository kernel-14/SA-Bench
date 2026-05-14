import argparse
import os
import json
import torch
import torch.distributed as dist
from accelerate import Accelerator # Import Accelerator

from config import Config
from model.navil import NaViLModel
from trainer import NaViLTrainer
from evaluation import NaViLEvaluator
from utils import setup_logging, is_main_process # Use the updated utils for logging
from transformers import AutoTokenizer # Removed AutoConfig, BitsAndBytesConfig as not directly used in main
from loguru import logger # Ensure logger is imported

def main():
    """
    Main entry point for the NaViL project.
    Handles argument parsing, configuration loading, distributed training setup,
    model initialization, and orchestrates training or evaluation based on mode.
    """
    parser = argparse.ArgumentParser(description="Train or evaluate NaViL model.")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to the YAML configuration file (e.g., config.yaml).")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "eval"],
                        help="Mode to run: 'train' for training, 'eval' for evaluation.")
    parser.add_argument("--model_variant", type=str, required=False,
                        choices=["navil_2b", "navil_9b"],
                        help="Model variant to use (e.g., 'navil_2b' or 'navil_9b'). Required for 'train' mode. For 'eval' mode, it can be inferred from checkpoint's config or explicitly specified if no config in checkpoint.")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to a model checkpoint. For 'train', used for resuming. For 'eval', used for loading model weights.")
    parser.add_argument("--eval_benchmarks", type=str, default=None,
                        help="Comma-separated list of benchmarks to run in 'eval' mode. Defaults to all benchmarks specified in the config.yaml.")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Output directory for logs, checkpoints, and evaluation results. Will be created if it does not exist.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training. This is typically set automatically by launch utilities (e.g., torch.distributed.launch or accelerate launch).")

    args = parser.parse_args()

    # 1. Configuration Loading
    # The Config class now handles merging variant-specific settings upon loading.
    config = Config()
    
    # If checkpoint_path is provided in eval mode, we might want to load the config from the checkpoint
    # first, and then potentially override with CLI args. For simplicity, we'll load base config first.
    # The `load_config` method will handle if `model_variant` is None for eval mode.
    config.load_config(args.config_path, args.model_variant)
    
    # Update config with CLI arguments that override YAML or set runtime paths
    config.output_dir = args.output_dir # Store output_dir from CLI in config
    # Ensure current_model_variant is set if provided via CLI for later use (e.g., evaluation naming)
    if args.model_variant:
        config.current_model_variant = args.model_variant

    # Create output directory early for logs and checkpoints
    os.makedirs(config.output_dir, exist_ok=True)

    # 2. Initialize Hugging Face Accelerator for distributed training and mixed precision
    # This also sets up torch.distributed backend if necessary.
    accelerator = Accelerator(
        gradient_accumulation_steps=config.get("common.gradient_accumulation_steps", 1),
        mixed_precision=config.get("common.numerical_precision", "no"), # map bfloat16/float16 to bf16/fp16
        log_with="tensorboard", # Or any other logging backend supported by accelerate
        project_dir=config.output_dir, # Use project_dir for tensorboard logs
    )
    
    # Store global rank and world size in config for convenience and consistent access
    config.global_rank = accelerator.process_index
    config.world_size = accelerator.num_processes
    config.is_main_process = accelerator.is_main_process

    # 3. Logging Setup
    # Pass accelerator's `is_main_process` and `process_index` to utils.setup_logging
    setup_logging(config.output_dir, accelerator.is_main_process, accelerator.process_index)
    
    if accelerator.is_main_process:
        logger.info(f"Configuration loaded (final merged view): {config}") # Config.__str__ provides JSON
        logger.info(f"Running in {args.mode} mode for model variant: {config.current_model_variant if config.current_model_variant else 'N/A'}")
        logger.info(f"Output directory: {config.output_dir}")
        logger.info(f"Device: {accelerator.device}")
        logger.info(f"Number of processes: {accelerator.num_processes}, Mixed precision: {accelerator.mixed_precision}")
        logger.info(f"Gradient accumulation steps: {accelerator.gradient_accumulation_steps}")

    # 4. Tokenizer Initialization
    llm_name_or_path = config.get('llm_name_or_path')
    if not llm_name_or_path and args.mode == "train":
        # llm_name_or_path is essential for training mode
        if accelerator.is_main_process:
            logger.error("LLM name or path is not specified in the configuration. Cannot initialize tokenizer for training.")
        return # Exit if essential config is missing.
    elif not llm_name_or_path and args.mode == "eval":
        # For eval, if not explicitly in config, it should ideally be derived from checkpoint's metadata.
        # But for now, we'll require it to be present for tokenizer init.
        if accelerator.is_main_process:
            logger.error("LLM name or path is not specified in the configuration. Cannot initialize tokenizer for evaluation.")
        return

    tokenizer = AutoTokenizer.from_pretrained(llm_name_or_path, trust_remote_code=True) # trust_remote_code for Qwen etc.

    # Add NaViL's specific special tokens to the tokenizer
    special_tokens_list = list(config.special_tokens.values())
    num_added_tokens = tokenizer.add_special_tokens({'additional_special_tokens': special_tokens_list})
    if accelerator.is_main_process:
        logger.info(f"Added {num_added_tokens} new special tokens to tokenizer: {special_tokens_list}")
        logger.info(f"Tokenizer vocabulary size after adding special tokens: {len(tokenizer)}")

    # 5. Model Instantiation
    if accelerator.is_main_process:
        logger.info("Initializing NaViLModel...")
    # The NaViLModel constructor will handle the initialization of VisualEncoder, Connector, and MoELLM
    # including injecting MoE layers into the base LLM.
    # MoELLM's internal AutoModelForCausalLM uses device_map="auto" to distribute model across devices.
    navil_model = NaViLModel(config, tokenizer)
    
    # MoELLM's __init__ already handles `resize_token_embeddings` after `add_special_tokens`.

    # 6. Main Logic (Train or Eval)
    if args.mode == 'train':
        trainer = NaViLTrainer(navil_model, tokenizer, config, accelerator) 
        trainer.train(args.checkpoint_path) # Trainer will handle loading checkpoint internally (Accelerator state)

    elif args.mode == 'eval':
        if not args.checkpoint_path:
            if accelerator.is_main_process:
                logger.error("A checkpoint path must be provided for evaluation mode.")
            return

        if accelerator.is_main_process:
            logger.info(f"Loading model checkpoint for evaluation from: {args.checkpoint_path}")
        
        # For evaluation, we need to load just the model's state dict.
        # The `accelerator.load_state` method is for loading full training states (model, optimizer, scheduler).
        # When only model weights are needed, direct `torch.load` is appropriate.
        try:
            # Load checkpoint data. Map to CPU first, then model.to(device) will distribute.
            checkpoint_data = torch.load(args.checkpoint_path, map_location="cpu")
            if "model_state_dict" in checkpoint_data:
                # If the checkpoint is a full training state (from Accelerator.save_state)
                navil_model.load_state_dict(checkpoint_data["model_state_dict"], strict=True)
                if accelerator.is_main_process:
                    logger.info("Model state dictionary loaded from full checkpoint.")
                # Optionally, if config was saved in checkpoint, could update config object here
                # if "config_snapshot" in checkpoint_data:
                #     config._config_data.update(EasyDict(checkpoint_data["config_snapshot"]))
            else:
                 # Assume the checkpoint file directly contains the model's state_dict
                 navil_model.load_state_dict(checkpoint_data, strict=True) 
                 if accelerator.is_main_process:
                    logger.info("Raw model state dictionary loaded from checkpoint file.")
        except Exception as e:
            if accelerator.is_main_process:
                logger.error(f"Failed to load model checkpoint from {args.checkpoint_path}: {e}")
            return
        
        # Prepare model for evaluation using accelerator (moves to device, handles DDP wrapping)
        navil_model = accelerator.prepare(navil_model)
        navil_model.eval() # Set model to evaluation mode

        evaluator = NaViLEvaluator(navil_model, tokenizer, config, accelerator)
        
        benchmarks_to_run_str = args.eval_benchmarks
        if benchmarks_to_run_str:
            benchmarks_to_run = [b.strip() for b in benchmarks_to_run_str.split(',')]
        else:
            benchmarks_to_run = config.get("evaluation.benchmarks", []) # Get from config if not specified via CLI

        if accelerator.is_main_process:
            logger.info(f"Evaluating on benchmarks: {benchmarks_to_run}")

        all_results = {}
        for benchmark_name in benchmarks_to_run:
            if accelerator.is_main_process:
                logger.info(f"Starting evaluation for benchmark: {benchmark_name}")
            try:
                results = evaluator.evaluate(benchmark_name)
                all_results[benchmark_name] = results
            except NotImplementedError as e:
                if accelerator.is_main_process:
                    logger.warning(f"Skipping benchmark '{benchmark_name}': {e}")
                all_results[benchmark_name] = {"error": "NotImplementedError", "message": str(e)}
            except Exception as e:
                if accelerator.is_main_process:
                    logger.error(f"Error evaluating benchmark '{benchmark_name}': {e}")
                all_results[benchmark_name] = {"error": str(e)}
        
        if accelerator.is_main_process:
            logger.info(f"--- Overall Evaluation Results ---")
            for benchmark, results in all_results.items():
                logger.info(f"{benchmark}: {json.dumps(results)}")
            
            # Save all results to a JSON file
            results_filename = f"{config.current_model_variant}_evaluation_results.json" if config.current_model_variant else "evaluation_results.json"
            results_filepath = os.path.join(config.output_dir, results_filename)
            with open(results_filepath, 'w') as f:
                json.dump(all_results, f, indent=4)
            logger.info(f"Evaluation complete. Results saved to {results_filepath}")

    if accelerator.is_main_process:
        logger.info("Program finished successfully.")

    # Accelerator handles cleanup of distributed processes
    # (e.g., dist.destroy_process_group() is implicitly called)

if __name__ == "__main__":
    main()

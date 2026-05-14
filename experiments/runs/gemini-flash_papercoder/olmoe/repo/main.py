import argparse
import os
import torch
import yaml
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs
from typing import Dict, Any, Tuple, Optional

# Local imports
from config import Config, ModelConfig, DataConfig, TrainingConfig, EvaluationConfig
from model.olmoe_model import OLMoEModel
from data.pretraining_dataset import PretrainingDataset
from data.adaptation_dataset import AdaptationDataset
from data.data_collator import OLMoEDataCollator
from training.loss_functions import LossCalculator
from training.optimizer_scheduler import OptimizerSchedulerFactory
from training.pretrainer import Pretrainer
from training.sft_trainer import SFTTrainer
from training.dpo_trainer import DPOTrainer
from evaluation.evaluator import Evaluator
from utils.logger import Logger


def main():
    """
    Main entry point for the OLMoE reproduction pipeline.
    Handles argument parsing, configuration loading, component initialization,
    and orchestrates the pretraining, adaptation, and evaluation phases.
    """
    parser = argparse.ArgumentParser(description="Reproduce OLMoE experiments.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="./config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["pretrain", "sft", "dpo", "all", "eval_only"],
        help="Operational mode: 'pretrain', 'sft', 'dpo', 'all' (sequential), or 'eval_only'.",
    )
    parser.add_argument(
        "--checkpoint_path_pretrained",
        type=str,
        default=None,
        help="Path to a pretrained model checkpoint (e.g., for resuming pretrain or starting SFT).",
    )
    parser.add_argument(
        "--checkpoint_path_sft",
        type=str,
        default=None,
        help="Path to an SFT model checkpoint (e.g., for resuming DPO or starting SFT eval).",
    )
    parser.add_argument(
        "--checkpoint_path_dpo",
        type=str,
        default=None,
        help="Path to a DPO model checkpoint (e.g., for final evaluation).",
    )
    args = parser.parse_args()

    # 1. Load Configuration
    config = Config.load_from_yaml(args.config_path)

    # 2. Initialize Accelerator for distributed training and mixed precision
    # Use training.gradient_accumulation_steps for pretrain, but trainers will set their own
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=config.training.precision,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        log_with="wandb",
        project_dir=config.training.checkpoint_dir,
        project_config=ProjectConfiguration(
            project_dir=config.training.checkpoint_dir,
            logging_dir=os.path.join(config.training.checkpoint_dir, "logs")
        ),
        kwargs_handlers=[ddp_kwargs],
    )

    # 3. Initialize Logger
    # Logger should be initialized only once per run on the main process
    logger = Logger(
        config.training.project_name,
        f"{config.training.run_name}_{args.mode}_{os.getpid()}", # Unique run name for wandb
        config
    )
    if accelerator.is_main_process:
        accelerator.print(f"Starting run in mode: {args.mode}")
        accelerator.print(f"Loaded configuration from: {args.config_path}")
        accelerator.print(f"Checkpoint directory: {config.training.checkpoint_dir}")
    
    # 4. Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_name)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            # Fallback for tokenizers without a specific pad_token_id
            accelerator.print(f"Warning: Tokenizer does not have a pad_token_id or eos_token_id. Adding '[PAD]' as pad_token.")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # This might change vocab size, ensure model is aware if vocab size was fixed.
            # Assuming AutoTokenizer handles this gracefully or config.model.vocab_size is sufficient.
        if accelerator.is_main_process:
            accelerator.print(f"Tokenizer pad_token_id set to: {tokenizer.pad_token_id}")

    # 5. Initialize Model
    model = OLMoEModel(config.model, tokenizer.vocab_size, config.data.max_seq_len)

    # 6. Initialize Loss Calculator
    loss_calculator = LossCalculator(
        config.training.lbl_weight, config.training.rz_loss_weight
    )

    # Placeholders for models at different stages
    model_pretrained = model
    model_sft = model
    model_instruct = model

    # --- Phase 0: Checkpoint Loading for starting point ---
    loaded_hf_model_path: Optional[str] = None
    if args.checkpoint_path_dpo is not None:
        loaded_hf_model_path = os.path.join(args.checkpoint_path_dpo, "hf_model")
        if accelerator.is_main_process:
            accelerator.print(f"Loading DPO model from {loaded_hf_model_path}")
    elif args.checkpoint_path_sft is not None:
        loaded_hf_model_path = os.path.join(args.checkpoint_path_sft, "hf_model")
        if accelerator.is_main_process:
            accelerator.print(f"Loading SFT model from {loaded_hf_model_path}")
    elif args.checkpoint_path_pretrained is not None:
        loaded_hf_model_path = os.path.join(args.checkpoint_path_pretrained, "hf_model")
        if accelerator.is_main_process:
            accelerator.print(f"Loading pretrained model from {loaded_hf_model_path}")

    if loaded_hf_model_path:
        # Load state dict directly to the model instance. Using map_location='cpu' to avoid
        # potential CUDA issues if not all devices are available during initial load.
        # Accelerator will then move it to the correct device.
        model.load_state_dict(torch.load(os.path.join(loaded_hf_model_path, "pytorch_model.bin"), map_location='cpu'))
        # The model variable now holds the loaded weights. This will be prepared by Accelerator later.

    # Prepare the base model (which might have loaded weights) with Accelerator
    # This applies FSDP and moves the model to the correct device.
    model = accelerator.prepare(model)
    model_pretrained = model # This is the prepared model
    model_sft = model # For SFT/DPO phases, starting from this (possibly loaded) model
    model_instruct = model # For DPO/eval phases, starting from this (possibly loaded) model

    # Ensure logger watches the (now prepared) model
    if accelerator.is_main_process:
        logger.watch_model(model) # Watch the prepared model instance

    accelerator.wait_for_everyone()

    # 7. Initialize Evaluator (needs the prepared model and accelerator)
    evaluator = Evaluator(model, tokenizer, config, logger, accelerator)

    # --- Phase 1: Pretraining ---
    if args.mode in ["pretrain", "all"]:
        if accelerator.is_main_process:
            accelerator.print("\n--- Starting Pretraining Phase ---")

        pretrain_train_ds = PretrainingDataset(config.data, tokenizer)
        # Preprocessing on main process only, then broadcast or load by others
        if accelerator.is_main_process:
            pretrain_train_ds.preprocess_data()
            pretrain_train_ds.shuffle_data() # Initial shuffle
        
        # Need to ensure all processes have a ready dataset object for DataLoader.
        # If preprocess_data modifies self.hf_dataset in place, then non-main processes need to load it.
        # For simplicity in this structure, we assume `preprocess_data` is called once and then
        # the HF dataset can be accessed by other processes if it's memory-mapped or shared.
        # A more robust approach might be to save/load processed dataset.
        accelerator.wait_for_everyone() # Ensure main process finishes preprocessing before dataloaders try to access.
        if not accelerator.is_main_process and pretrain_train_ds.hf_dataset is None:
             pretrain_train_ds.preprocess_data() # Other processes load/access preprocessed data

        # Pretraining progress evaluation datasets (e.g., small validation sets)
        pretrain_eval_ds_map: Dict[str, Dataset] = {} # Empty for now, as datasets need to be implemented
        # Example if you had `MMLUDataset` etc.:
        # if accelerator.is_main_process:
        #     mmlu_eval_ds = MMLUDataset(config.data, tokenizer, split="validation")
        #     mmlu_eval_ds.preprocess_data()
        #     pretrain_eval_ds_map["MMLU_val"] = mmlu_eval_ds
        accelerator.wait_for_everyone() # Synchronize after potential eval data preprocessing

        # Instantiate optimizer and scheduler for pretraining
        pretrain_optimizer, pretrain_lr_scheduler = OptimizerSchedulerFactory.create_pretrain_optimizer_and_scheduler(
            model_pretrained, config
        )

        pretrainer = Pretrainer(
            model_pretrained,
            pretrain_train_ds,
            pretrain_eval_ds_map, # Pass the map of eval datasets
            config,
            tokenizer,
            logger,
            loss_calculator,
            pretrain_optimizer,
            pretrain_lr_scheduler,
        )
        pretrainer.train()
        model_pretrained = pretrainer.model # Get the final trained model from pretrainer
        accelerator.wait_for_everyone()

    # --- Phase 2: Post-Pretraining Evaluation ---
    if args.mode in ["pretrain", "all", "eval_only"]:
        if accelerator.is_main_process:
            accelerator.print("\n--- Running Post-Pretraining Evaluation ---")
        
        # The evaluator instance uses the `model_pretrained` (which is the prepared model)
        eval_metrics = evaluator.evaluate_post_pretraining()
        if accelerator.is_main_process:
            logger.log({f"final_eval_post_pretrain/{k}": v for k, v in eval_metrics.items()}, step=-1)
        accelerator.wait_for_everyone()


    # --- Phase 3: SFT (Supervised Fine-Tuning) ---
    if args.mode in ["sft", "all"]:
        if accelerator.is_main_process:
            accelerator.print("\n--- Starting SFT Phase ---")
        
        sft_train_ds = AdaptationDataset(config.data, tokenizer, is_dpo=False)
        sft_eval_ds = AdaptationDataset(config.data, tokenizer, is_dpo=False) # For SFT evaluation on specific adaptation data
        if accelerator.is_main_process:
            sft_train_ds.preprocess_data()
            sft_eval_ds.preprocess_data()
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            sft_train_ds.preprocess_data() # Other processes load/access preprocessed data
            sft_eval_ds.preprocess_data() # Other processes load/access preprocessed data


        # Calculate num_sft_steps to correctly initialize constant LR scheduler
        num_sft_steps = config.training.sft_epochs * (
            len(sft_train_ds) // config.training.sft_global_batch_size_samples
        ) # Use floor division for steps based on global batches

        sft_optimizer, sft_lr_scheduler = OptimizerSchedulerFactory.create_sft_optimizer_and_scheduler(
            model_sft, config, num_sft_steps
        )

        sft_trainer = SFTTrainer(
            model_sft, # Pass the model prepared by Accelerator
            sft_train_ds,
            sft_eval_ds,
            config,
            tokenizer,
            logger,
            loss_calculator,
            sft_optimizer,
            sft_lr_scheduler,
        )
        sft_trainer.train()
        model_sft = sft_trainer.model # Get the fine-tuned model
        model_instruct = model_sft # Reference for the next phase
        accelerator.wait_for_everyone()

    # --- Phase 4: DPO (Direct Preference Optimization) ---
    if args.mode in ["dpo", "all"]:
        if accelerator.is_main_process:
            accelerator.print("\n--- Starting DPO Phase ---")

        dpo_train_ds = AdaptationDataset(config.data, tokenizer, is_dpo=True)
        dpo_eval_ds = AdaptationDataset(config.data, tokenizer, is_dpo=True) # For DPO evaluation on specific adaptation data
        if accelerator.is_main_process:
            dpo_train_ds.preprocess_data()
            dpo_eval_ds.preprocess_data()
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            dpo_train_ds.preprocess_data() # Other processes load/access preprocessed data
            dpo_eval_ds.preprocess_data() # Other processes load/access preprocessed data

        # Calculate num_dpo_steps to correctly initialize constant LR scheduler
        num_dpo_steps = config.training.dpo_epochs * (
            len(dpo_train_ds) // config.training.dpo_global_batch_size_samples
        ) # Use floor division for steps based on global batches

        dpo_optimizer, dpo_lr_scheduler = OptimizerSchedulerFactory.create_dpo_optimizer_and_scheduler(
            model_instruct, config, num_dpo_steps
        )

        dpo_trainer = DPOTrainer(
            model_instruct, # Pass the model prepared by Accelerator
            dpo_train_ds,
            dpo_eval_ds,
            config,
            tokenizer,
            logger,
            loss_calculator,
            dpo_optimizer,
            dpo_lr_scheduler,
        )
        dpo_trainer.train()
        model_instruct = dpo_trainer.model # Get the final instruction-tuned model
        accelerator.wait_for_everyone()

    # --- Phase 5: Post-Adaptation Evaluation (Final Evaluation) ---
    if args.mode in ["sft", "dpo", "all", "eval_only"]:
        if accelerator.is_main_process:
            accelerator.print("\n--- Running Post-Adaptation (Final) Evaluation ---")

        # The evaluator instance uses the `model_instruct` (the most recently updated prepared model)
        # It handles running evaluations on all specified adaptation tasks.
        final_eval_metrics = evaluator.evaluate_adaptation()
        if accelerator.is_main_process:
            logger.log({f"final_eval_adaptation/{k}": v for k, v in final_eval_metrics.items()}, step=-1)
        accelerator.wait_for_everyone()

    # Finalization
    if accelerator.is_main_process:
        accelerator.print("\n--- OLMoE Reproduction Pipeline Finished ---")
        logger.finish()

if __name__ == "__main__":
    main()


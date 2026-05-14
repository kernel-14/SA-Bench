import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup
import deepspeed # For PPO stage
import json # For loading deepspeed config
import random
from loguru import logger
from typing import Dict, Any, Optional

# To avoid circular imports, define Config as DictConfig directly.
from omegaconf import DictConfig, OmegaConf # Re-import DictConfig for explicit type hinting

# Import custom modules
from data_loader import DataLoader
from models import SFTModel, RewardModel, PolicyModel, ValueModel
from macro_action_handler import MacroActionHandler
from ppo_algorithm import PPOAlgorithm
from utils import TokenizerWrapper, log_metrics

# Alias DictConfig for clearer type hinting consistent with other files
Config = DictConfig

class RLHFTrainer:
    """
    Orchestrates the entire multi-stage training process for MA-RLHF,
    including Supervised Fine-Tuning (SFT), Reward Model (RM) training,
    and Macro-Action Proximal Policy Optimization (MA-PPO).
    """

    def __init__(self, config: Config):
        """
        Initializes the RLHFTrainer, setting up models, data loaders,
        tokenizer, macro action handler, and the PPO algorithm.

        Args:
            config: A DictConfig object containing the global and stage-specific configurations.
        """
        self.config: Config = config
        
        # Set random seed for reproducibility
        if self.config.global.seed is not None:
            random.seed(self.config.global.seed)
            torch.manual_seed(self.config.global.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.global.seed)
            logger.info(f"Set random seed to {self.config.global.seed}")

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(self.config.global.logging_dir)

        # 1. Initialize TokenizerWrapper
        model_name_for_tokenizer = self.config.model_configs[self.config.resolved_model_id].name
        self.tokenizer_wrapper = TokenizerWrapper(model_name_for_tokenizer)

        # 2. Initialize DataLoader
        self.data_loader = DataLoader(self.config, self.tokenizer_wrapper)

        # 3. Instantiate Models
        # Base model name for SFT, Policy, and Value models (Reward model is also based on SFT)
        sft_policy_value_model_name = self.config.model_configs[self.config.resolved_model_id].name
        
        self.sft_model = SFTModel(sft_policy_value_model_name, self.config)
        self.reward_model = RewardModel(sft_policy_value_model_name, self.config) # RM also based on SFT model architecture
        self.policy_model = PolicyModel(sft_policy_value_model_name, self.config)
        self.value_model = ValueModel(sft_policy_value_model_name, self.config)

        # 4. Instantiate Macro Action Handler
        self.macro_action_handler = MacroActionHandler(self.config, self.tokenizer_wrapper)

        # 5. Instantiate PPO Algorithm
        self.ppo_algorithm = PPOAlgorithm(
            self.policy_model,
            self.value_model,
            self.reward_model,
            self.sft_model,
            self.tokenizer_wrapper,
            self.config,
            self.macro_action_handler,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"RLHFTrainer initialized. Training will run on device: {self.device}")

        # Ensure output directory exists
        os.makedirs(self.config.global.output_dir, exist_ok=True)
        os.makedirs(self.config.global.logging_dir, exist_ok=True)
        logger.info(f"Output directory: {self.config.global.output_dir}")
        logger.info(f"Logging directory: {self.config.global.logging_dir}")


    def train_sft(self) -> None:
        """
        Executes the Supervised Fine-Tuning stage, training self.sft_model.
        """
        logger.info("Starting SFT training stage...")
        sft_cfg = self.config.sft_config # Resolved config from initial load_config
        
        sft_dataloader = self.data_loader.load_sft_data(self.config.resolved_task_name)
        if sft_dataloader is None:
            logger.warning("No SFT data loaded. Skipping SFT training.")
            return

        self.sft_model.model.train() # Ensure model is in training mode
        self.sft_model.model.to(self.device) # Ensure model is on correct device

        optimizer = AdamW(self.sft_model.parameters(), lr=sft_cfg.learning_rate)
        
        # Calculate total training steps for scheduler
        num_training_steps = len(sft_dataloader) * sft_cfg.epochs
        num_warmup_steps = int(num_training_steps * sft_cfg.warmup_ratio)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        global_step = 0
        for epoch in range(sft_cfg.epochs):
            logger.info(f"SFT Epoch {epoch + 1}/{sft_cfg.epochs}")
            for step, batch in enumerate(sft_dataloader):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                loss = self.sft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                
                metrics = {
                    "loss": loss.item(),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
                log_metrics(self.writer, metrics, global_step, "sft_train")
        
        # Set SFT model to evaluation mode after training, as it acts as reference later.
        self.sft_model.model.eval()

        # Save SFT model
        output_path = os.path.join(self.config.global.output_dir, "sft_model")
        self.sft_model.model.save_pretrained(output_path)
        self.tokenizer_wrapper.tokenizer.save_pretrained(output_path)
        logger.info(f"SFT model saved to {output_path}")

    def train_rm(self) -> None:
        """
        Trains the Reward Model (self.reward_model) to align with human preferences.
        """
        logger.info("Starting RM training stage...")
        rm_cfg = self.config.rm_config # Resolved config from initial load_config
        
        if rm_cfg.get('skip_training', False):
            logger.info("RM training skipped as per configuration (e.g., for APPS dataset).")
            # If RM is skipped, the value model will later be initialized from SFT
            # We don't save anything if training is skipped.
            return

        # Initialize Reward Model weights from SFT model
        # The RewardModel instance already loaded a CausalLM, now copy SFT weights.
        # `strict=False` because the RewardModel adds a value_head.
        self.reward_model.model.load_state_dict(self.sft_model.model.state_dict(), strict=False)
        logger.info("Reward Model base LLM initialized with SFT model weights.")
        
        rm_dataloader = self.data_loader.load_rm_data(self.config.resolved_task_name)
        if rm_dataloader is None:
            logger.warning("No RM data loaded. Skipping RM training.")
            return

        # Ensure RM is in training mode (its value head needs training)
        self.reward_model.train() 
        self.reward_model.model.to(self.device)
        self.reward_model.value_head.to(self.device)

        optimizer = AdamW(self.reward_model.parameters(), lr=rm_cfg.learning_rate)

        num_training_steps = len(rm_dataloader) * rm_cfg.epochs
        num_warmup_steps = int(num_training_steps * rm_cfg.warmup_ratio)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        global_step = 0
        for epoch in range(rm_cfg.epochs):
            logger.info(f"RM Epoch {epoch + 1}/{rm_cfg.epochs}")
            for step, batch in enumerate(rm_dataloader):
                # Move batch to device
                prompt_ids = batch['prompt_ids'].to(self.device)
                chosen_ids = batch['chosen_ids'].to(self.device)
                rejected_ids = batch['rejected_ids'].to(self.device)

                loss = self.reward_model(prompt_ids=prompt_ids, chosen_ids=chosen_ids, rejected_ids=rejected_ids)
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                
                metrics = {
                    "loss": loss.item(),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
                log_metrics(self.writer, metrics, global_step, "rm_train")
        
        # Set RM model to evaluation mode after training, as it acts as a fixed reward provider.
        self.reward_model.eval()

        # Save RM model
        output_path = os.path.join(self.config.global.output_dir, "reward_model")
        # Saving the base model (LM part)
        self.reward_model.model.save_pretrained(output_path)
        # Saving the custom value head
        torch.save(self.reward_model.value_head.state_dict(), os.path.join(output_path, "value_head.pt"))
        self.tokenizer_wrapper.tokenizer.save_pretrained(output_path)
        logger.info(f"Reward model saved to {output_path}")

    def train_ppo(self) -> None:
        """
        Executes the Reinforcement Learning (MA-PPO) stage, optimizing the policy and value models.
        """
        logger.info("Starting PPO training stage...")
        ppo_cfg = self.config.ppo_config # Resolved config from initial load_config

        # 1. Initialize Policy Model weights from SFT model
        self.ppo_algorithm.policy_model.model.load_state_dict(self.sft_model.model.state_dict())
        logger.info("PPO Policy Model initialized with SFT model weights.")

        # 2. Initialize Value Model weights
        if self.config.rm_config.skip_training:
            # For tasks like APPS, where RM is skipped, initialize Value Model from SFT.
            self.ppo_algorithm.value_model.model.load_state_dict(self.sft_model.model.state_dict())
            # For the value_head in ValueModel, it's typically randomly initialized if not from RM.
            logger.info("PPO Value Model base LLM initialized with SFT model weights (RM skipped).")
        else:
            # Initialize Value Model from Reward Model
            self.ppo_algorithm.value_model.model.load_state_dict(self.reward_model.model.state_dict(), strict=False)
            self.ppo_algorithm.value_model.value_head.load_state_dict(self.reward_model.value_head.state_dict())
            logger.info("PPO Value Model base LLM and value head initialized with Reward Model weights.")

        ppo_dataloader = self.data_loader.load_ppo_data(self.config.resolved_task_name)
        if ppo_dataloader is None:
            logger.warning("No PPO data loaded. Skipping PPO training.")
            return
        
        # 3. DeepSpeed Initialization
        # Initialize distributed environment
        # deepspeed.init_distributed() will be called internally by accelerate if using Accelerator.
        # If running stand-alone deepspeed, it needs to be called.
        # For simplicity, assuming a single-node setup or user manually calls deepspeed.init_distributed.
        # A more robust solution would integrate with `accelerate` or ensure DDP is set up.
        # The design mentioned DeepSpeed-Chat, which typically handles this setup.
        # Assuming DeepSpeed is set up if running with `deepspeed` launcher.
        
        # Load DeepSpeed config file
        deepspeed_config_path = self.config.global.deepspeed_config_path
        if not os.path.exists(deepspeed_config_path):
            raise FileNotFoundError(f"DeepSpeed config file not found at: {deepspeed_config_path}")
        
        with open(deepspeed_config_path, 'r') as f:
            deepspeed_config = json.load(f)

        # DeepSpeed wraps the model and optimizer. We update the models and optimizers within ppo_algorithm.
        # This makes the ppo_algorithm instance operate on DeepSpeed-wrapped engines.
        
        # Wrap Policy Model with DeepSpeed
        self.ppo_algorithm.policy_model.model, self.ppo_algorithm.policy_optimizer, _, _ = deepspeed.initialize(
            model=self.ppo_algorithm.policy_model.model,
            optimizer=self.ppo_algorithm.policy_optimizer,
            config_params=deepspeed_config,
            # Ensure value_head parameters are also included if PolicyModel has one, though unlikely for policy.
            # BaseLLM's parameters() should generally include all parameters of the underlying model.
        )
        logger.info("PPO Policy Model wrapped with DeepSpeed engine.")

        # Wrap Value Model with DeepSpeed
        # ValueModel also has a `value_head` which is an `nn.Linear` module.
        # `ppo_algorithm.value_model.parameters()` should already include both base LM and value_head params.
        self.ppo_algorithm.value_model.model, self.ppo_algorithm.value_model.value_head_optimizer, _, _ = deepspeed.initialize(
            model=self.ppo_algorithm.value_model.model, # Assuming BaseLLM's model is the core part
            optimizer=self.ppo_algorithm.value_optimizer, # PPOAlgorithm's value_optimizer should optimize both
            config_params=deepspeed_config
        )
        logger.info("PPO Value Model wrapped with DeepSpeed engine.")
        

        global_step = 0
        for epoch in range(ppo_cfg.epochs):
            logger.info(f"PPO Epoch {epoch + 1}/{ppo_cfg.epochs}")
            for step, batch_prompts in enumerate(ppo_dataloader):
                policy_loss, critic_loss, metrics = self.ppo_algorithm.step(batch_prompts)

                global_step += 1
                log_metrics(self.writer, metrics, global_step, "ppo_train")

            # Save PPO models periodically (e.g., after each outer epoch)
            # DeepSpeed provides a specific way to save checkpoints.
            # We save the policy model and value model separately.
            policy_output_path = os.path.join(self.config.global.output_dir, f"ppo_policy_model_epoch_{epoch+1}")
            value_output_path = os.path.join(self.config.global.output_dir, f"ppo_value_model_epoch_{epoch+1}")
            
            # DeepSpeed save_checkpoint requires a tag and saves in a directory named by tag.
            # It saves the wrapped model state.
            self.ppo_algorithm.policy_model.model.save_checkpoint(policy_output_path)
            self.ppo_algorithm.value_model.model.save_checkpoint(value_output_path)

            # Save the tokenizer separately (it's not part of the DeepSpeed engine)
            self.tokenizer_wrapper.tokenizer.save_pretrained(policy_output_path)
            self.tokenizer_wrapper.tokenizer.save_pretrained(value_output_path)

            logger.info(f"PPO policy model saved to {policy_output_path}")
            logger.info(f"PPO value model saved to {value_output_path}")

        logger.info("PPO training stage completed.")


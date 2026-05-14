import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import math
import logging
from typing import Dict, Any, Tuple, Optional, Callable, Union, List

# Local imports
from config import Config
from model import HiMARModel
from data import DataModule
from utils import _mask_tokens, get_noise_scheduler, _add_noise_to_latents, EMAModel
from evaluation import Evaluator # Import Evaluator here for validation

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class Trainer:
    """
    Orchestrates the training process for the Hi-MAR model, including optimization,
    learning rate scheduling, EMA updates, and periodic validation.
    """
    def __init__(
        self,
        model: HiMARModel,
        data_module: DataModule,
        global_config: Dict[str, Any], # Full global config from config.py
        device: str
    ):
        """
        Initializes the Trainer.

        Args:
            model: The HiMARModel instance to be trained.
            data_module: An instance of DataModule providing data loaders.
            global_config: The full loaded configuration dictionary from config.py.
            device: The computational device ('cuda' or 'cpu').
        """
        self.model = model.to(device)
        self.data_module = data_module
        self.global_config = global_config
        self.device = device
        self.global_step = 0
        
        # Get the learned mask token embedding from the model
        # Needs to be on the correct device as well
        self.mask_token_embedding: torch.Tensor = self.model.get_mask_token_embedding().to(self.device)

        # Determine the active training task (ImageNet or MS-COCO)
        if global_config['training']['imagenet']['enabled']:
            self.task_type = 'imagenet'
        elif global_config['training']['mscoco']['enabled']:
            self.task_type = 'mscoco'
        else:
            raise ValueError("No training task (imagenet or mscoco) is enabled in config.yaml. Please set 'enabled: true' for one task.")
        
        self.training_cfg: Dict[str, Any] = Config.get_training_config(self.task_type)
        self.generation_cfg: Dict[str, Any] = Config.get_generation_config()
        self.evaluation_cfg: Dict[str, Any] = Config.get_evaluation_config(self.task_type)

        # Get tokenizer latent dimensions for masking (using data_module's tokenizer)
        self.high_res_latent_h: int
        self.high_res_latent_w: int
        self.low_res_latent_h: int
        self.low_res_latent_w: int
        self.high_res_latent_h, self.high_res_latent_w = self.data_module.tokenizer.get_latent_hw('high')
        self.low_res_latent_h, self.low_res_latent_w = self.data_module.tokenizer.get_latent_hw('low')

        # Optimizer setup
        optimizer_params = self.training_cfg['optimizer']
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.training_cfg['learning_rate'],
            betas=optimizer_params['betas'],
            eps=optimizer_params['eps'],
            weight_decay=optimizer_params['weight_decay']
        )

        # Learning Rate Scheduler setup
        self.lr_scheduler = self._get_lr_scheduler()

        # EMA Model setup
        self.ema_model = EMAModel(self.model, decay=self.training_cfg['ema_momentum'])

        # Noise Scheduler for diffusion process
        self.noise_scheduler_num_timesteps: int = self.generation_cfg.get("num_train_timesteps", 1000) # Default to 1000
        self.noise_scheduler_fn: Callable = get_noise_scheduler(
            self.generation_cfg['noise_scheduler_type'],
            num_train_timesteps=self.noise_scheduler_num_timesteps
        )
        
        logger.info(f"Trainer initialized for {self.task_type}.")
        logger.info(f"Optimizer: {self.optimizer}")
        logger.info(f"Initial Learning Rate: {self.training_cfg['learning_rate']:.6f}")
        logger.info(f"EMA Momentum: {self.training_cfg['ema_momentum']}")

    def _get_lr_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Creates and returns the learning rate scheduler based on the configuration.
        Handles both epoch-based (ImageNet) and step-based (MS-COCO) warmups.
        """
        train_dataloader = self.data_module.get_train_dataloader()
        steps_per_epoch = len(train_dataloader)
        max_train_steps = self.training_cfg['epochs'] * steps_per_epoch

        warmup_steps = 0
        if self.training_cfg.get("warmup_epochs") is not None:
            warmup_steps = self.training_cfg["warmup_epochs"] * steps_per_epoch
        elif self.training_cfg.get("warmup_steps") is not None:
            warmup_steps = self.training_cfg["warmup_steps"]
        
        if warmup_steps > max_train_steps:
             logger.warning(f"Warmup steps ({warmup_steps}) are greater than total training steps ({max_train_steps}). "
                            "Warmup will be capped at total steps.")
             warmup_steps = max_train_steps # Cap warmup to total steps if it exceeds

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            # Constant LR after warmup, as typically used for Hi-MAR ImageNet and MS-COCO
            return 1.0

        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        return scheduler

    def _compute_loss(self, predicted_noise: torch.Tensor, target_noise: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Mean Squared Error (MSE) loss between predicted and target noise.
        """
        return F.mse_loss(predicted_noise, target_noise)

    def _train_step(self, batch: Dict[str, Any], epoch: int) -> Dict[str, float]:
        """
        Executes a single training step for a given batch, including forward/backward passes
        for both phases, optimizer updates, and EMA update.
        """
        self.model.train() # Ensure model is in training mode
        
        high_res_images: torch.Tensor = batch['high_res_image'].to(self.device)
        low_res_images: torch.Tensor = batch['low_res_image'].to(self.device)
        
        conditions: Union[torch.Tensor, int] = batch['conditions']
        # If conditions are class IDs (int), they remain on CPU; model handles embedding.
        # If conditions are text embeddings (tensor), move them to device.
        if isinstance(conditions, torch.Tensor):
            conditions = conditions.to(self.device)

        batch_size: int = high_res_images.shape[0]

        # Tokenize images (encode returns (B, N_tokens, latent_channels))
        high_res_latents: torch.Tensor = self.data_module.tokenizer.encode(high_res_images, 'high')
        low_res_latents: torch.Tensor = self.data_module.tokenizer.encode(low_res_images, 'low')

        # Sample timesteps for diffusion
        timesteps: torch.Tensor = torch.randint(
            0, self.noise_scheduler_num_timesteps, (batch_size,), device=self.device
        ).long()
        
        # --- Phase 1: Low-resolution prediction ---
        self.optimizer.zero_grad() # Zero gradients for this optimization step

        # Add noise to original low-res latents to get x_i^t and the noise epsilon
        noisy_low_res_latents, epsilon_low = _add_noise_to_latents(
            low_res_latents, timesteps, self.noise_scheduler_fn
        )
        
        # Mask noisy low-res latents to create input for the transformer
        masked_low_res_latents, _ = _mask_tokens(
            noisy_low_res_latents.clone(), # Clone to avoid modifying original noisy_low_res_latents in place
            self.training_cfg["phase1_masking_strategy"],
            self.training_cfg["phase1_masking_params"],
            self.mask_token_embedding,
            return_mask=False
        )

        # Forward pass for Phase 1
        predicted_noise_ph1: torch.Tensor
        low_res_transformer_output: torch.Tensor # This will be the pivot for Phase 2
        predicted_noise_ph1, low_res_transformer_output = self.model.forward_phase1(
            masked_low_res_latents, conditions, timesteps, scale_id=0
        )
        loss_ph1: torch.Tensor = self._compute_loss(predicted_noise_ph1, epsilon_low)
        loss_ph1.backward()

        # --- Phase 2: High-resolution prediction ---

        # Add noise to original high-res latents
        noisy_high_res_latents, epsilon_high = _add_noise_to_latents(
            high_res_latents, timesteps, self.noise_scheduler_fn
        )

        # Mask noisy high-res latents to create input for the transformer
        masked_high_res_latents, _ = _mask_tokens(
            noisy_high_res_latents.clone(), # Clone to avoid modifying original noisy_high_res_latents
            self.training_cfg["phase2_masking_strategy"],
            self.training_cfg["phase2_masking_params"],
            self.mask_token_embedding,
            return_mask=False
        )
        
        # Forward pass for Phase 2, using low_res_transformer_output as pivots
        # `low_res_transformer_output` is detached to ensure no gradients flow back to Phase 1's computation
        predicted_noise_ph2: torch.Tensor = self.model.forward_phase2(
            masked_high_res_latents, # input to transformer
            noisy_high_res_latents,  # input to diffusion transformer head (y^t in paper)
            low_res_transformer_output.detach(), # Pivots from Phase 1, detached
            conditions, timesteps, scale_id=1
        )
        loss_ph2: torch.Tensor = self._compute_loss(predicted_noise_ph2, epsilon_high)
        loss_ph2.backward()

        # Optimizer step
        self.optimizer.step()
        
        # LR scheduler step (per-step, not per-epoch)
        # Note: If gradient_accumulation_steps > 1, this needs adjustment.
        # For now, assuming gradient_accumulation_steps=1 as per config default.
        self.lr_scheduler.step()
        
        # Update EMA model
        self.ema_model.update(self.model)

        self.global_step += 1
        return {
            'total_loss': (loss_ph1 + loss_ph2).item(),
            'loss_ph1': loss_ph1.item(), 
            'loss_ph2': loss_ph2.item(),
            'lr': self.optimizer.param_groups[0]['lr']
        }

    def train(self) -> None:
        """
        Main training loop. Iterates through epochs, performs training steps,
        logs progress, and triggers validation.
        """
        train_loader: DataLoader = self.data_module.get_train_dataloader()
        
        total_epochs: int = self.training_cfg['epochs']
        log_every_n_steps: int = self.training_cfg['log_every_n_steps']
        validate_every_n_epochs: int = self.training_cfg['validate_every_n_epochs']
        output_dir: str = self.global_config['output_dir']
        project_name: str = self.global_config['project_name']
        
        # Create checkpoint directory
        checkpoint_dir: str = os.path.join(output_dir, project_name, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        logger.info("Starting training...")
        for epoch in range(total_epochs):
            logger.info(f"Epoch {epoch+1}/{total_epochs}")
            epoch_total_losses: List[float] = []
            epoch_losses_ph1: List[float] = []
            epoch_losses_ph2: List[float] = []

            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
                step_metrics: Dict[str, float] = self._train_step(batch, epoch)
                epoch_total_losses.append(step_metrics['total_loss'])
                epoch_losses_ph1.append(step_metrics['loss_ph1'])
                epoch_losses_ph2.append(step_metrics['loss_ph2'])

                if self.global_step % log_every_n_steps == 0:
                    avg_loss_ph1: float = sum(epoch_losses_ph1[-log_every_n_steps:]) / log_every_n_steps
                    avg_loss_ph2: float = sum(epoch_losses_ph2[-log_every_n_steps:]) / log_every_n_steps
                    avg_total_loss: float = sum(epoch_total_losses[-log_every_n_steps:]) / log_every_n_steps
                    logger.info(
                        f"Step {self.global_step}, LR: {step_metrics['lr']:.6f}, "
                        f"Total Loss: {avg_total_loss:.4f}, Phase1 Loss: {avg_loss_ph1:.4f}, "
                        f"Phase2 Loss: {avg_loss_ph2:.4f}"
                    )
            
            if epoch_total_losses: # Avoid division by zero if an epoch has no batches
                avg_epoch_total_loss: float = sum(epoch_total_losses) / len(epoch_total_losses)
                logger.info(f"Epoch {epoch+1} finished. Avg Total Loss: {avg_epoch_total_loss:.4f}")

            # Save checkpoint and run validation periodically
            if (epoch + 1) % validate_every_n_epochs == 0 or (epoch + 1) == total_epochs:
                self.save_checkpoint(epoch + 1, checkpoint_dir)
                val_metrics: Dict[str, float] = self.evaluate_validation(epoch + 1)
                metric_str: str = ", ".join([f"{k}: {v:.4f}" for k,v in val_metrics.items()])
                logger.info(f"Validation Epoch {epoch+1}: {metric_str}")
        
        logger.info("Training complete.")

    def save_checkpoint(self, epoch: int, checkpoint_dir: str):
        """Saves the model and EMA model checkpoints."""
        model_path: str = os.path.join(checkpoint_dir, f"model_epoch_{epoch:04d}.pt")
        ema_model_path: str = os.path.join(checkpoint_dir, f"ema_model_epoch_{epoch:04d}.pt")
        
        torch.save(self.model.state_dict(), model_path)
        torch.save(self.ema_model.ema_model.state_dict(), ema_model_path)
        logger.info(f"Saved model checkpoint to {model_path}")
        logger.info(f"Saved EMA model checkpoint to {ema_model_path}")

    @torch.no_grad()
    def evaluate_validation(self, epoch: int) -> Dict[str, float]:
        """
        Performs validation using the EMA model and returns evaluation metrics.
        The paper evaluates on 50K generated samples for ImageNet. For faster
        intermediate validation, this might generate fewer samples if needed,
        or trigger the full evaluation pipeline if configured.
        """
        self.model.eval() # Set base model to eval mode
        self.ema_model.ema_model.eval() # Ensure EMA model is in eval mode

        logger.info(f"Running validation for epoch {epoch}...")

        # Initialize Generator with the EMA model for evaluation
        generator = Generator(
            model=self.ema_model.ema_model, # Use EMA model for generation
            tokenizer=self.data_module.tokenizer,
            clip_encoder=self.data_module.clip_encoder,
            config=self.global_config, # Pass full global config, Generator will extract its own
            device=self.device
        )

        evaluator = Evaluator(
            generator=generator,
            config=self.global_config, # Pass full global config, Evaluator will extract its own
            device=self.device
        )
        
        metrics: Dict[str, float] = {}
        if self.task_type == 'imagenet':
            eval_cfg = Config.get_evaluation_config('imagenet')
            num_samples: int = eval_cfg['num_samples']
            
            # For class-conditional ImageNet, generate for all 1000 classes.
            imagenet_class_ids: List[int] = list(range(1000))
            
            # Evaluate with CFG ON for both phases (as per paper's interpretation of CFG for evaluation)
            if eval_cfg['eval_with_cfg']:
                logger.info("Evaluating ImageNet with CFG...")
                current_metrics: Dict[str, float] = evaluator.evaluate_imagenet(
                    conditions=imagenet_class_ids,
                    num_samples=num_samples,
                    guidance_scale=self.generation_cfg['guidance_scale']
                )
                metrics.update({f"w_cfg_{k}": v for k, v in current_metrics.items()})
                
            # Evaluate without CFG for dense tokens, while keeping CFG for Phase 1 as per paper.
            if eval_cfg['eval_without_cfg']:
                logger.info("Evaluating ImageNet without CFG for dense tokens (Phase 2)...")
                current_metrics: Dict[str, float] = evaluator.evaluate_imagenet(
                    conditions=imagenet_class_ids,
                    num_samples=num_samples,
                    guidance_scale=0.0 # guidance_scale=0 effectively turns off CFG for Phase 2
                )
                metrics.update({f"wo_cfg_{k}": v for k, v in current_metrics.items()})

        elif self.task_type == 'mscoco':
            eval_cfg = Config.get_evaluation_config('mscoco')
            num_samples: int = eval_cfg['num_samples']
            
            # Load validation prompts from file
            evaluation_prompts_file: str = eval_cfg['evaluation_prompts_file']
            if not os.path.exists(evaluation_prompts_file):
                logger.error(f"MS-COCO evaluation prompts file not found: {evaluation_prompts_file}. Skipping MS-COCO evaluation.")
                return {"fid": float('nan'), "t2icompbench": float('nan')}
            with open(evaluation_prompts_file, 'r') as f:
                prompts_full: List[str] = json.load(f)
            prompts: List[str] = prompts_full[:num_samples] # Take `num_samples` prompts

            if eval_cfg['eval_with_cfg']:
                logger.info("Evaluating MS-COCO with CFG...")
                current_metrics: Dict[str, float] = evaluator.evaluate_mscoco(
                    prompts=prompts,
                    num_samples=num_samples,
                    guidance_scale=self.generation_cfg['guidance_scale']
                )
                metrics.update({f"w_cfg_{k}": v for k, v in current_metrics.items()})
            
            # The paper's Table 3 for MS-COCO only shows FID, implying CFG is likely used.
            # No 'w/o CFG' is explicitly discussed or presented for MS-COCO in the paper.
        
        return metrics

# Import Generator here to resolve potential circular dependency, as Evaluator also imports Generator.
import json # Used for loading prompts in evaluate_validation for MS-COCO
from generator import Generator


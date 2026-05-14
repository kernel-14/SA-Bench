"""
NaViL Training Module.

Implements the three-stage training process described in Section 4.2:

Stage 1: Multi-modal Generative Pre-training
  - S1.1: 500M image-text pairs, freeze textual parameters, train vision-specific 
          parameters only (visual encoder, MLP projector, MoE visual experts)
  - S1.2: 185M high-quality data, unfreeze self-attention textual parameters

Stage 2: Supervised Fine-tuning
  - 68M high-quality multimodal data, all parameters unfrozen

Training hyperparameters from Table 7 (NaViL-2B):
  - Optimizer: AdamW (β1=0.9, β2=0.95, eps=1e-8)
  - Precision: bfloat16
  - Weight decay: 0.05 (S1.1), 0.1 (S1.2), 0.01 (S2)
  - Learning rate: 5e-5 (S1), 2e-5 (S2)
  - LR schedule: constant with warm-up (S1), cosine decay (S2)
  - Warm-up steps: 200
  - Global batch size: 7000 (S1.1), 4614 (S1.2)
  - Max sequence length: 16384
  - Max image patches: 4096
"""

import os
import math
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .model import NaViLModel, NaViLConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training NaViL."""
    
    # Stage 1.1: Large-scale multimodal pretraining
    s1_1_steps: int = 121887  # ~500M samples / 7000 batch size * epochs
    s1_1_batch_size: int = 7000
    s1_1_learning_rate: float = 5e-5
    s1_1_weight_decay: float = 0.05
    s1_1_freeze_text: bool = True
    
    # Stage 1.2: High-quality alignment
    s1_2_steps: int = 40000
    s1_2_batch_size: int = 4614
    s1_2_learning_rate: float = 5e-5
    s1_2_weight_decay: float = 0.1
    s1_2_unfreeze_attention: bool = True
    
    # Stage 2: Supervised fine-tuning
    s2_steps: int = 30000
    s2_batch_size: int = 2234
    s2_learning_rate: float = 2e-5
    s2_weight_decay: float = 0.01
    
    # General
    warmup_steps: int = 200
    max_sequence_length: int = 16384
    max_image_patches: int = 4096
    gradient_accumulation_steps: int = 1
    precision: str = 'bfloat16'
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    
    # Output
    output_dir: str = './checkpoints'
    save_steps: int = 5000
    logging_steps: int = 100


class NaViLTrainer:
    """
    Trainer for the three-stage NaViL training process.
    """
    
    def __init__(
        self,
        model: NaViLModel,
        config: TrainingConfig,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.config = config
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.model.to(self.device)
        self.global_step = 0
        
    def train_stage1_1(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        num_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Stage 1.1: Multi-modal Generative Pre-training (large-scale).
        
        Freeze textual parameters, train only:
        - Visual encoder
        - MLP projector
        - MoE visual experts
        
        Uses 500M web-scale image-text pairs.
        LR schedule: constant with warm-up.
        """
        logger.info("=" * 60)
        logger.info("Stage 1.1: Multi-modal Generative Pre-training")
        logger.info("=" * 60)
        
        if num_steps is None:
            num_steps = self.config.s1_1_steps
            
        # Freeze textual parameters
        self._set_text_params_requires_grad(False)
        self._set_visual_params_requires_grad(True)
        
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.s1_1_learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_eps,
            weight_decay=self.config.s1_1_weight_decay,
        )
        
        scheduler = self._create_constant_with_warmup_scheduler(optimizer, num_steps)
        
        return self._train_loop(
            train_dataloader, 
            optimizer, 
            scheduler, 
            num_steps,
            val_dataloader,
            stage_name="S1.1",
        )
    
    def train_stage1_2(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        num_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Stage 1.2: High-quality alignment pre-training.
        
        Trains on 185M high-quality multimodal + language data.
        Unfreezes self-attention textual parameters.
        LR schedule: constant with warm-up.
        """
        logger.info("=" * 60)
        logger.info("Stage 1.2: High-quality Alignment Pre-training")
        logger.info("=" * 60)
        
        if num_steps is None:
            num_steps = self.config.s1_2_steps
            
        # Unfreeze attention parameters
        self._set_all_params_requires_grad(True)
        
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.s1_2_learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_eps,
            weight_decay=self.config.s1_2_weight_decay,
        )
        
        scheduler = self._create_constant_with_warmup_scheduler(optimizer, num_steps)
        
        return self._train_loop(
            train_dataloader,
            optimizer,
            scheduler,
            num_steps,
            val_dataloader,
            stage_name="S1.2",
        )
    
    def train_stage2(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        num_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Stage 2: Supervised Fine-tuning.
        
        Trains on 68M high-quality multimodal data.
        All parameters unfrozen.
        LR schedule: cosine decay.
        """
        logger.info("=" * 60)
        logger.info("Stage 2: Supervised Fine-tuning")
        logger.info("=" * 60)
        
        if num_steps is None:
            num_steps = self.config.s2_steps
            
        # All parameters trainable
        self._set_all_params_requires_grad(True)
        
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.s2_learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_eps,
            weight_decay=self.config.s2_weight_decay,
        )
        
        scheduler = self._create_cosine_scheduler(optimizer, num_steps)
        
        return self._train_loop(
            train_dataloader,
            optimizer,
            scheduler,
            num_steps,
            val_dataloader,
            stage_name="S2",
        )
    
    def train_full(
        self,
        s1_1_dataloader: DataLoader,
        s1_2_dataloader: Optional[DataLoader] = None,
        s2_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        """
        Run the full three-stage training pipeline.
        """
        results = {}
        
        # Stage 1.1
        results['s1_1'] = self.train_stage1_1(s1_1_dataloader, val_dataloader)
        
        # Stage 1.2
        if s1_2_dataloader is not None:
            results['s1_2'] = self.train_stage1_2(s1_2_dataloader, val_dataloader)
        
        # Stage 2
        if s2_dataloader is not None:
            results['s2'] = self.train_stage2(s2_dataloader, val_dataloader)
        
        return results
    
    def _train_loop(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: LambdaLR,
        num_steps: int,
        val_dataloader: Optional[DataLoader],
        stage_name: str,
    ) -> Dict[str, Any]:
        """Main training loop."""
        self.model.train()
        
        total_loss = 0.0
        log_interval = self.config.logging_steps
        metrics = {'loss_history': [], 'val_loss_history': []}
        
        data_iter = iter(dataloader)
        
        for step in range(num_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs['loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item()
            self.global_step += 1
            
            # Logging
            if step % log_interval == 0 and step > 0:
                avg_loss = total_loss / log_interval
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"[{stage_name}] Step {step}/{num_steps} | "
                    f"Loss: {avg_loss:.4f} | LR: {lr:.2e}"
                )
                metrics['loss_history'].append((step, avg_loss))
                total_loss = 0.0
            
            # Validation
            if val_dataloader is not None and step % (log_interval * 10) == 0:
                val_loss = self._validate(val_dataloader)
                metrics['val_loss_history'].append((step, val_loss))
                logger.info(f"[{stage_name}] Step {step} | Val Loss: {val_loss:.4f}")
            
            # Save checkpoint
            if step % self.config.save_steps == 0 and step > 0:
                self._save_checkpoint(stage_name, step)
        
        return metrics
    
    def _validate(self, dataloader: DataLoader) -> float:
        """Compute validation loss."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs['loss'].item()
                num_batches += 1
                
                # Limit validation to 100 batches
                if num_batches >= 100:
                    break
        
        self.model.train()
        return total_loss / max(num_batches, 1)
    
    def _set_text_params_requires_grad(self, requires_grad: bool):
        """Freeze/unfreeze text (LLM) parameters."""
        # Visual encoder and connector always controllable
        for p in self.model.visual_encoder.parameters():
            p.requires_grad = not requires_grad  # opposite: freeze text → train visual
        
        for p in self.model.connector.parameters():
            p.requires_grad = not requires_grad
        
        # MoE text experts
        for layer in self.model.llm_layers:
            if hasattr(layer.attn, 'q_proj_text'):
                layer.attn.q_proj_text.weight.requires_grad = requires_grad
                layer.attn.k_proj_text.weight.requires_grad = requires_grad
                layer.attn.v_proj_text.weight.requires_grad = requires_grad
                layer.attn.o_proj_text.weight.requires_grad = requires_grad
            if hasattr(layer.attn, 'q_proj_visual'):
                layer.attn.q_proj_visual.weight.requires_grad = not requires_grad
                layer.attn.k_proj_visual.weight.requires_grad = not requires_grad
                layer.attn.v_proj_visual.weight.requires_grad = not requires_grad
                layer.attn.o_proj_visual.weight.requires_grad = not requires_grad
                
            if hasattr(layer.ffn, 'gate_proj_text'):
                layer.ffn.gate_proj_text.weight.requires_grad = requires_grad
                layer.ffn.up_proj_text.weight.requires_grad = requires_grad
                layer.ffn.down_proj_text.weight.requires_grad = requires_grad
            if hasattr(layer.ffn, 'gate_proj_visual'):
                layer.ffn.gate_proj_visual.weight.requires_grad = not requires_grad
                layer.ffn.up_proj_visual.weight.requires_grad = not requires_grad
                layer.ffn.down_proj_visual.weight.requires_grad = not requires_grad
    
    def _set_visual_params_requires_grad(self, requires_grad: bool):
        """Set requires_grad for visual-specific parameters."""
        for p in self.model.visual_encoder.parameters():
            p.requires_grad = requires_grad
        for p in self.model.connector.parameters():
            p.requires_grad = requires_grad
            
        # MoE visual experts
        for layer in self.model.llm_layers:
            if hasattr(layer.attn, 'q_proj_visual'):
                layer.attn.q_proj_visual.weight.requires_grad = requires_grad
                layer.attn.k_proj_visual.weight.requires_grad = requires_grad
                layer.attn.v_proj_visual.weight.requires_grad = requires_grad
                layer.attn.o_proj_visual.weight.requires_grad = requires_grad
            if hasattr(layer.ffn, 'gate_proj_visual'):
                layer.ffn.gate_proj_visual.weight.requires_grad = requires_grad
                layer.ffn.up_proj_visual.weight.requires_grad = requires_grad
                layer.ffn.down_proj_visual.weight.requires_grad = requires_grad
    
    def _set_all_params_requires_grad(self, requires_grad: bool):
        """Set requires_grad for all parameters."""
        for p in self.model.parameters():
            p.requires_grad = requires_grad
    
    def _create_constant_with_warmup_scheduler(
        self, optimizer: torch.optim.Optimizer, num_steps: int
    ) -> LambdaLR:
        """Create constant LR with linear warmup."""
        warmup_steps = self.config.warmup_steps
        
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0
        
        return LambdaLR(optimizer, lr_lambda)
    
    def _create_cosine_scheduler(
        self, optimizer: torch.optim.Optimizer, num_steps: int
    ) -> LambdaLR:
        """Create cosine decay LR with linear warmup."""
        warmup_steps = self.config.warmup_steps
        
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, num_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        return LambdaLR(optimizer, lr_lambda)
    
    def _save_checkpoint(self, stage_name: str, step: int):
        """Save model checkpoint."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            self.config.output_dir,
            f"navil_{stage_name}_step{step}.pt"
        )
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'global_step': self.global_step,
            'config': self.config,
        }, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        logger.info(f"Checkpoint loaded from {checkpoint_path}")

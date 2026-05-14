"""
This module provides a `OptimizerSchedulerFactory` class responsible for creating
and configuring the AdamW optimizer and learning rate schedulers for different
training phases (pretraining, SFT, and DPO) of the OLMoE model.
"""

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import math
from typing import Tuple, Dict, Any, List

# Local imports from the project structure
from config import Config
from model.olmoe_model import OLMoEModel


class OptimizerSchedulerFactory:
    """
    A factory class for creating optimizers and learning rate schedulers
    tailored for the OLMoE model's pretraining and adaptation phases.
    """

    @staticmethod
    def create_pretrain_optimizer_and_scheduler(
        model: OLMoEModel, config: Config
    ) -> Tuple[optim.Optimizer, lr_scheduler._LRScheduler]:
        """
        Creates the AdamW optimizer and a combined learning rate scheduler
        for the pretraining phase. The scheduler implements a three-phase
        learning rate schedule: linear warmup, cosine decay, and linear annealing.

        Args:
            model: The OLMoEModel instance.
            config: The global configuration object, containing training and data settings.

        Returns:
            A tuple containing the initialized AdamW optimizer and the
            configured learning rate scheduler.
        """
        training_config = config.training
        data_config = config.data

        # Initialize AdamW optimizer
        # As per the paper (§4.2.4), weight decay is applied to all parameters.
        optimizer = optim.AdamW(
            model.parameters(), # Using model.parameters() as all params share same weight decay
            lr=training_config.learning_rate_peak,
            betas=(training_config.adam_beta1, training_config.adam_beta2),
            eps=training_config.adam_epsilon,
            weight_decay=training_config.weight_decay,
        )

        # Calculate total number of training steps based on total tokens and effective batch size
        effective_tokens_per_step = training_config.global_batch_size_samples * data_config.max_seq_len
        if effective_tokens_per_step <= 0:
            raise ValueError(
                "Effective tokens per step must be positive to calculate num_training_steps. "
                "Check global_batch_size_samples and max_seq_len in config."
            )
        num_training_steps = int(training_config.total_tokens / effective_tokens_per_step)

        # Calculate annealing phase steps and its start step
        annealing_steps = int(training_config.annealing_tokens / effective_tokens_per_step)
        # Ensure annealing steps are not negative or exceed total steps
        annealing_steps = max(1, min(annealing_steps, num_training_steps))
        annealing_start_step = num_training_steps - annealing_steps

        # Define the lambda function for the learning rate schedule
        def lr_lambda(current_step: int) -> float:
            # Phase 1: Linear Warmup
            if current_step < training_config.warmup_steps:
                return float(current_step) / max(1.0, float(training_config.warmup_steps))

            # After warmup, if current_step is before annealing starts
            if current_step < annealing_start_step:
                # Phase 2: Cosine Decay
                cosine_decay_duration = annealing_start_step - training_config.warmup_steps
                
                # Handle cases where cosine decay phase is skipped or very short
                if cosine_decay_duration <= 0:
                    # If no cosine decay duration, immediately drop to min_lr_cosine
                    current_lr_value = training_config.learning_rate_min
                else:
                    progress_in_cosine_decay = (float(current_step) - training_config.warmup_steps) / cosine_decay_duration
                    # Clamp progress to [0, 1] for robustness
                    progress_in_cosine_decay = max(0.0, min(1.0, progress_in_cosine_decay))
                    
                    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress_in_cosine_decay))
                    
                    current_lr_value = training_config.learning_rate_min + \
                                       (training_config.learning_rate_peak - training_config.learning_rate_min) * cosine_factor
                
                # Return multiplier relative to peak_lr for LambdaLR
                return current_lr_value / training_config.learning_rate_peak

            else:
                # Phase 3: Linear Annealing
                # The learning rate at the start of annealing is the minimum learning rate
                # achieved at the end of the cosine decay phase.
                lr_at_annealing_start = training_config.learning_rate_min
                
                # Guard against division by zero if annealing_steps is 0
                if annealing_steps <= 0:
                    return training_config.annealing_min_lr_final / training_config.learning_rate_peak

                progress_in_linear_annealing = (float(current_step) - annealing_start_step) / annealing_steps
                # Clamp progress to [0, 1] for robustness
                progress_in_linear_annealing = max(0.0, min(1.0, progress_in_linear_annealing))
                
                current_lr_value = lr_at_annealing_start * (1 - progress_in_linear_annealing) + \
                                   training_config.annealing_min_lr_final * progress_in_linear_annealing
                
                # Return multiplier relative to peak_lr for LambdaLR
                return current_lr_value / training_config.learning_rate_peak

        # Create the learning rate scheduler
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return optimizer, scheduler

    @staticmethod
    def create_sft_optimizer_and_scheduler(
        model: OLMoEModel, config: Config, num_sft_steps: int
    ) -> Tuple[optim.Optimizer, lr_scheduler._LRScheduler]:
        """
        Creates the AdamW optimizer and a constant learning rate scheduler
        for the Supervised Fine-Tuning (SFT) phase.

        Args:
            model: The OLMoEModel instance.
            config: The global configuration object.
            num_sft_steps: The total number of steps for SFT.

        Returns:
            A tuple containing the initialized AdamW optimizer and the
            configured learning rate scheduler.
        """
        training_config = config.training

        optimizer = optim.AdamW(
            model.parameters(),
            lr=training_config.sft_lr,
            betas=(training_config.adam_beta1, training_config.adam_beta2),
            eps=training_config.adam_epsilon,
            weight_decay=training_config.weight_decay,
        )

        # The paper specifies a "constant learning rate" for SFT.
        # ConstantLR ensures the initial LR is maintained for `total_iters`.
        scheduler = lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=num_sft_steps
        )
        return optimizer, scheduler

    @staticmethod
    def create_dpo_optimizer_and_scheduler(
        model: OLMoEModel, config: Config, num_dpo_steps: int
    ) -> Tuple[optim.Optimizer, lr_scheduler._LRScheduler]:
        """
        Creates the AdamW optimizer and a constant learning rate scheduler
        for the Direct Preference Optimization (DPO) phase.

        Args:
            model: The OLMoEModel instance.
            config: The global configuration object.
            num_dpo_steps: The total number of steps for DPO.

        Returns:
            A tuple containing the initialized AdamW optimizer and the
            configured learning rate scheduler.
        """
        training_config = config.training

        optimizer = optim.AdamW(
            model.parameters(),
            lr=training_config.dpo_lr,
            betas=(training_config.adam_beta1, training_config.adam_beta2),
            eps=training_config.adam_epsilon,
            weight_decay=training_config.weight_decay,
        )

        # The paper implies a constant learning rate for DPO.
        # ConstantLR ensures the initial LR is maintained for `total_iters`.
        scheduler = lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=num_dpo_steps
        )
        return optimizer, scheduler


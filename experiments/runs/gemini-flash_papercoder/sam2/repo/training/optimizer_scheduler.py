import torch
import torch.nn as nn
import torch.optim as optim
import math
from typing import Any, Dict, List, Optional, Tuple

# Placeholder for Config type hint to avoid circular import with config.py
# In a real project, this would be 'from config import Config'
Config = Any


class OptimizerScheduler:
    """
    Manages the AdamW optimizer and implements a learning rate schedule
    combining warmup, reciprocal square-root decay, and cooldown,
    along with layer-wise decay.
    """

    def __init__(self, model: nn.Module, config: Config, total_steps: int, stage_config_key: str):
        """
        Initializes the optimizer and sets up the LR schedule parameters.

        Args:
            model (nn.Module): The torch.nn.Module (e.g., SAM2Model) whose parameters
                               will be optimized.
            config (Config): An instance of the Config class, providing access to
                             all hyperparameters.
            total_steps (int): The total number of optimization steps (iterations)
                               for the current training stage.
            stage_config_key (str): A string (e.g., "training.pretrain", "training.full_train",
                                    "training.finetune") used to retrieve the specific training
                                    stage's hyperparameters from the config object.
        """
        self._config = config
        self.total_steps: int = total_steps
        self._current_step: int = 0
        self.stage_config_key: str = stage_config_key

        # Retrieve stage-specific hyperparameters, providing default values
        self.base_lr: float = self._config.get(f"{self.stage_config_key}.learning_rate", 0.0004)
        self.warmup_iters: int = self._config.get(f"{self.stage_config_key}.warmup_iters", 1000)
        self.cooldown_iters: int = self._config.get(f"{self.stage_config_key}.cooldown_iters", 5000)
        self.lr_schedule_timescale: int = self._config.get(f"{self.stage_config_key}.lr_schedule_timescale", 1000)
        self.layer_wise_decay_factor: float = self._config.get(f"{self.stage_config_key}.layer_wise_decay", 1.0)
        self.optimizer_momentum: List[float] = self._config.get(f"{self.stage_config_key}.optimizer_momentum", [0.9, 0.999])
        self.weight_decay: float = self._config.get(f"{self.stage_config_key}.weight_decay", 0.1)
        self.freeze_image_encoder: bool = self._config.get(f"{self.stage_config_key}.freeze_image_encoder", False)

        # 1. Parameter Grouping and Layer-wise Decay
        image_encoder_params: List[nn.Parameter] = []
        other_model_params: List[nn.Parameter] = []

        # Iterate through model parameters to separate and apply freezing/decay logic
        for name, param in model.named_parameters():
            # Skip parameters that are already not trainable (e.g., frozen by ImageEncoder's own init)
            if not param.requires_grad:
                continue
            
            if name.startswith('image_encoder.'):
                if self.freeze_image_encoder:
                    # If this stage explicitly requires freezing image encoder, set requires_grad to False
                    param.requires_grad = False
                    continue # Do not include frozen parameters in the optimizer
                image_encoder_params.append(param)
            else:
                other_model_params.append(param)

        param_groups: List[Dict[str, Any]] = []

        if image_encoder_params:
            param_groups.append({
                "params": image_encoder_params,
                "initial_lr": self.base_lr,
                "lr_multiplier": self.layer_wise_decay_factor # Apply layer-wise decay for image encoder
            })
        if other_model_params:
            param_groups.append({
                "params": other_model_params,
                "initial_lr": self.base_lr,
                "lr_multiplier": 1.0 # No layer-wise decay for other parameters by default
            })
        
        if not param_groups:
            # If no trainable parameters are found (e.g., all frozen), create a dummy optimizer
            # This prevents errors but indicates nothing will be optimized.
            print("Warning: No trainable parameters found or all parameters are frozen for this stage. Optimizer will be a dummy.")
            self.optimizer = optim.AdamW([torch.tensor(0.0)], lr=0.0) 
            self.base_lr = 0.0 # Set base_lr to 0 to reflect no active learning
            return

        # 2. Optimizer Initialization
        self.optimizer = optim.AdamW(
            param_groups,
            lr=self.base_lr, # This 'lr' will be immediately updated by the scheduler's first step()
            betas=tuple(self.optimizer_momentum),
            weight_decay=self.weight_decay
        )

    def _calculate_lr_multiplier(self) -> float:
        """
        Calculates a global LR scaling factor for the current step,
        combining warmup, reciprocal square-root decay, and cooldown.

        Returns:
            float: The global LR multiplier.
        """
        current_step_f: float = float(self._current_step)

        if current_step_f < self.warmup_iters:
            # Warmup Phase: Linear increase
            if self.warmup_iters == 0:
                multiplier = 1.0 # No warmup, start at full LR (or main schedule start)
            else:
                multiplier = current_step_f / self.warmup_iters
        elif current_step_f >= self.total_steps:
            # Beyond total_steps, LR should be 0 or minimum. Cooldown should lead to this.
            multiplier = 0.0
        elif current_step_f >= (self.total_steps - self.cooldown_iters):
            # Cooldown Phase: Linear decay to zero
            if self.cooldown_iters == 0:
                # If cooldown_iters is 0, this means no explicit cooldown period.
                # The LR should just follow the main schedule up to total_steps and then drop to 0.
                multiplier = 0.0 # Will hit this condition only at current_step == total_steps for cooldown_iters = 0.
            else:
                # Calculate the LR multiplier at the effective start of the cooldown period
                effective_step_at_cooldown_start: float = float(max(1, self.total_steps - self.cooldown_iters))
                
                # Ensure lr_schedule_timescale is not zero for decay calculation
                decay_timescale: float = max(1.0, float(self.lr_schedule_timescale))
                
                # Calculate the LR scale that would have been applied at cooldown start by the main schedule
                initial_cooldown_scale: float = 1.0 / math.sqrt(max(1.0, (effective_step_at_cooldown_start - self.warmup_iters + 1) / decay_timescale))
                
                # Calculate cooldown progress (0.0 at start of cooldown, 1.0 at total_steps)
                cooldown_progress: float = (current_step_f - (self.total_steps - self.cooldown_iters)) / self.cooldown_iters
                
                # Linearly decay from initial_cooldown_scale to 0
                multiplier = initial_cooldown_scale * max(0.0, (1.0 - cooldown_progress))
        else:
            # Reciprocal Square-root Decay Phase (Main Schedule)
            # effective_step_for_decay considers steps after warmup
            effective_step_for_decay: float = float(current_step_f - self.warmup_iters + 1)
            
            # Ensure lr_schedule_timescale is not zero
            decay_timescale = max(1.0, float(self.lr_schedule_timescale))
            
            multiplier = 1.0 / math.sqrt(max(1.0, effective_step_for_decay / decay_timescale))
        
        return multiplier

    def step(self) -> None:
        """
        Updates the learning rates for all parameter groups and performs an optimizer step.
        """
        self._current_step += 1

        if self.base_lr == 0.0: # If nothing is trainable (e.g., dummy optimizer was created)
            self.optimizer.step()
            return

        global_lr_multiplier: float = self._calculate_lr_multiplier()

        for param_group in self.optimizer.param_groups:
            # The final LR for a group is base_lr * global_multiplier * group_specific_multiplier
            param_group['lr'] = param_group['initial_lr'] * global_lr_multiplier * param_group.get('lr_multiplier', 1.0)
        
        self.optimizer.step()

    def zero_grad(self) -> None:
        """
        Zeros the gradients of all optimized parameters.
        """
        self.optimizer.zero_grad()

    def get_lr(self) -> float:
        """
        Returns the current learning rate of the first parameter group.
        Useful for logging.

        Returns:
            float: The current learning rate. Returns 0.0 if no parameter groups are found.
        """
        if self.optimizer.param_groups:
            return self.optimizer.param_groups[0]['lr']
        return 0.0


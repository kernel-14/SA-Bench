"""
Exponential Moving Average (EMA) for model parameters.
"""

import copy
import torch
import torch.nn as nn


class EMA:
    """
    Maintains an exponential moving average of model parameters.
    Used during training to improve generation quality at inference.
    """

    def __init__(self, model, momentum=0.9999):
        """
        Args:
            model: the model to track
            momentum: EMA momentum (higher = slower update)
        """
        self.momentum = momentum
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()

        # Disable gradients for EMA model
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        """Update EMA parameters."""
        # Handle DDP wrapper
        if hasattr(model, 'module'):
            model_params = dict(model.module.named_parameters())
            model_buffers = dict(model.module.named_buffers())
        else:
            model_params = dict(model.named_parameters())
            model_buffers = dict(model.named_buffers())

        ema_params = dict(self.ema_model.named_parameters())
        ema_buffers = dict(self.ema_model.named_buffers())

        for name, param in model_params.items():
            if name in ema_params:
                ema_params[name].mul_(self.momentum).add_(param.data, alpha=1 - self.momentum)

        for name, buffer in model_buffers.items():
            if name in ema_buffers:
                ema_buffers[name].copy_(buffer.data)

    def state_dict(self):
        return {
            'momentum': self.momentum,
            'ema_model': self.ema_model.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.momentum = state_dict['momentum']
        self.ema_model.load_state_dict(state_dict['ema_model'])

    def get_model(self):
        """Return the EMA model for inference."""
        return self.ema_model

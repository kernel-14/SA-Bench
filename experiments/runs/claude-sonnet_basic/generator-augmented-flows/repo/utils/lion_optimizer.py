"""
Lion optimizer from Chen et al. (2023) "Symbolic Discovery of Optimization Algorithms".
Implementation based on https://github.com/lucidrains/lion-pytorch

The paper uses Lion optimizer for all experiments.
"""

import torch
from torch.optim.optimizer import Optimizer


class Lion(Optimizer):
    """
    Lion (EvoLved Sign Momentum) optimizer.
    
    From: "Symbolic Discovery of Optimization Algorithms" (Chen et al., 2023)
    
    Update rule:
        c_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        theta_t = theta_{t-1} - lr * (sign(c_t) + lambda * theta_{t-1})
        m_t = beta2 * m_{t-1} + (1 - beta2) * g_t
    """
    
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        """
        Args:
            params: Model parameters
            lr: Learning rate
            betas: Coefficients for computing running averages of gradient
            weight_decay: Weight decay (L2 penalty)
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Lion does not support sparse gradients')

                state = self.state[p]
                beta1, beta2 = group['betas']

                # Initialize state
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']

                # Weight decay
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Compute update direction
                update = exp_avg * beta1 + grad * (1 - beta1)
                
                # Apply sign update
                p.add_(update.sign_(), alpha=-group['lr'])

                # Update exponential moving average
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss

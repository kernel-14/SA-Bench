"""
Deep Repeated ConvLSTM (DRC) Agent Architecture.

Implements DRC(D,N) as described in the paper (Guez et al., 2019):
- D ConvLSTM layers with N internal ticks per step
- 32 channels, kernel size 3, single-layer zero padding
- Bottom-up skip connections: i_t to all layers
- Top-down skip connections: final layer output to bottom layer on next tick (via addition)
- Pool-and-Inject: spatially pooled h added to each layer's input
- Actor-critic with policy and value heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import numpy as np


class ConvLSTMCell(nn.Module):
    """Standard ConvLSTM cell: gates = Conv(concat(x, h_prev))."""
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding)
        self.hidden_dim = hidden_dim
    
    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, g, o = gates.chunk(4, dim=1)
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next


class PoolAndInject(nn.Module):
    """Mean+max spatial pool, affine, broadcast back to spatial dims."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(2 * hidden_dim, hidden_dim)
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, C, H, W = h.shape
        mean_pooled = h.mean(dim=[2, 3])
        max_pooled = h.amax(dim=[2, 3])
        pooled = torch.cat([mean_pooled, max_pooled], dim=1)
        out = self.linear(pooled)
        return out.unsqueeze(-1).unsqueeze(-1).expand(B, C, H, W)


class DRCLayer(nn.Module):
    """
    Single DRC ConvLSTM layer with pool-and-inject.
    
    All layers share the same structure: 32-channel input, 32-channel hidden.
    Bottom-up: receives i_t (layer 0) or prev layer output (layers 1+).
    Top-down: bottom layer additionally adds final layer's prev tick h (via addition).
    Pool-and-inject: all layers add spatially pooled own prev h.
    """
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        # Input to ConvLSTM: always 32 channels (from below) + hidden (32) = 64 total
        self.lstm = ConvLSTMCell(hidden_dim, hidden_dim)
        self.pool_inject = PoolAndInject(hidden_dim)
    
    def forward(self, x, h_prev, c_prev, top_down=None):
        """
        Args:
            x: (B, 32, H, W) bottom-up input
            h_prev: (B, 32, H, W) previous hidden state
            c_prev: (B, 32, H, W) previous cell state
            top_down: Optional (B, 32, H, W) top-down signal to ADD (not concat)
        """
        lstm_input = x
        if top_down is not None:
            lstm_input = lstm_input + top_down
        lstm_input = lstm_input + self.pool_inject(h_prev)
        return self.lstm(lstm_input, h_prev, c_prev)


class DRCEncoder(nn.Module):
    """Observation encoder: 7 -> 32 channels."""
    def __init__(self, in_channels: int = 7, out_channels: int = 32):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class DRCAgent(nn.Module):
    """
    DRC(D,N) Agent.
    
    Default: DRC(3,3) - 3 ConvLSTM layers, 3 internal ticks per step.
    
    Args:
        D: Number of layers
        N: Internal ticks per step
        hidden_dim: Channels (default 32)
        action_space: Number of actions (default 5)
        input_channels: Observation channels (default 7)
    """
    def __init__(self, D=3, N=3, hidden_dim=32, action_space=5, input_channels=7):
        super().__init__()
        self.D = D
        self.N = N
        self.hidden_dim = hidden_dim
        self.spatial_size = 8
        
        self.encoder = DRCEncoder(input_channels, hidden_dim)
        self.layers = nn.ModuleList([DRCLayer(hidden_dim) for _ in range(D)])
        
        # Output head: concat(final hidden, encoding) -> conv -> relu -> flatten -> heads
        self.output_conv = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1)
        flat_dim = hidden_dim * 8 * 8
        self.policy_head = nn.Linear(flat_dim, action_space)
        self.value_head = nn.Linear(flat_dim, 1)
        
        # State containers
        self.h_t = None
        self.c_t = None
    
    def reset_state(self, batch_size=1, device=None):
        shape = (batch_size, self.hidden_dim, self.spatial_size, self.spatial_size)
        self.h_t = [torch.zeros(*shape, device=device) for _ in range(self.D)]
        self.c_t = [torch.zeros(*shape, device=device) for _ in range(self.D)]
    
    def forward(self, x_t, return_all_states=False):
        B = x_t.shape[0]
        i_t = self.encoder(x_t)  # (B, 32, 8, 8)
        
        if self.h_t is None or self.h_t[0] is None or self.h_t[0].shape[0] != B:
            self.reset_state(B, x_t.device)
        
        all_states = {} if return_all_states else None
        top_down = None  # from final layer's previous tick hidden state
        
        for tick in range(self.N):
            if return_all_states:
                all_states[f'tick_{tick}'] = {}
            
            new_h = [None] * self.D
            new_c = [None] * self.D
            
            for d in range(self.D):
                is_bottom = (d == 0)
                x = i_t if is_bottom else self.h_t[d-1]  # bottom-up input
                td = top_down if is_bottom else None     # top-down only for bottom
                
                h_new, c_new = self.layers[d](x, self.h_t[d], self.c_t[d], td)
                new_h[d] = h_new
                new_c[d] = c_new
                
                if return_all_states:
                    all_states[f'tick_{tick}'][f'layer_{d}'] = c_new.clone()
            
            self.h_t = new_h
            self.c_t = new_c
            top_down = self.h_t[-1]  # final layer output for next tick's top-down
        
        # Output
        h_final = self.h_t[-1]
        combined = torch.cat([h_final, i_t], dim=1)  # (B, 64, 8, 8)
        combined = self.output_conv(combined)         # (B, 32, 8, 8)
        combined = F.relu(combined)
        flat = combined.reshape(B, -1)                 # (B, 2048)
        
        policy_logits = self.policy_head(flat)
        value = self.value_head(flat)
        
        if return_all_states:
            return policy_logits, value, all_states
        return policy_logits, value
    
    def act_greedy(self, x_t):
        with torch.no_grad():
            logits, _ = self.forward(x_t)
        return logits.argmax(dim=-1).item()
    
    def act_sample(self, x_t):
        with torch.no_grad():
            logits, _ = self.forward(x_t)
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, 1).item()


class ResNetBlock(nn.Module):
    """Simplified residual block: conv -> norm -> relu -> conv -> norm -> + -> relu."""
    def __init__(self, channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(1, channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return F.relu(out + residual)


class ResNetAgent(nn.Module):
    """ResNet agent as described in Appendix G: 24 residual blocks, 32 channels."""
    def __init__(self, num_blocks=24, channels=32, action_space=5, input_channels=7):
        super().__init__()
        self.input_conv = nn.Conv2d(input_channels, channels, 3, padding=1)
        self.blocks = nn.ModuleList([ResNetBlock(channels) for _ in range(num_blocks)])
        self.mlp = nn.Linear(channels * 8 * 8, 256)
        self.policy_head = nn.Linear(256, action_space)
        self.value_head = nn.Linear(256, 1)
    
    def forward(self, x_t, return_all_states=False):
        B = x_t.shape[0]
        x = F.relu(self.input_conv(x_t))
        all_states = {} if return_all_states else None
        
        for i, block in enumerate(self.blocks):
            x = block(x)
            if return_all_states:
                all_states[f'layer_{i}'] = x.clone()
        
        flat = x.reshape(B, -1)
        hidden = F.relu(self.mlp(flat))
        policy_logits = self.policy_head(hidden)
        value = self.value_head(hidden)
        
        if return_all_states:
            return policy_logits, value, all_states
        return policy_logits, value

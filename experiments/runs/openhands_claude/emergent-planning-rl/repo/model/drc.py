import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Any

from model.convlstm import DRCEncoder, DRCStack


class DRCAgent(nn.Module):
    """
    Deep Repeated ConvLSTM (DRC) agent as described in Guez et al. (2019).
    
    Architecture:
      1. Convolutional encoder: obs -> encoding i_t (8x8x32)
      2. DRC stack: D ConvLSTM layers, N ticks per step
      3. Output head: concat(h^D_{t,N}, i_t) -> affine -> ReLU -> policy + value
    
    The agent studied in the paper is DRC(3,3):
      D=3 layers, N=3 ticks, 32 channels, kernel 3x3, 8x8 spatial dims.
    """

    def __init__(
        self,
        obs_channels: int = 7,
        num_actions: int = 5,
        num_layers: int = 3,
        num_ticks: int = 3,
        hidden_channels: int = 32,
        encoder_channels: int = 32,
        kernel_size: int = 3,
        padding: int = 1,
        grid_size: int = 8,
        output_hidden_size: int = 256,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_ticks = num_ticks
        self.hidden_channels = hidden_channels
        self.grid_size = grid_size
        self.num_actions = num_actions

        self.encoder = DRCEncoder(
            in_channels=obs_channels,
            out_channels=encoder_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.drc_stack = DRCStack(
            num_layers=num_layers,
            num_ticks=num_ticks,
            encoding_channels=encoder_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            height=grid_size,
            width=grid_size,
        )

        # Output: concat final hidden output h^D_{t,N} with encoding i_t
        # Both are (B, 32, 8, 8) -> flatten -> (B, 32*8*8 + 32*8*8) = (B, 4096)
        concat_size = (hidden_channels + encoder_channels) * grid_size * grid_size
        self.output_fc = nn.Linear(concat_size, output_hidden_size)

        self.policy_head = nn.Linear(output_hidden_size, num_actions)
        self.value_head = nn.Linear(output_hidden_size, 1)

    def forward(
        self,
        obs: torch.Tensor,
        hidden_states: Optional[List[torch.Tensor]] = None,
        cell_states: Optional[List[torch.Tensor]] = None,
        return_all_ticks: bool = False,
    ) -> Dict[str, Any]:
        """
        Args:
            obs: (B, H, W, C) or (B, C, H, W) observation
            hidden_states: list of D tensors (B, C_h, H, W)
            cell_states: list of D tensors (B, C_h, H, W)
            return_all_ticks: whether to return cell states at every tick
        Returns:
            dict with keys: policy_logits, value, hidden_states, cell_states,
                           encoding, all_tick_cell_states (if requested)
        """
        if obs.dim() == 4 and obs.shape[-1] == 7:
            obs = obs.permute(0, 3, 1, 2).contiguous()

        encoding = self.encoder(obs)

        new_hidden, new_cell, all_tick_cells = self.drc_stack(
            encoding, hidden_states, cell_states, return_all_ticks=return_all_ticks
        )

        final_hidden = new_hidden[-1]
        concat = torch.cat([final_hidden, encoding], dim=1)
        concat_flat = concat.view(concat.shape[0], -1)
        out = F.relu(self.output_fc(concat_flat))

        policy_logits = self.policy_head(out)
        value = self.value_head(out).squeeze(-1)

        return {
            "policy_logits": policy_logits,
            "value": value,
            "hidden_states": new_hidden,
            "cell_states": new_cell,
            "encoding": encoding,
            "all_tick_cell_states": all_tick_cells,
        }

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        return self.drc_stack.init_hidden(batch_size, device)

    def get_action(
        self,
        obs: torch.Tensor,
        hidden_states: Optional[List[torch.Tensor]] = None,
        cell_states: Optional[List[torch.Tensor]] = None,
        greedy: bool = True,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], Dict]:
        """
        Single-step action selection.
        Returns: action, new_hidden_states, new_cell_states, extras
        """
        with torch.no_grad():
            out = self.forward(obs, hidden_states, cell_states)

        logits = out["policy_logits"]
        if greedy:
            action = logits.argmax(dim=-1)
        else:
            action = torch.distributions.Categorical(logits=logits).sample()

        return action, out["hidden_states"], out["cell_states"], out

    def get_cell_states(
        self,
        obs: torch.Tensor,
        hidden_states: Optional[List[torch.Tensor]] = None,
        cell_states: Optional[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor]:
        """Returns cell states after processing obs."""
        with torch.no_grad():
            out = self.forward(obs, hidden_states, cell_states)
        return out["cell_states"]


class ResNetBlock(nn.Module):
    """Simplified residual block as described in Appendix G."""

    def __init__(self, channels: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.ln1 = nn.LayerNorm([channels, 8, 8])
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.ln2 = nn.LayerNorm([channels, 8, 8])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.ln1(self.conv1(x)))
        out = self.ln2(self.conv2(out))
        return F.relu(out + residual)


class ResNetAgent(nn.Module):
    """
    ResNet agent for Sokoban (Appendix G).
    24 simplified residual blocks, 32 channels, no pooling/downsampling.
    """

    def __init__(
        self,
        obs_channels: int = 7,
        num_actions: int = 5,
        num_blocks: int = 24,
        channels: int = 32,
        grid_size: int = 8,
        mlp_hidden: int = 256,
    ):
        super().__init__()
        self.num_blocks = num_blocks

        self.input_conv = nn.Conv2d(obs_channels, channels, 3, padding=1)

        self.blocks = nn.ModuleList([
            ResNetBlock(channels) for _ in range(num_blocks)
        ])

        flat_size = channels * grid_size * grid_size
        self.mlp = nn.Sequential(
            nn.Linear(flat_size, mlp_hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(mlp_hidden, num_actions)
        self.value_head = nn.Linear(mlp_hidden, 1)

    def forward(
        self, obs: torch.Tensor, return_intermediates: bool = False
    ) -> Dict[str, Any]:
        if obs.dim() == 4 and obs.shape[-1] == 7:
            obs = obs.permute(0, 3, 1, 2).contiguous()

        x = F.relu(self.input_conv(obs))
        intermediates = [x] if return_intermediates else None

        for block in self.blocks:
            x = block(x)
            if return_intermediates:
                intermediates.append(x)

        flat = x.view(x.shape[0], -1)
        hidden = self.mlp(flat)
        policy_logits = self.policy_head(hidden)
        value = self.value_head(hidden).squeeze(-1)

        return {
            "policy_logits": policy_logits,
            "value": value,
            "intermediates": intermediates,
        }

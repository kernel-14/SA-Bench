import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):
    """
    A single ConvLSTM cell.
    Based on "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting"
    by Shi et al. (2015).
    """
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int, padding: int):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = padding

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=True)

    def forward(self, input_tensor: torch.Tensor, cur_state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

class ResBlock(nn.Module):
    """
    A simplified residual block for the ResNet agent (Appendix G).
    """
    def __init__(self, channels: int, grid_size: int = 8, kernel_size: int = 3, padding: int = 1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
        # LayerNorm applied over channels, H, W
        self.norm1 = nn.LayerNorm([channels, grid_size, grid_size]) 
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.norm2 = nn.LayerNorm([channels, grid_size, grid_size])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out += identity
        out = F.relu(out)
        return out

class PoolAndInject(nn.Module):
    """
    Pool-and-Inject mechanism as described in DRC agent architecture (Appendix E.3).
    Allows spatial information to spread rapidly.
    """
    def __init__(self, channels: int, grid_size: int):
        super(PoolAndInject, self).__init__()
        self.channels = channels
        self.grid_size = grid_size
        
        self.affine = nn.Linear(2 * channels, grid_size * grid_size * channels)

    def forward(self, h_prev: torch.Tensor) -> torch.Tensor:
        # h_prev shape: (batch_size, channels, H, W)
        mean_pooled = F.adaptive_avg_pool2d(h_prev, (1, 1)).squeeze(-1).squeeze(-1) # (batch_size, channels)
        max_pooled = F.adaptive_max_pool2d(h_prev, (1, 1)).squeeze(-1).squeeze(-1)  # (batch_size, channels)
        
        pooled_concat = torch.cat([mean_pooled, max_pooled], dim=1) # (batch_size, 2*channels)
        
        # Apply affine transformation and reshape
        p_hat = self.affine(pooled_concat) # (batch_size, H*W*channels)
        p = p_hat.view(-1, self.channels, self.grid_size, self.grid_size) # (batch_size, channels, H, W)
        
        return p

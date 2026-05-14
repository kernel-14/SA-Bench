import torch
import torch.nn as nn
import torch.nn.functional as F

from .fno_block import FNOBlock1d, FNOBlock2d # Assuming fno_block.py contains the FNOBlock definitions

class FNO1d(nn.Module):
    def __init__(self, modes, width, in_channels, out_channels, num_fno_layers):
        super(FNO1d, self).__init__()

        self.modes = modes
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_fno_layers = num_fno_layers

        self.padding = 9 # pad the domain if input is non-periodic

        # Lifting layer
        self.p = nn.Linear(in_channels, self.width)
        
        # FNO blocks
        self.fno_blocks = FNOBlock1d(self.width, self.modes, self.num_fno_layers)

        # Projection layer
        self.q = nn.Linear(self.width, out_channels)

    def forward(self, x):
        # x: (batchsize, x_dim, in_channels)
        
        # Lifting
        x = self.p(x) # (batchsize, x_dim, width)
        x = x.permute(0, 2, 1) # (batchsize, width, x_dim)

        # FNO blocks
        x = F.pad(x, [0, self.padding]) # pad the domain if input is non-periodic
        x = self.fno_blocks(x)
        x = x[..., :-self.padding] # unpad

        # Projection
        x = x.permute(0, 2, 1) # (batchsize, x_dim, width)
        x = self.q(x) # (batchsize, x_dim, out_channels)

        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, in_channels, out_channels, num_fno_layers):
        super(FNO2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_fno_layers = num_fno_layers

        self.padding = 9 # pad the domain if input is non-periodic

        # Lifting layer
        self.p = nn.Linear(in_channels, self.width)
        
        # FNO blocks
        self.fno_blocks = FNOBlock2d(self.width, self.modes1, self.modes2, self.num_fno_layers)

        # Projection layer
        self.q = nn.Linear(self.width, out_channels)

    def forward(self, x):
        # x: (batchsize, x_dim, y_dim, in_channels)
        
        # Lifting
        x = self.p(x) # (batchsize, x_dim, y_dim, width)
        x = x.permute(0, 3, 1, 2) # (batchsize, width, x_dim, y_dim)

        # FNO blocks
        x = F.pad(x, [0, self.padding, 0, self.padding]) # pad the domain if input is non-periodic
        x = self.fno_blocks(x)
        x = x[..., :-self.padding, :-self.padding] # unpad

        # Projection
        x = x.permute(0, 2, 3, 1) # (batchsize, x_dim, y_dim, width)
        x = self.q(x) # (batchsize, x_dim, y_dim, out_channels)

        return x

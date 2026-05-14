import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """1D Fourier layer with complex multiplication in frequency domain."""
    
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )
    
    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.shape[-1] // 2 + 1,
                            dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = torch.einsum(
            'bix,iom->box',
            x_ft[:, :, :self.modes],
            self.weights
        )
        x = torch.fft.irfft(out_ft, n=x.shape[-1])
        return x


class SpectralConv2d(nn.Module):
    """2D Fourier layer."""
    
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
    
    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.shape[-2], x.shape[-1] // 2 + 1,
                            dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            'bixy,ioxy->boxy',
            x_ft[:, :, :self.modes1, :self.modes2],
            self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
            'bixy,ioxy->boxy',
            x_ft[:, :, -self.modes1:, :self.modes2],
            self.weights2
        )
        x = torch.fft.irfft2(out_ft, s=(x.shape[-2], x.shape[-1]))
        return x


class FNO1D(nn.Module):
    """1D Fourier Neural Operator for 1D PDE problems.
    
    Following Li et al. 2021.
    Hyperparameters from Table 26:
    - modes: 16
    - width: 64
    - n_layers: 4
    """
    
    def __init__(self, in_channels=1, out_channels=1, modes=16, width=64, n_layers=4,
                 hidden_ch_lift=256, hidden_ch_proj=256):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.lifting = nn.Conv1d(in_channels, width, 1)
        self.projection = nn.Conv1d(width, hidden_ch_proj, 1)
        
        self.spectral_layers = nn.ModuleList([
            SpectralConv1d(width, width, modes) for _ in range(n_layers)
        ])
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(width, width, 1) for _ in range(n_layers)
        ])
        
        self.final = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(hidden_ch_proj, hidden_ch_proj // 2, 1),
            nn.GELU(),
            nn.Conv1d(hidden_ch_proj // 2, out_channels, 1),
        )
        
        self.n_layers = n_layers
    
    def forward(self, x):
        """
        Args:
            x: (B, C_in, X) input
        Returns:
            (B, C_out, X) output
        """
        h = self.lifting(x)
        
        for i in range(self.n_layers):
            h_s = self.spectral_layers[i](h)
            h_c = self.conv_layers[i](h)
            h = h_s + h_c
            h = F.gelu(h)
        
        h = self.projection(h)
        h = self.final(h)
        
        return h


class FNO2D(nn.Module):
    """2D Fourier Neural Operator for 2D PDE problems.
    
    Hyperparameters from Table 32:
    - modes: 16
    - width: 64
    - n_layers: 4
    """
    
    def __init__(self, in_channels=1, out_channels=1, modes1=16, modes2=16, width=64, n_layers=4):
        super().__init__()
        self.lifting = nn.Conv2d(in_channels, width, 1)
        self.projection = nn.Conv2d(width, 256, 1)
        
        self.spectral_layers = nn.ModuleList([
            SpectralConv2d(width, width, modes1, modes2) for _ in range(n_layers)
        ])
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(width, width, 1) for _ in range(n_layers)
        ])
        
        self.final = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(256, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, out_channels, 1),
        )
        
        self.n_layers = n_layers
    
    def forward(self, x):
        h = self.lifting(x)
        
        for i in range(self.n_layers):
            h_s = self.spectral_layers[i](h)
            h_c = self.conv_layers[i](h)
            h = h_s + h_c
            h = F.gelu(h)
        
        h = self.projection(h)
        h = self.final(h)
        
        return h


class FNOModel(nn.Module):
    """FNO wrapper for autoregressive simulation.
    
    Trains on (u_t, f_t) -> (u_{t+1, ..., t+k})
    """
    
    def __init__(self, model_1d=True, in_channels=1, out_channels=1, **kwargs):
        super().__init__()
        if model_1d:
            self.fno = FNO1D(in_channels, out_channels, **kwargs)
        else:
            self.fno = FNO2D(in_channels, out_channels, **kwargs)
        self.model_1d = model_1d
    
    def forward(self, x):
        return self.fno(x)
    
    def rollout(self, u0, f, n_steps):
        """Autoregressive rollout."""
        outputs = [u0]
        current = u0
        
        for step in range(n_steps):
            if self.model_1d:
                inp = torch.cat([current, f[:, step:step+1]], dim=1)
            else:
                inp = torch.cat([current, f[:, step:step+1]], dim=1)
            current = self.fno(inp)
            outputs.append(current)
        
        return torch.stack(outputs, dim=2)  # (B, C, T, X) or (B, C, T, H, W)

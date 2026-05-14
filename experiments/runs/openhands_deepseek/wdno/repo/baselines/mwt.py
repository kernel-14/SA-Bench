import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiwaveletKernel1D(nn.Module):
    """1D Multiwavelet kernel layer.
    
    Following Gupta et al. 2021.
    Uses Legendre polynomial basis.
    Hyperparameters from Table 29:
    - wavelet basis: 'legendre'
    - Fourier modes: 10
    - kernel_size: 4
    """
    
    def __init__(self, in_channels, out_channels, modes=10, kernel_size=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.kernel_size = kernel_size
        
        # Learnable multiwavelet filters
        self.filter_weights = nn.Parameter(
            torch.randn(in_channels, out_channels, modes, kernel_size) * 0.1
        )
        
        # Convolutional skip connection
        self.conv_skip = nn.Conv1d(in_channels, out_channels, 1)
    
    def _legendre_basis(self, x, degree):
        """Compute Legendre polynomials up to given degree."""
        results = [torch.ones_like(x), x]
        for n in range(2, degree + 1):
            results.append(
                ((2 * n - 1) * x * results[-1] - (n - 1) * results[-2]) / n
            )
        return torch.stack(results, dim=-1)
    
    def forward(self, x):
        """
        Args:
            x: (B, C_in, L) input
        Returns:
            (B, C_out, L) output
        """
        B, C, L = x.shape
        
        # Compute multiwavelet decomposition
        # Use Legendre basis over small windows
        window_size = L // self.modes
        if window_size < 1:
            window_size = 1
        
        # Reshape into windows
        x_windows = x[:, :, :window_size * self.modes].reshape(
            B, C, self.modes, window_size
        )
        
        # Legendre basis values
        xs = torch.linspace(-1, 1, window_size, device=x.device)
        basis = self._legendre_basis(xs, self.kernel_size - 1)  # (window_size, kernel_size)
        
        # Project onto basis
        projections = torch.einsum('bcmw,wk->bcmk', x_windows, basis)  # (B, C, modes, kernel_size)
        
        # Transform in multiwavelet domain
        transformed = torch.einsum('bcmk,iomk->bom', projections, self.filter_weights)
        
        # Reconstruct using inverse project
        reconstructed = torch.einsum('bom,wk->bowm', transformed, basis)
        reconstructed = reconstructed.reshape(B, self.out_channels, self.modes * window_size)
        
        # Handle remainder with skip connection
        skip = self.conv_skip(x)
        if reconstructed.shape[-1] < skip.shape[-1]:
            pad = skip.shape[-1] - reconstructed.shape[-1]
            reconstructed = F.pad(reconstructed, (0, pad), mode='replicate')
        
        return reconstructed + skip


class MWT1D(nn.Module):
    """1D Multiwavelet Neural Operator.
    
    From Table 29:
    - Training batch size: 256
    - Training epochs: 300
    - Optimizer: Adam
    - LR scheduler: MultiStepLR
    """
    
    def __init__(self, in_channels=1, out_channels=1, hidden_dim=64, modes=10, kernel_size=4, n_layers=4):
        super().__init__()
        
        self.lifting = nn.Conv1d(in_channels, hidden_dim, 1)
        
        self.mwt_layers = nn.ModuleList([
            MultiwaveletKernel1D(hidden_dim, hidden_dim, modes, kernel_size)
            for _ in range(n_layers)
        ])
        
        self.final = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim // 2, 1),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 2, out_channels, 1),
        )
    
    def forward(self, x):
        h = self.lifting(x)
        
        for mwt_layer in self.mwt_layers:
            h = mwt_layer(h)
            h = F.gelu(h)
        
        return self.final(h)


class MWT2D(nn.Module):
    """2D Multiwavelet Neural Operator.
    
    From Table 34:
    - wavelet basis: 'legendre'
    - Fourier modes: 12
    - kernel_size: 3
    """
    
    def __init__(self, in_channels=1, out_channels=1, hidden_dim=32, modes=12, n_layers=3):
        super().__init__()
        
        self.lifting = nn.Conv2d(in_channels, hidden_dim, 1)
        
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            )
            for _ in range(n_layers)
        ])
        
        self.final = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, out_channels, 1),
        )
    
    def forward(self, x):
        h = self.lifting(x)
        
        for conv_layer in self.conv_layers:
            h = conv_layer(h) + h
            h = F.gelu(h)
        
        return self.final(h)

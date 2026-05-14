import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1D(nn.Module):
    """1D CNN baseline for Burgers simulation.
    
    Following Hwang et al. 2022 and Table 28.
    Uses convolutional autoencoder architecture.
    """
    
    def __init__(self, in_channels=1, out_channels=1, hidden_dim=64, latent_dim=256):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=5, padding=2),
            nn.ELU(),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv1d(hidden_dim * 4, latent_dim, kernel_size=5, stride=2, padding=2),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, hidden_dim * 4, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ELU(),
            nn.ConvTranspose1d(hidden_dim * 4, hidden_dim * 2, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ELU(),
            nn.ConvTranspose1d(hidden_dim * 2, hidden_dim, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ELU(),
            nn.Conv1d(hidden_dim, out_channels, kernel_size=5, padding=2),
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)


class UNet2D(nn.Module):
    """2D U-Net baseline for fluid simulation.
    
    Standard U-Net architecture as described in Ronneberger et al. 2015.
    """
    
    def __init__(self, in_channels=1, out_channels=1, init_features=64):
        super().__init__()
        
        # Encoder
        self.enc1 = self._block(in_channels, init_features)
        self.enc2 = self._block(init_features, init_features * 2)
        self.enc3 = self._block(init_features * 2, init_features * 4)
        self.enc4 = self._block(init_features * 4, init_features * 8)
        
        # Bottleneck
        self.bottleneck = self._block(init_features * 8, init_features * 16)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(init_features * 16, init_features * 8, 2, 2)
        self.dec4 = self._block(init_features * 16, init_features * 8)
        
        self.upconv3 = nn.ConvTranspose2d(init_features * 8, init_features * 4, 2, 2)
        self.dec3 = self._block(init_features * 8, init_features * 4)
        
        self.upconv2 = nn.ConvTranspose2d(init_features * 4, init_features * 2, 2, 2)
        self.dec2 = self._block(init_features * 4, init_features * 2)
        
        self.upconv1 = nn.ConvTranspose2d(init_features * 2, init_features, 2, 2)
        self.dec1 = self._block(init_features * 2, init_features)
        
        self.final = nn.Conv2d(init_features, out_channels, 1)
    
    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        
        b = self.bottleneck(F.max_pool2d(e4, 2))
        
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.final(d1)


class CNNModel(nn.Module):
    """CNN baseline wrapper matching paper interface."""
    
    def __init__(self, model_type='1d', **kwargs):
        super().__init__()
        self.model_type = model_type
        if model_type == '1d':
            self.model = CNN1D(**kwargs)
        else:
            self.model = UNet2D(**kwargs)
    
    def forward(self, x):
        return self.model(x)

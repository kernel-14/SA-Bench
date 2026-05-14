import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatioTemporalEncoder(nn.Module):
    """Spatial-temporal encoder for OFormer (Li et al. 2023).
    
    From Table 30:
    - Input channels: 3
    - Embedding dim of token: 96
    - Embedding dim of encoded sequence: 256
    - Heads: 4
    - Depth: 6
    - Resolution: 120
    """
    
    def __init__(self, in_channels=3, token_dim=96, embed_dim=256, heads=4, depth=6, resolution=120):
        super().__init__()
        self.token_dim = token_dim
        self.embed_dim = embed_dim
        self.resolution = resolution
        
        # Token embedding
        self.token_proj = nn.Linear(in_channels, token_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, resolution, token_dim) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=heads, dim_feedforward=embed_dim,
            batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Cross-attention projection
        self.cross_attn_proj = nn.Linear(token_dim, embed_dim)
    
    def forward(self, x):
        """
        Args:
            x: (B, L, C_in) input sequence
        Returns:
            encoded: (B, embed_dim) global encoding
        """
        B, L, C = x.shape
        
        # Token embedding
        tokens = self.token_proj(x)  # (B, L, token_dim)
        tokens = tokens + self.pos_encoding[:, :L, :]
        
        # Self-attention
        encoded = self.transformer(tokens)  # (B, L, token_dim)
        
        # Pool and project
        encoded = encoded.mean(dim=1)  # (B, token_dim)
        encoded = self.cross_attn_proj(encoded)  # (B, embed_dim)
        
        return encoded


class PointWiseDecoder(nn.Module):
    """Point-wise decoder for OFormer.
    
    From Table 30:
    - Out channels: 1
    - Embedding dim: 256
    - Scale: 120
    """
    
    def __init__(self, out_channels=1, embed_dim=256, resolution=120):
        super().__init__()
        self.resolution = resolution
        
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
        )
        
        self.query_embed = nn.Parameter(torch.randn(1, resolution, embed_dim) * 0.02)
        
        self.final = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, out_channels),
        )
    
    def forward(self, encoded):
        """
        Args:
            encoded: (B, embed_dim) global encoding
        Returns:
            output: (B, C_out, L) decoded output
        """
        B = encoded.shape[0]
        L = self.resolution
        
        # Expand and process
        h = self.proj(encoded)  # (B, embed_dim)
        h = h.unsqueeze(1).repeat(1, L, 1)  # (B, L, embed_dim)
        
        # Cross-attention with query embeddings
        queries = self.query_embed.repeat(B, 1, 1)  # (B, L, embed_dim)
        combined = torch.cat([h, queries], dim=-1)  # (B, L, embed_dim*2)
        
        output = self.final(combined)  # (B, L, out_channels)
        return output.permute(0, 2, 1)  # (B, out_channels, L)


class OFormer1D(nn.Module):
    """Operator Transformer for 1D problems.
    
    Following Li et al. 2023, Table 30 hyperparameters.
    """
    
    def __init__(self, in_channels=1, out_channels=1, token_dim=96, embed_dim=256,
                 heads=4, depth=6, resolution=120):
        super().__init__()
        
        self.encoder = SpatioTemporalEncoder(
            in_channels=in_channels + 2,  # u + initial condition features
            token_dim=token_dim,
            embed_dim=embed_dim,
            heads=heads,
            depth=depth,
            resolution=resolution,
        )
        
        self.decoder = PointWiseDecoder(
            out_channels=out_channels,
            embed_dim=embed_dim,
            resolution=resolution,
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C, L) or (B, L) input
        Returns:
            (B, C_out, L) output
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        B, C, L = x.shape
        
        # Convert to sequence format (B, L, C)
        x_seq = x.permute(0, 2, 1)
        
        # Add features if needed
        if x_seq.shape[-1] < 3:
            padding = torch.zeros(B, L, 3 - x_seq.shape[-1], device=x.device)
            x_seq = torch.cat([x_seq, padding], dim=-1)
        
        encoded = self.encoder(x_seq)
        output = self.decoder(encoded)
        
        return output


class OFormer2D(nn.Module):
    """Operator Transformer for 2D problems.
    
    From Table 35:
    - SpatialTemporalEncoder2D
    - Input channels: 3
    - Token dim: 49
    - Embed dim: 192
    - Heads: 1
    - Depth: 5
    """
    
    def __init__(self, in_channels=1, out_channels=1, token_dim=49, embed_dim=192, heads=4, depth=4):
        super().__init__()
        
        self.lifting = nn.Conv2d(in_channels, token_dim, 3, padding=1)
        self.projection = nn.Conv2d(token_dim, 128, 1)
        
        # Transformer on flattened spatial features
        self.pos_encoding = nn.Parameter(torch.randn(1, 4096, token_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=heads, dim_feedforward=embed_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        self.final = nn.Sequential(
            nn.Conv2d(token_dim, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, out_channels, 1),
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        h = self.lifting(x)  # (B, token_dim, H, W)
        
        # Flatten to sequence
        h_flat = h.flatten(2).permute(0, 2, 1)  # (B, H*W, token_dim)
        h_flat = h_flat + self.pos_encoding[:, :h_flat.shape[1], :]
        
        h_flat = self.transformer(h_flat)
        
        # Reshape back
        h = h_flat.permute(0, 2, 1).reshape(B, -1, H, W)
        h = h[:, :self.projection.in_channels]
        
        output = self.final(h)
        
        return output

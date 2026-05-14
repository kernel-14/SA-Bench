
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from config import P2VAEConfig, FMTConfig
from modules import P2VAEEncoder, P2VAEDecoder, TimestepEmbedder, DiffusionForcingGRU, TransformerBlock

class P2VAE(nn.Module):
    """
    Pretrained Physics Variational Autoencoder (P2VAE).
    Combines an encoder and decoder to compress and reconstruct physical fields.
    """
    def __init__(self, model_size: str = "16M"):
        super().__init__()
        if model_size == "16M":
            base_channels = P2VAEConfig.BASE_DIM_16M
        elif model_size == "87M":
            base_channels = P2VAEConfig.BASE_DIM_87M
        else:
            raise ValueError(f"Unknown P2VAE model size: {model_size}")

        self.encoder = P2VAEEncoder(base_channels=base_channels)
        self.decoder = P2VAEDecoder(base_channels=base_channels)

        # For VAE, usually there are mean and log_var layers after encoder,
        # and then a reparameterization trick.
        # The paper describes L_VAE = 1/2 E[||x - x_hat||^2] + beta KL(q(y|x) || p(y)).
        # This implies a standard VAE structure with a Gaussian prior p(y).
        self.quant_conv = nn.Conv2d(P2VAEConfig.LATENT_CHANNELS, 2 * P2VAEConfig.LATENT_CHANNELS, 1)
        self.post_quant_conv = nn.Conv2d(P2VAEConfig.LATENT_CHANNELS, P2VAEConfig.LATENT_CHANNELS, 1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        moments = self.quant_conv(h)
        mu, log_var = torch.chunk(moments, 2, dim=1)
        return mu, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        # Reparameterization trick
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + std * eps
        reconstruction = self.decode(z)
        return reconstruction, mu, log_var

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Helper to get a sample from the latent space without full VAE output."""
        mu, log_var = self.encode(x)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z


class FlowMarchingTransformer(nn.Module):
    """
    Flow Marching Transformer (FMT) model.
    Learns a unified velocity field for flow matching using a Transformer architecture
    conditioned by diffusion forcing.
    """
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 num_layers: int,
                 latent_channels: int = P2VAEConfig.LATENT_CHANNELS,
                 head_dim: int = FMTConfig.HEAD_DIM,
                 dropout: float = 0.0):
        super().__init__()
        self.latent_channels = latent_channels
        self.embed_dim = embed_dim

        self.timestep_embedder = TimestepEmbedder(embed_dim)
        # Assuming the K parameter (bridge parameter) is also embedded and combined
        self.k_embedder = TimestepEmbedder(embed_dim) # Re-use timestep embedder for k

        # Initial projection of the latent state to the transformer's embedding dimension
        self.input_proj = nn.Conv2d(latent_channels, embed_dim, kernel_size=1)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, head_dim, embed_dim, dropout)
            for _ in range(num_layers)
        ])

        # Final output projection (velocity field has same channels as latent)
        self.output_proj = nn.Conv2d(embed_dim, latent_channels, kernel_size=1)

        # Diffusion Forcing GRU
        self.diffusion_forcing_gru = DiffusionForcingGRU(
            input_dim=embed_dim, # input to GRU is compressed x_t_k tokens (which are embed_dim)
            hidden_dim=embed_dim, # h_s has same dimension as embed_dim
            num_heads=num_heads,
            head_dim=head_dim
        )
        self.initial_h = nn.Parameter(torch.randn(1, embed_dim)) # Initial hidden state for GRU

    def forward(self,
                x_t_k: torch.Tensor, # current state x_t^k (latent)
                t: torch.Tensor,     # timestep t for flow marching [batch_size,]
                k: torch.Tensor,     # bridge parameter k for flow marching [batch_size,]
                h_prev: torch.Tensor # previous latent state from diffusion forcing [batch_size, embed_dim]
                ) -> torch.Tensor:
        
        batch_size = x_t_k.shape[0]

        # 1. Prepare conditioning information
        t_embed = self.timestep_embedder(t) # [batch_size, embed_dim]
        k_embed = self.k_embedder(k)       # [batch_size, embed_dim]
        
        # Combine t_embed, k_embed, and h_prev for conditional input to Transformer
        # The paper mentions 'g_theta(x_t_k, t, h)' and h is evolved by GRU.
        # It's typical for t and k to be used with the condition h.
        # Let's concatenate them as the condition vector for AdaLN-Zero.
        # The exact combination is not specified, a common approach is to sum or concatenate.
        # For simplicity, let's sum them as is common in many conditional diffusion models.
        # h_prev already acts as a condition.
        # Let's make the main conditioning vector the sum of h_prev, t_embed, and k_embed.
        
        # This will be the `cond` input to TransformerBlock via AdaLNZero.
        # AdaLNZero expects a single vector for condition.
        # We need to ensure that the sum results in a vector of `embed_dim`.
        cond_input = h_prev + t_embed + k_embed # [batch_size, embed_dim]

        # 2. Project latent spatial tokens to transformer's embedding space
        # x_t_k is c16p16, so it's [B, C, H, W] -> [B, C, 16, 16]
        # We need to flatten H,W to sequence_length and transpose C to last dim for Transformer.
        x = self.input_proj(x_t_k) # [batch_size, embed_dim, H, W]
        x = rearrange(x, 'b c h w -> b (h w) c') # [batch_size, sequence_length, embed_dim]

        # 3. Apply Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, cond_input) # Pass the combined condition

        # 4. Project back to latent spatial tokens
        x = rearrange(x, 'b (h w) c -> b c h w', h=int(x_t_k.shape[-2]), w=int(x_t_k.shape[-1]))
        output_velocity = self.output_proj(x) # [batch_size, latent_channels, H, W]

        return output_velocity

    def get_initial_h_state(self, batch_size: int) -> torch.Tensor:
        return self.initial_h.repeat(batch_size, 1) # [batch_size, embed_dim]

    def update_diffusion_forcing_gru(self,
                                     h_prev: torch.Tensor,
                                     x_t_k_latent: torch.Tensor,
                                     t_s: torch.Tensor) -> torch.Tensor:
        """
        Updates the diffusion forcing hidden state.
        x_t_k_latent is the _projected_ latent state before transformer blocks.
        """
        # x_t_k_latent is [batch, latent_channels, H, W]
        # Need to project it to embed_dim and flatten for the GRU's cross-attention input
        x_t_k_proj = self.input_proj(x_t_k_latent) # [batch, embed_dim, H, W]
        x_t_k_tokens = rearrange(x_t_k_proj, 'b c h w -> b (h w) c') # [batch, num_tokens, embed_dim]

        return self.diffusion_forcing_gru(h_prev, x_t_k_tokens, t_s)


class PDEFoundationModel(nn.Module):
    """
    Combined PDE Foundation Model consisting of P2VAE and Flow Marching Transformer.
    """
    def __init__(self, p2vae_model_size: str = "16M", fmt_model_size: str = "Small"):
        super().__init__()
        self.p2vae = P2VAE(model_size=p2vae_model_size)

        if fmt_model_size == "Small":
            embed_dim = FMTConfig.EMBED_DIM_SMALL
            num_layers = 4 # Example, typically specified in paper or derived from param count
            num_heads = embed_dim // FMTConfig.HEAD_DIM
        elif fmt_model_size == "Base":
            embed_dim = FMTConfig.EMBED_DIM_BASE
            num_layers = 8 # Example
            num_heads = embed_dim // FMTConfig.HEAD_DIM
        elif fmt_model_size == "Large":
            embed_dim = FMTConfig.EMBED_DIM_LARGE
            num_layers = 12 # Example
            num_heads = embed_dim // FMTConfig.HEAD_DIM
        else:
            raise ValueError(f"Unknown FMT model size: {fmt_model_size}")
        
        self.fmt = FlowMarchingTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            latent_channels=P2VAEConfig.LATENT_CHANNELS
        )

    def forward(self,
                x_0: torch.Tensor, # physical state at t=0
                x_1: torch.Tensor, # physical state at t=1 (target)
                t: torch.Tensor,   # flow marching parameter t
                k: torch.Tensor,   # flow marching parameter k
                h_prev: torch.Tensor # hidden state for diffusion forcing
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # 1. Encode x_0 and x_1 to latent space
        z_0 = self.p2vae.get_latent(x_0)
        z_1 = self.p2vae.get_latent(x_1)

        # 2. Construct x_t^k in latent space
        # Equation: x_t^k = mu_t + sigma_t * z, mu_t = t * x_1 + k * (1 - t) * x_0, sigma_t = (1 - t) * (1 - k)
        # Here x_0, x_1 are latent representations z_0, z_1.
        # z is sampled from N(0, I)
        batch_size = z_0.shape[0]
        device = z_0.device
        latent_shape = z_0.shape[1:] # C, H, W

        # Ensure t, k have correct shapes for broadcasting
        t_reshaped = t.view(batch_size, 1, 1, 1)
        k_reshaped = k.view(batch_size, 1, 1, 1)

        # Draw noise from N(0, I) for mu_t + sigma_t * z
        z_noise = torch.randn(batch_size, *latent_shape, device=device) # [B, C, H, W]

        mu_t = t_reshaped * z_1 + k_reshaped * (1 - t_reshaped) * z_0
        sigma_t = (1 - t_reshaped) * (1 - k_reshaped)
        
        z_t_k = mu_t + sigma_t * z_noise # This is x_t^k in the paper
        
        # 3. Predict velocity field with FMT
        # The paper says: "we adopted u_t^k = (x_1 - x_t^k) / (1 - t) as the training objective,
        # so that x_t_k and t are sufficient as inputs."
        # And the loss is || (1-t) g_theta(x_t_k, t) - (x_1 - x_t_k) ||^2.
        # This implies g_theta predicts the velocity u_t^k directly.
        # The input k is also important for the formulation of x_t^k.
        predicted_velocity = self.fmt(z_t_k, t, k, h_prev) # [batch, latent_channels, H, W]

        # Target for the velocity field (x_1 - x_t_k)
        target_residual = z_1 - z_t_k # [batch, latent_channels, H, W]
        
        # The paper specifies the loss as:
        # L_CFM = 1/2 E[ || (1 - t) g_theta(x_t_k, t, h) - (x_1 - x_t_k) ||^2 ]
        # So the model predicts g_theta, and we compute the target residual.

        return predicted_velocity, target_residual, z_t_k


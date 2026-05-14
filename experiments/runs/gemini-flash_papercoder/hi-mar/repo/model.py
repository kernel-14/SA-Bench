import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Tuple, List, Optional, Union, Callable

# Assuming utils.py is in the same directory or accessible via import
from utils import AdaLN, AdaLNZero

class HiMARTransformerBlock(nn.Module):
    """
    Implements a single scale-aware Transformer block as described in Section 3.2
    (Figure 2c) of the paper. It integrates AdaLN-Zero to leverage scale-specific information.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the HiMARTransformerBlock.

        Args:
            config: A dictionary containing model-specific parameters,
                    including hidden_size, num_attention_heads, dropout_rate, etc.
        """
        super().__init__()
        self.hidden_size = config["himar_hidden_size"]
        num_attention_heads = config["num_attention_heads"]
        dropout_rate = config["dropout_rate"]
        ffn_multiplier = config["ffn_multiplier"]
        activation_fn = self._get_activation_function(config["activation_fn"])

        # AdaLN-Zero for conditioning Attention and FFN
        # The context_embedding_dim should be himar_hidden_size, as the scale embedding
        # and other context vectors are projected to this dimension.
        self.adaln_zero = AdaLNZero(self.hidden_size, self.hidden_size)

        # Layer Normalization for Attention
        # Note: AdaLN-Zero applies the affine parameters, so standard LN should be non-affine.
        # However, the paper's equation shows alpha * LN(z) + beta, implying LN is first applied
        # and then modulated. We'll use the common practice for DiT/AdaLN-Zero where LN is non-affine.
        self.norm1 = nn.LayerNorm(self.hidden_size, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout_rate,
            batch_first=True
        )

        # Layer Normalization for FFN
        self.norm2 = nn.LayerNorm(self.hidden_size, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_size, int(self.hidden_size * ffn_multiplier)),
            activation_fn(),
            nn.Dropout(dropout_rate),
            nn.Linear(int(self.hidden_size * ffn_multiplier), self.hidden_size),
            nn.Dropout(dropout_rate)
        )

    def _get_activation_function(self, name: str) -> Callable[[], nn.Module]:
        """Returns the activation function module based on its name."""
        if name.lower() == "gelu":
            return nn.GELU
        elif name.lower() == "relu":
            return nn.ReLU
        elif name.lower() == "silu":
            return nn.SiLU
        else:
            raise ValueError(f"Unsupported activation function: {name}")

    def forward(self, tokens: torch.Tensor, adaln_context: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the HiMARTransformerBlock.

        Args:
            tokens: Input tensor of shape (B, N, D_model).
            adaln_context: Context vector for AdaLN-Zero, shape (B, D_model).

        Returns:
            Output tensor of shape (B, N, D_model).
        """
        # Get AdaLN-Zero parameters for Attention and FFN
        alpha1, beta1, gamma1, alpha2, beta2, gamma2 = self.adaln_zero(adaln_context)

        # Apply AdaLN for Attention
        norm_tokens_attn = self.norm1(tokens)
        modulated_norm_tokens_attn = alpha1.unsqueeze(1) * norm_tokens_attn + beta1.unsqueeze(1)
        
        # Self-Attention block with residual connection
        attn_output, _ = self.attn(modulated_norm_tokens_attn, modulated_norm_tokens_attn, modulated_norm_tokens_attn)
        tokens = tokens + gamma1.unsqueeze(1) * attn_output

        # Apply AdaLN for FFN
        norm_tokens_ffn = self.norm2(tokens)
        modulated_norm_tokens_ffn = alpha2.unsqueeze(1) * norm_tokens_ffn + beta2.unsqueeze(1)
        
        # FFN block with residual connection
        ffn_output = self.ffn(modulated_norm_tokens_ffn)
        tokens = tokens + gamma2.unsqueeze(1) * ffn_output

        return tokens


class MLPDiffusionHead(nn.Module):
    """
    Implements the MLP-based diffusion head for Phase 1 (Figure 2d).
    It consists of a stack of MLP layers with AdaLN conditioned on timestep embeddings.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the MLPDiffusionHead.

        Args:
            config: A dictionary containing model-specific parameters.
        """
        super().__init__()
        self.input_dim = config["diff_head1_input_dim"] # Output dim of HiMAR Transformer
        self.hidden_size = config["diff_head1_hidden_size"]
        self.num_layers = config["diff_head1_layers"]
        self.output_dim = config["tokenizer_latent_channels"] # Predict noise in latent space
        activation_fn = self._get_activation_function(config["activation_fn"])

        # AdaLN will be conditioned on the time embedding, which is himar_hidden_size
        self.adaln = AdaLN(self.hidden_size, config["himar_hidden_size"])

        layers = []
        # First layer maps from input_dim to hidden_size
        layers.append(nn.Linear(self.input_dim, self.hidden_size))
        layers.append(activation_fn())

        for _ in range(self.num_layers - 1): # Additional hidden layers
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(activation_fn())

        # Final layer maps to output_dim (latent_channels) to predict noise
        layers.append(nn.Linear(self.hidden_size, self.output_dim))

        self.mlp_stack = nn.Sequential(*layers)

    def _get_activation_function(self, name: str) -> Callable[[], nn.Module]:
        """Returns the activation function module based on its name."""
        if name.lower() == "gelu":
            return nn.GELU
        elif name.lower() == "relu":
            return nn.ReLU
        elif name.lower() == "silu":
            return nn.SiLU
        else:
            raise ValueError(f"Unsupported activation function: {name}")

    def forward(self, conditional_tokens: torch.Tensor, timesteps_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLPDiffusionHead.

        Args:
            conditional_tokens: The conditional tokens from HiMAR Transformer, shape (B, N, D_model).
            timesteps_embedding: Timestep embedding, shape (B, D_model).

        Returns:
            Predicted noise of shape (B, N, C_latent).
        """
        # Apply AdaLN before each FFN-like block (or once at the start of the stack)
        # Based on the diagram, AdaLN operates on the input to each 'block'.
        # For simplicity, we apply it to the input of the stack.
        # The paper's formulation for MLPDiffusionHead is quite generic.
        # We'll treat it as a stack of (Linear -> Activation -> AdaLN (on residual part))
        # Or, a series of AdaLN -> Linear blocks.
        # The diagram implies that AdaLN is applied to the input of the *entire* MLP block.
        # Let's apply AdaLN to the input, then pass through the MLP stack.
        # The `adaln` in `utils.py` will modulate based on the timesteps_embedding.

        # The paper says "adaLN, layernorm, and feed-forward layers".
        # Let's interpret it as: conditional_tokens -> AdaLN (with time_emb) -> FFN stack
        
        # AdaLN expects (B, N, hidden_size) or (B, N_tokens, D_model) -> output of same shape
        # Here, conditional_tokens are (B, N_tokens, D_model)
        # timesteps_embedding is (B, D_model)
        
        # To strictly follow the AdaLN equation from utils.py, the input `x` should have `hidden_size`
        # as its last dimension. The `conditional_tokens` are (B, N_tokens, D_model).
        # We need to apply AdaLN for each token. The AdaLN `self.hidden_size` is the `diff_head1_hidden_size`.
        # So we need to project conditional_tokens from `input_dim` to `hidden_size` first.
        
        x = self.mlp_stack[0](conditional_tokens) # First linear layer
        x = self.mlp_stack[1](x) # First activation
        
        # For subsequent blocks, we can apply AdaLN
        for i in range(2, len(self.mlp_stack) - 1, 2): # Iterate over (Linear, Activation) pairs
            x = self.adaln(x, timesteps_embedding) # Apply AdaLN
            x = self.mlp_stack[i](x) # Linear
            x = self.mlp_stack[i+1](x) # Activation

        # Final projection layer to output_dim
        x = self.mlp_stack[-1](x)
        
        return x


class DiffusionTransformerHead(nn.Module):
    """
    Implements the Diffusion Transformer Head for Phase 2 (Figure 2e).
    It contains a stack of Transformer blocks, where each block's AdaLN-Zero
    is conditioned by a context vector `c` (timestep embedding + pooled conditional tokens).
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the DiffusionTransformerHead.

        Args:
            config: A dictionary containing model-specific parameters.
        """
        super().__init__()
        self.latent_channels = config["tokenizer_latent_channels"]
        self.hidden_size = config["himar_hidden_size"] # Transformer hidden size
        self.num_layers = config["diff_head2_layers"]
        
        # Input projection from latent_channels to himar_hidden_size
        self.input_proj = nn.Linear(self.latent_channels, self.hidden_size)
        
        # Output projection from himar_hidden_size to latent_channels (predict noise)
        self.output_proj = nn.Linear(self.hidden_size, self.latent_channels)
        
        # Positional embeddings (learned or sinusoidal)
        # Assuming fixed max sequence length. High-res latents are 32x32 = 1024 tokens.
        self.max_seq_len = config["high_res_image_size"] // config["vae_downsampling_factor"]
        self.max_seq_len *= self.max_seq_len # e.g. 32*32 = 1024
        
        # Use sinusoidal positional embedding for simplicity, similar to original Transformers
        self.pos_emb = nn.Parameter(torch.randn(1, self.max_seq_len, self.hidden_size) * 0.02)
        
        # Stack of HiMARTransformerBlock. Each block will receive the same context_vector_c.
        self.transformer_blocks = nn.ModuleList([
            HiMARTransformerBlock({
                "himar_hidden_size": self.hidden_size,
                "num_attention_heads": config["num_attention_heads"],
                "dropout_rate": config["dropout_rate"],
                "ffn_multiplier": config["ffn_multiplier"],
                "activation_fn": config["activation_fn"],
            })
            for _ in range(self.num_layers)
        ])

    def forward(self, noisy_latents: torch.Tensor, context_vector_c: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the DiffusionTransformerHead.

        Args:
            noisy_latents: Noise-corrupted latent tokens, shape (B, N_high, C_latent).
            context_vector_c: Context vector obtained by summing time embedding and pooled conditional tokens,
                              shape (B, D_model).

        Returns:
            Predicted noise of shape (B, N_high, C_latent).
        """
        batch_size, seq_len, _ = noisy_latents.shape
        
        # Project noisy latents to transformer hidden size
        x = self.input_proj(noisy_latents)
        
        # Add positional embeddings
        x = x + self.pos_emb[:, :seq_len, :]

        # Pass through transformer blocks, conditioning each with context_vector_c
        for block in self.transformer_blocks:
            x = block(x, context_vector_c)
        
        # Project back to latent_channels to predict noise
        predicted_noise = self.output_proj(x)
        return predicted_noise


class HiMARModel(nn.Module):
    """
    Hierarchical Masked Autoregressive Model (Hi-MAR) as described in the paper.
    It orchestrates the two-phase generation process using a shared Transformer backbone
    and two distinct diffusion heads.
    """
    def __init__(self, config: Dict[str, Any], tokenizer_latent_channels: int, num_scales: int):
        """
        Initializes the HiMARModel.

        Args:
            config: A dictionary containing the full model configuration.
            tokenizer_latent_channels: The channel dimension of the VAE latent tokens.
            num_scales: Number of scales (2 for Hi-MAR: low-res and high-res).
        """
        super().__init__()
        self.config = config
        self.device = config["device"]
        self.tokenizer_latent_channels = tokenizer_latent_channels
        self.himar_hidden_size = config["himar_hidden_size"]
        self.conditional_type = config["conditional_type"]

        # 1. Input/Output Projections for visual tokens
        # Maps VAE latent channel dim to transformer hidden dim
        self.input_proj = nn.Linear(self.tokenizer_latent_channels, self.himar_hidden_size)
        # Maps transformer hidden dim back to VAE latent channel dim (for output prediction in general, not diffusion heads directly)
        self.output_proj = nn.Linear(self.himar_hidden_size, self.tokenizer_latent_channels)
        
        # 2. Mask token embedding (a learned parameter for replacing masked visual tokens)
        self.mask_token_embedding = nn.Parameter(torch.randn(self.himar_hidden_size))

        # 3. Time Embedding (for Diffusion Process)
        self.time_embedding = nn.Sequential(
            nn.Linear(self.himar_hidden_size, self.himar_hidden_size),
            nn.SiLU(),
            nn.Linear(self.himar_hidden_size, self.himar_hidden_size)
        )
        # 4. Scale Embeddings (for two phases: low-res and high-res)
        self.scale_embeddings = nn.Embedding(num_scales, self.himar_hidden_size) # 0 for low-res, 1 for high-res

        # 5. Conditional Embeddings (Class or Text)
        if self.conditional_type == "class":
            self.num_classes = config.get("num_classes", 1000) # Default to ImageNet classes
            self.class_embedding = nn.Embedding(self.num_classes + 1, self.himar_hidden_size) # +1 for null class token
            nn.init.normal_(self.class_embedding.weight, std=0.02)
            self.null_class_token = torch.tensor(self.num_classes, device=self.device) # Last index for null token
        elif self.conditional_type == "text":
            # CLIP embeddings are already himar_hidden_size (or will be mapped to it)
            # A linear layer to map CLIP text embeddings to himar_hidden_size if different
            self.clip_projection = nn.Identity() # Placeholder, assume CLIP output matches himar_hidden_size
            # If `clip_text_encoder_dim` != `himar_hidden_size`, then
            # self.clip_projection = nn.Linear(clip_text_encoder_dim, self.himar_hidden_size)
            self.null_text_embedding = nn.Parameter(torch.randn(self.himar_hidden_size)) # A learned null text embedding
        else:
            raise ValueError(f"Unsupported conditional type: {self.conditional_type}")

        # 6. Positional Embeddings (for visual tokens)
        self.high_res_latent_h, self.high_res_latent_w = self._get_latent_hw_from_res(config["high_res_image_size"])
        self.low_res_latent_h, self.low_res_latent_w = self._get_latent_hw_from_res(config["low_res_image_size"])

        self.max_seq_len_low = self.low_res_latent_h * self.low_res_latent_w
        self.max_seq_len_high = self.high_res_latent_h * self.high_res_latent_w

        # Max sequence length for the transformer backbone (conditions + low-res + high-res)
        # Max conditional tokens for CLIP are usually 77.
        self.max_transformer_seq_len = self.max_seq_len_high + self.max_seq_len_low + config.get("max_cond_seq_len", 77)
        self.pos_embed = nn.Parameter(torch.randn(1, self.max_transformer_seq_len, self.himar_hidden_size) * 0.02)

        # 7. Hi-MAR Transformer Blocks (shared backbone)
        self.transformer_blocks = nn.ModuleList([
            HiMARTransformerBlock({
                "himar_hidden_size": self.himar_hidden_size,
                "num_attention_heads": config["num_attention_heads"],
                "dropout_rate": config["dropout_rate"],
                "ffn_multiplier": config["ffn_multiplier"],
                "activation_fn": config["activation_fn"],
            })
            for _ in range(config["himar_transformer_layers"])
        ])

        # 8. Diffusion Heads
        self.mlp_diffusion_head = MLPDiffusionHead({
            "diff_head1_input_dim": config["himar_hidden_size"],
            "diff_head1_hidden_size": config["diff_head1_hidden_size"],
            "diff_head1_layers": config["diff_head1_layers"],
            "tokenizer_latent_channels": self.tokenizer_latent_channels,
            "activation_fn": config["activation_fn"],
            "himar_hidden_size": config["himar_hidden_size"] # for AdaLN context dim
        })

        self.diffusion_transformer_head = DiffusionTransformerHead({
            "diff_head2_layers": config["diff_head2_layers"],
            "himar_hidden_size": config["himar_hidden_size"],
            "tokenizer_latent_channels": self.tokenizer_latent_channels,
            "num_attention_heads": config["num_attention_heads"],
            "dropout_rate": config["dropout_rate"],
            "ffn_multiplier": config["ffn_multiplier"],
            "activation_fn": config["activation_fn"],
            "high_res_image_size": config["high_res_image_size"],
            "vae_downsampling_factor": config["high_res_image_size"] // self.high_res_latent_h # Reconstruct from latent_h
        })

    def _get_latent_hw_from_res(self, image_res: int) -> Tuple[int, int]:
        """Helper to get latent height/width based on image resolution, assuming 8x downsampling."""
        # This is an assumption as VAE downsampling factor is not explicitly in config.
        # It's derived from the tokenizer class based on paper text (256->32, 128->16).
        downsampling_factor = 8 # Consistent with common VAEs and paper's context
        return image_res // downsampling_factor, image_res // downsampling_factor

    def _get_time_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Generates sinusoidal time embeddings.
        """
        # Adapted from DDPM / Transformer positional embedding schemes
        # This creates a scalar time embedding that is then projected.
        freqs = torch.exp(-math.log(10000.0) * torch.arange(start=0, end=self.himar_hidden_size // 2, dtype=torch.float32, device=self.device) / (self.himar_hidden_size // 2))
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.himar_hidden_size % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1) # Pad if odd
        return self.time_embedding(embedding)

    def _get_condition_tokens(self, conditions: Union[torch.Tensor, List[int]], batch_size: int) -> torch.Tensor:
        """
        Prepares conditioning tokens (class or text embeddings).
        """
        if self.conditional_type == "class":
            if isinstance(conditions, list): # List of class IDs
                conditions = torch.tensor(conditions, dtype=torch.long, device=self.device)
            elif conditions.ndim == 0: # Scalar class ID
                conditions = conditions.unsqueeze(0)
            elif conditions.ndim == 1 and conditions.shape[0] == 1: # Batch of 1 scalar class ID
                pass
            
            if conditions.shape[0] != batch_size:
                # If conditions is a single class ID, broadcast it.
                if conditions.numel() == 1:
                    conditions = conditions.repeat(batch_size)
                else:
                    raise ValueError(f"Batch size mismatch for class conditions. Expected {batch_size}, got {conditions.shape[0]}")

            # For class-conditional, output is (B, 1, D_model)
            return self.class_embedding(conditions).unsqueeze(1)
        elif self.conditional_type == "text":
            # Input `conditions` should be (B, L_text, D_clip) from CLIPTextEncoder
            # It's already been processed by CLIPTextEncoder which handles batching
            return self.clip_projection(conditions)
        else:
            raise ValueError(f"Unknown conditional type {self.conditional_type}")
    
    def get_mask_token_embedding(self) -> torch.Tensor:
        """Returns the learned mask token embedding for use in utils._mask_tokens."""
        return self.mask_token_embedding.clone().detach() # Return a detached copy

    def get_null_conditions(self, batch_size: int, device: str) -> torch.Tensor:
        """
        Generates null conditioning tokens for Classifier-Free Guidance.
        """
        if self.conditional_type == "class":
            # Use the special null class token index
            return self.class_embedding(self.null_class_token.repeat(batch_size)).unsqueeze(1)
        elif self.conditional_type == "text":
            # Use a learned null text embedding, expanded to the appropriate sequence length if needed
            # Assuming null_text_embedding is (D_model), expand to (B, L_text, D_model)
            # L_text is usually 77 from CLIP
            max_cond_seq_len = self.config.get("max_cond_seq_len", 77)
            return self.null_text_embedding.unsqueeze(0).unsqueeze(0).repeat(batch_size, max_cond_seq_len, 1)
        else:
            raise ValueError(f"Unknown conditional type {self.conditional_type}")

    def forward_phase1(
        self,
        masked_low_res_tokens: torch.Tensor, # (B, N_low, C_latent)
        conditions: Union[torch.Tensor, List[int]], # (B, L_cond, D_cond) or (B,) class IDs
        timesteps: torch.Tensor, # (B,)
        # mask_indices: torch.Tensor, # (B, N_low) boolean mask, not directly used in forward
        scale_id: int = 0 # 0 for low-res (Phase 1)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass for Phase 1: generating low-resolution tokens.

        Args:
            masked_low_res_tokens: Masked low-resolution VAE latents, shape (B, N_low, C_latent).
            conditions: Conditioning information (class IDs or CLIP text embeddings).
            timesteps: Batch of timesteps for diffusion, shape (B,).
            scale_id: Identifier for the current scale (0 for low-res).

        Returns:
            A tuple:
            - predicted_noise_low: Predicted noise for masked low-res tokens, shape (B, N_low, C_latent).
            - full_transformer_output_phase1: Full output from the HiMAR Transformer, used as pivots for Phase 2.
                                              Shape (B, L_total, D_model).
        """
        batch_size, num_low_res_tokens, _ = masked_low_res_tokens.shape
        
        # 1. Prepare embeddings
        time_emb = self._get_time_embedding(timesteps) # (B, D_model)
        scale_emb = self.scale_embeddings(torch.tensor(scale_id, device=self.device)).unsqueeze(0).repeat(batch_size, 1) # (B, D_model)

        # The adaln_context for HimarTransformerBlocks is derived from scale_emb and time_emb.
        # The paper: "scale vector v, which is leveraged to regress the scale and shift parameters"
        # and "To explicitly encode scale information, we introduce a learnable scale vector for each resolution,
        # which is injected into the Transformer backbone via AdaLN-Zero operations".
        # This implies `adaln_context` for `HiMARTransformerBlock` is `scale_emb` *itself*.
        # The `time_emb` is added to conditional tokens.
        
        # For HiMAR Transformer, `adaln_context` is `scale_emb` as described in paper, section 3.2.
        adaln_context_for_himar_blocks = scale_emb # (B, D_model)

        # Prepare conditional tokens
        cond_tokens = self._get_condition_tokens(conditions, batch_size) # (B, L_cond, D_model)
        
        # Project visual tokens
        projected_low_res_tokens = self.input_proj(masked_low_res_tokens) # (B, N_low, D_model)

        # Concatenate conditional tokens and visual tokens
        # Current length = L_cond + N_low
        x = torch.cat([cond_tokens, projected_low_res_tokens], dim=1)

        # Add positional embeddings
        current_seq_len = x.shape[1]
        x = x + self.pos_embed[:, :current_seq_len, :]

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, adaln_context_for_himar_blocks)
        
        full_transformer_output_phase1 = x

        # Extract transformer output corresponding to low-res visual tokens
        # It's after the conditional tokens
        low_res_visual_tokens_transformer_output = x[:, cond_tokens.shape[1]:, :] # (B, N_low, D_model)

        # Feed to MLPDiffusionHead. This head is conditioned on time_emb.
        predicted_noise_low = self.mlp_diffusion_head(low_res_visual_tokens_transformer_output, time_emb)

        return predicted_noise_low, full_transformer_output_phase1

    def forward_phase2(
        self,
        masked_high_res_tokens: torch.Tensor, # (B, N_high, C_latent)
        noisy_high_res_tokens: torch.Tensor, # (B, N_high, C_latent), actual input for DiffusionTransformerHead
        low_res_transformer_output: torch.Tensor, # (B, L_cond + N_low, D_model) from Phase 1
        conditions: Union[torch.Tensor, List[int]], # (B, L_cond, D_cond) or (B,) class IDs
        timesteps: torch.Tensor, # (B,)
        # mask_indices: torch.Tensor, # (B, N_high) boolean mask, not directly used in forward
        scale_id: int = 1 # 1 for high-res (Phase 2)
    ) -> torch.Tensor:
        """
        Performs the forward pass for Phase 2: generating high-resolution tokens.

        Args:
            masked_high_res_tokens: Masked high-resolution VAE latents, shape (B, N_high, C_latent).
            noisy_high_res_tokens: The noise-corrupted high-resolution VAE latents, fed directly
                                   into the DiffusionTransformerHead as y^0. Shape (B, N_high, C_latent).
            low_res_transformer_output: Output from the Phase 1 HiMAR Transformer, used as pivots.
                                        Shape (B, L_cond + N_low, D_model).
            conditions: Conditioning information (class IDs or CLIP text embeddings).
            timesteps: Batch of timesteps for diffusion, shape (B,).
            scale_id: Identifier for the current scale (1 for high-res).

        Returns:
            predicted_noise_high: Predicted noise for masked high-res tokens, shape (B, N_high, C_latent).
        """
        batch_size, num_high_res_tokens, _ = masked_high_res_tokens.shape

        # 1. Prepare embeddings
        time_emb = self._get_time_embedding(timesteps) # (B, D_model)
        scale_emb = self.scale_embeddings(torch.tensor(scale_id, device=self.device)).unsqueeze(0).repeat(batch_size, 1) # (B, D_model)

        # For HiMAR Transformer, `adaln_context` is `scale_emb` as described in paper, section 3.2.
        adaln_context_for_himar_blocks = scale_emb # (B, D_model)

        # Prepare conditional tokens (though already part of low_res_transformer_output, keeping explicit for clarity)
        # cond_tokens = self._get_condition_tokens(conditions, batch_size) # (B, L_cond, D_model)
        
        # Project visual tokens
        projected_high_res_tokens = self.input_proj(masked_high_res_tokens) # (B, N_high, D_model)

        # Concatenate (low_res_transformer_output as pivots) and (projected masked high-res tokens)
        # This forms the input sequence for the shared HiMAR Transformer backbone.
        # Paper: "the Transformer takes the concatenation of context tokens, small scale conditional tokens
        # and the masked dense visual tokens as input to generate dense conditional tokens"
        # `low_res_transformer_output` already contains `context tokens + small scale conditional tokens`.
        x = torch.cat([low_res_transformer_output, projected_high_res_tokens], dim=1)

        # Add positional embeddings
        current_seq_len = x.shape[1]
        x = x + self.pos_embed[:, :current_seq_len, :]

        # Pass through transformer blocks (shared backbone)
        for block in self.transformer_blocks:
            x = block(x, adaln_context_for_himar_blocks)
        
        # Extract the transformer output corresponding to the high-res visual tokens
        # It's after the (L_cond + N_low) tokens from phase 1
        high_res_visual_tokens_transformer_output = x[:, low_res_transformer_output.shape[1]:, :] # (B, N_high, D_model)

        # Prepare context vector 'c' for Diffusion Transformer Head
        # Paper: "c denotes the context vector obtained by summating the time step embedding and the conditional tokens"
        # We'll use mean pooling of the high-res visual tokens' transformer output
        pooled_conditional_tokens = high_res_visual_tokens_transformer_output.mean(dim=1) # (B, D_model)
        context_vector_c = time_emb + pooled_conditional_tokens # (B, D_model)

        # Feed to DiffusionTransformerHead
        # The DiffusionTransformerHead takes the noisy_high_res_tokens (y^t) directly as its input sequence
        # and `context_vector_c` as its conditioning for AdaLN-Zero.
        predicted_noise_high = self.diffusion_transformer_head(noisy_high_res_tokens, context_vector_c)

        return predicted_noise_high

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Returns a list of all trainable parameters in the model."""
        return list(self.parameters())


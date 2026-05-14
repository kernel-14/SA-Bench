import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )
        self.dim = dim # Store dim to use in forward pass

    def forward(self, t):
        # For diffusion models, t is typically a scalar timestep which needs to be embedded.
        # A common way is to use sinusoidal embeddings then pass through MLP.
        # This is a placeholder for a proper sinusoidal timestep embedding + MLP.
        # Assuming t is already a 1D tensor representing embedded time or needs simple linear projection.
        # For simplicity, let's assume t comes in as (B,) and we project it to (B, dim)
        if t.dim() == 1: # (B,)
            # Create a basic sinusoidal embedding for the timestep
            # This is a common practice in diffusion models (e.g., from DDPM/DiT)
            # The `dim` should be divisible by 2 for sin/cos pairs.
            half_dim = self.dim // 2
            # Ensure t is float for division
            embeddings = math.log(10000) / (half_dim - 1)
            embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
            embeddings = t.float().unsqueeze(1) * embeddings.unsqueeze(0) # (B, 1) * (1, half_dim) -> (B, half_dim)
            embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1) # (B, dim)
        elif t.dim() == 2 and t.shape[1] == self.dim: # (B, dim) already an embedding
            embeddings = t
        else:
            raise ValueError(f"TimestepEmbedding expects t of shape (B,) or (B, {self.dim}), but got {t.shape}")

        return self.mlp(embeddings)


class AdaLNZero(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.linear = nn.Linear(dim, 6 * dim) # predict alpha1, beta1, gamma1, alpha2, beta2, gamma2

    def forward(self, x, c):
        # c is the conditioning vector (either scale vector or context vector)
        # For Hi-MAR Transformer, c is scale_vec.
        # For Diffusion Transformer Head, c is combined timestep embedding and conditional tokens.
        shift_scale_gamma = self.linear(c).unsqueeze(1) # unsqueeze for token dimension: (B, 1, 6*dim)
        alpha1, beta1, gamma1, alpha2, beta2, gamma2 = torch.chunk(shift_scale_gamma, 6, dim=-1)
        return self.norm(x), alpha1, beta1, gamma1, alpha2, beta2, gamma2

class ScaleAwareTransformerBlock(nn.Module):
    """Transformer block for Hi-MAR Transformer, conditioned by a scale vector."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.adaln_zero = AdaLNZero(dim)
        self.attn = SelfAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, conditioning_vector):
        # conditioning_vector here would be the 'scale_vec' for Hi-MAR Transformer blocks
        # or 'c' (time+conditional_tokens) for DiffusionTransformerHead blocks.
        norm_x, alpha1, beta1, gamma1, alpha2, beta2, gamma2 = self.adaln_zero(x, conditioning_vector)
        
        # Attention block with AdaLN-Zero
        # Equation: z_a = z^i + gamma_1 * Attention(alpha_1 * LN(z^i) + beta_1)
        z_a = x + gamma1 * self.attn(alpha1 * norm_x + beta1)
        
        # FFN block with AdaLN-Zero
        # The paper's equation for FFN part reuses AdaLNZero for norm_z_a
        # We need to re-normalize z_a using the same conditioning parameters.
        norm_z_a, _, _, _, _, _ = self.adaln_zero(z_a, conditioning_vector) # Apply AdaLNZero for normalization using the same conditioning params
        
        # Equation: z^(i+1) = z_a + gamma_2 * FFN(alpha_2 * LN(z_a) + beta_2)
        x = z_a + gamma2 * self.mlp(alpha2 * norm_z_a + beta2)
        return x

class DiffusionTransformerHead(nn.Module):
    """Diffusion Transformer Head as described in Section 3.3, using ScaleAwareTransformerBlocks."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_layers=6):
        super().__init__()
        # The paper states: "c denotes the context vector obtained by summating the time step embedding and the conditional tokens"
        # The timestep embedder is part of the overall model, but its output contributes to 'c'.
        # The timestep_embedder here is to handle the time component of 'c'.
        self.timestep_embedder = TimestepEmbedding(dim) # To embed diffusion timestep
        self.blocks = nn.ModuleList([
            ScaleAwareTransformerBlock(
                dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop, act_layer=act_layer, norm_layer=norm_layer
            ) for _ in range(num_layers)
        ])
        self.final_layer = nn.Linear(dim, dim) # To predict epsilon noise in the latent space

    def forward(self, x_noise_corrupted, conditional_tokens_from_transformer_summary, timestep):
        # The paper states: "c denotes the context vector obtained by summating the time step embedding and the conditional tokens"
        # And the input of the first block is the noise-corrupted vector.

        # Embed timestep: (B, dim)
        t_emb = self.timestep_embedder(timestep)
        
        # Form the context vector 'c' by summing timestep embedding and conditional tokens summary
        # `conditional_tokens_from_transformer_summary` is expected to be (B, dim).
        # It represents the aggregated conditional information from the HiMAR Transformer output for phase 2.
        context_vec_c = t_emb + conditional_tokens_from_transformer_summary # (B, dim)

        x = x_noise_corrupted # Input of the first block is the noise-corrupted vector (B, N_high, latent_dim)
        
        # Loop through Diffusion Transformer Head blocks
        for block in self.blocks:
            # Each block in DiffusionTransformerHead is a ScaleAwareTransformerBlock
            # which takes `x` (tokens) and `conditioning_vector` (`context_vec_c`).
            x = block(x, context_vec_c) 
            
        return self.final_layer(x)


class HiMAR(nn.Module):
    """Hierarchical Masked Autoregressive Model (Hi-MAR) as described in the paper."""
    def __init__(self, img_size_low, img_size_high, patch_size, in_chans_vae_latent, embed_dim,
                 depth, num_heads, mlp_ratio, qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm, num_classes=1000, context_dim=None, 
                 diff_head1_out_dim=None, diff_head2_num_layers=None):
        super().__init__()
        
        # VAE and image tokenization:
        # The paper states: "MAR first utilizes the pre-trained variational autoencoder (VAE) to encode I into latent representations"
        # and "encode low-resolution (128x128) and high-resolution (256x256) images into latent representations".
        # For this reproduction, we assume a pre-trained VAE exists. The `latent_proj` layers below are simplified
        # placeholders for projecting the VAE's latent feature maps into transformer embeddings.
        # `in_chans_vae_latent` is the channel dimension of the VAE's latent output (e.g., 4).

        self.latent_proj_low = nn.Linear(in_chans_vae_latent, embed_dim) # From VAE latent channels to embed_dim
        self.latent_proj_high = nn.Linear(in_chans_vae_latent, embed_dim) # From VAE latent channels to embed_dim

        # Calculate number of patches from image sizes and VAE latent resolution
        # The paper implies latent representations I' in R^(h x w x d).
        # We assume `patch_size` here represents the VAE's total downsampling factor (e.g., 8 or 16).
        vae_downsampling_factor = patch_size # Renaming for clarity based on common VAE usage

        latent_res_low = img_size_low // vae_downsampling_factor
        latent_res_high = img_size_high // vae_downsampling_factor

        num_patches_low = latent_res_low * latent_res_low
        num_patches_high = latent_res_high * latent_res_high

        self.pos_embed_low = nn.Parameter(torch.zeros(1, num_patches_low, embed_dim))
        self.pos_embed_high = nn.Parameter(torch.zeros(1, num_patches_high, embed_dim))
        
        self.num_classes = num_classes
        # Context token for class-conditional or text-to-image
        # If text_features are provided, `context_dim` is their embedding dimension (e.g., CLIP output dim).
        # We will project this to `embed_dim` if necessary.
        self.context_projection = None
        if context_dim is not None and context_dim != embed_dim:
            self.context_projection = nn.Linear(context_dim, embed_dim)
        self.context_embed_dim = embed_dim # This will be the dimension of context tokens after projection

        if num_classes > 0:
            self.label_embed = nn.Embedding(num_classes, self.context_embed_dim)
        
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Scale embedding: "sinusoidal embedding is fed into MLP layers to generate scale vector v"
        # This encodes a discrete scale ID (e.g., 0 for low-res, 1 for high-res) into a continuous vector.
        # Using a small MLP for simple integer scale ID here.
        self.scale_embedder = nn.Sequential(
            nn.Linear(1, embed_dim), # Input is a single scale ID, output to embed_dim
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.transformer_blocks = nn.ModuleList([
            ScaleAwareTransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer
            ) for _ in range(depth)
        ])

        # Diffusion Head 1 (MLP-based) for the first phase (low-resolution)
        # It predicts epsilon noise for the masked low-resolution tokens.
        # Output dimension for diff_head1 should be `in_chans_vae_latent` to predict noise in latent space.
        diff_head1_out_dim = diff_head1_out_dim or in_chans_vae_latent
        self.diff_head1 = MLP(embed_dim, out_features=diff_head1_out_dim) 

        # Diffusion Head 2 (Diffusion Transformer Head) for the second phase (high-resolution)
        # Paper states "The number of Transformer blocks in the diffusion head on both phases of Hi-MAR-B/L/H is 6/8/12."
        # So `diff_head2_num_layers` might be different from `depth`.
        diff_head2_num_layers = diff_head2_num_layers or depth # Default to main transformer depth if not specified

        self.diff_head2 = DiffusionTransformerHead(
            dim=in_chans_vae_latent, # Input to DiffHead2 is the noise-corrupted VAE latent itself
            num_heads=num_heads, mlp_ratio=mlp_ratio, num_layers=diff_head2_num_layers,
            act_layer=act_layer, norm_layer=norm_layer
        )
        
        # Re-initialize weights. Not strictly necessary for reproduction but good practice.
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x_low_latent_in, x_high_latent_in, mask_low, mask_high, labels=None, text_features=None, timestep=None):
        """
        Args:
            x_low_latent_in (Tensor): Noise-corrupted low-resolution VAE latent tokens (B, N_low, C_latent).
                                     These are the tokens that will be masked and predicted.
            x_high_latent_in (Tensor): Noise-corrupted high-resolution VAE latent tokens (B, N_high, C_latent).
                                      These are the tokens that will be masked and predicted.
            mask_low (Tensor): Binary mask for low-resolution tokens (B, N_low). 1 for unmasked (visible), 0 for masked.
            mask_high (Tensor): Binary mask for high-resolution tokens (B, N_high). 1 for unmasked (visible), 0 for masked.
            labels (Tensor, optional): Class labels for class-conditional generation (B,). Defaults to None.
            text_features (Tensor, optional): Text embeddings for text-to-image generation (B, N_text, D_text). Defaults to None.
            timestep (Tensor, optional): Diffusion timestep (B,). Required for Diffusion Transformer Head. Defaults to None.
        
        Returns:
            tuple: A tuple containing:
                - predicted_noise_low (Tensor): Predicted noise for low-resolution latents (B, N_low, C_latent).
                - predicted_noise_high_epsilon (Tensor): Predicted noise for high-resolution latents (B, N_high, C_latent).
        """
        B = x_low_latent_in.shape[0]

        # Prepare context tokens (class labels or text features)
        # These tokens will be prepended to the visual token sequence.
        context_tokens = []
        if labels is not None and self.num_classes > 0:
            context_tokens.append(self.label_embed(labels).unsqueeze(1)) # (B, 1, embed_dim)
        if text_features is not None:
            if self.context_projection:
                text_features = self.context_projection(text_features)
            if text_features.dim() == 2: # (B, D_text) -> (B, 1, D_text) if it's a single text embedding per batch
                text_features = text_features.unsqueeze(1)
            context_tokens.append(text_features)
        
        # Combine all context tokens
        if context_tokens:
            context_tokens_seq = torch.cat(context_tokens, dim=1) # (B, num_context_tokens, embed_dim)
            num_context_tokens = context_tokens_seq.shape[1]
        else:
            context_tokens_seq = torch.empty(B, 0, self.context_embed_dim, device=x_low_latent_in.device)
            num_context_tokens = 0


        # --- Phase 1: Bidirectional autoregressive modeling over low-resolution visual tokens ---
        # Convert VAE latents to transformer embeddings
        # These are the *input* latents for the transformer, which are masked versions of the original latents.
        x_low_emb = self.latent_proj_low(x_low_latent_in)
        x_low_emb = x_low_emb + self.pos_embed_low # Add positional embeddings

        # Apply masking to input tokens before feeding to Transformer
        # In MAR, masked tokens are often replaced with a learnable mask embedding or zeroed out.
        # Here, `mask_low` is 0 for masked, 1 for unmasked. We zero out masked tokens.
        # For training, `x_low_latent_in` is noise-corrupted. We are masking these noise-corrupted tokens.
        x_low_masked_transformer_input = x_low_emb * mask_low.unsqueeze(-1) # (B, N_low, embed_dim)

        # Concatenate context tokens with low-resolution masked visual tokens
        x_low_transformer_input = torch.cat((context_tokens_seq, x_low_masked_transformer_input), dim=1)
        x_low_transformer_input = self.pos_drop(x_low_transformer_input)

        # Scale vector for low-resolution (scale ID = 0)
        # This `scale_vec_low` (B, embed_dim) will be the `conditioning_vector` for ScaleAwareTransformerBlock.
        scale_id_low = torch.tensor([0.], device=x_low_latent_in.device).unsqueeze(0).expand(B, -1) # (B, 1)
        scale_vec_low = self.scale_embedder(scale_id_low) # (B, embed_dim) 
        
        # Hi-MAR Transformer blocks for phase 1
        # The transformer blocks process all tokens (context + low-res visual tokens)
        for block in self.transformer_blocks:
            x_low_transformer_input = block(x_low_transformer_input, scale_vec_low)
        
        # Extract the conditional tokens corresponding to the low-resolution visual tokens
        # These are Z^s from the paper, output of the transformer blocks for Phase 1.
        conditional_tokens_low_phase1 = x_low_transformer_input[:, num_context_tokens:] # (B, N_low, embed_dim)

        # Diffusion Head 1 (MLP-based) for low-resolution prediction
        # This head predicts the noise `epsilon` for the masked low-resolution tokens.
        # The objective is `L(z_i, x_i) = E_{epsilon, t} [ || epsilon - epsilon_theta(x_i^t | t, z_i) ||^2 ]` (Eq. 50).
        # `conditional_tokens_low_phase1` acts as `z_i` (Transformer output conditioned on context and visible tokens).
        # The input `x_i^t` (noise-corrupted masked token) is what we are trying to denoise.
        # For an MLP head, it operates independently on each token. So it takes `z_i` and predicts `epsilon`.
        predicted_noise_low = self.diff_head1(conditional_tokens_low_phase1) # (B, N_low, in_chans_vae_latent)


        # --- Phase 2: Autoregressive modeling over dense visual tokens, guided by Phase 1 outputs ---
        # Convert VAE latents to transformer embeddings
        # These are the *input* latents for the transformer, which are masked versions of the original latents.
        x_high_emb = self.latent_proj_high(x_high_latent_in)
        x_high_emb = x_high_emb + self.pos_embed_high # Add positional embeddings

        # Apply masking to input tokens
        x_high_masked_transformer_input = x_high_emb * mask_high.unsqueeze(-1) # (B, N_high, embed_dim)

        # Concatenate context tokens, conditional tokens from phase 1, and masked high-res tokens
        # Paper: "Transformer takes the concatenation of context tokens, small scale conditional tokens and the masked dense visual tokens as input"
        # `conditional_tokens_low_phase1.detach()` because these are from a previous phase and may not require gradients through them for phase 2. (Training-inference consistency).
        x_high_transformer_input = torch.cat((context_tokens_seq, conditional_tokens_low_phase1.detach(), x_high_masked_transformer_input), dim=1)
        x_high_transformer_input = self.pos_drop(x_high_transformer_input)

        # Scale vector for high-resolution (scale ID = 1)
        scale_id_high = torch.tensor([1.], device=x_high_latent_in.device).unsqueeze(0).expand(B, -1) # (B, 1)
        scale_vec_high = self.scale_embedder(scale_id_high) # (B, embed_dim)

        # Hi-MAR Transformer blocks for phase 2 (shared weights)
        for block in self.transformer_blocks:
            x_high_transformer_input = block(x_high_transformer_input, scale_vec_high)

        # Extract the conditional tokens for high-resolution (Z^l) from the output of the Transformer
        # These are the transformer outputs corresponding to the high-resolution visual tokens.
        start_idx_high_tokens_in_output = num_context_tokens + conditional_tokens_low_phase1.shape[1]
        conditional_tokens_high_phase2_output = x_high_transformer_input[:, start_idx_high_tokens_in_output:] # (B, N_high, embed_dim)

        # Diffusion Head 2 (Diffusion Transformer Head) for high-resolution prediction
        # Requires: `x_noise_corrupted` (the input to be denoised, typically the noise-corrupted VAE latent),
        #           `conditional_tokens_from_transformer` (summary context from the HiMAR transformer),
        #           and `timestep`.
        
        if timestep is None:
            raise ValueError("Timestep must be provided for Diffusion Transformer Head in Phase 2.")
        
        # The `conditional_tokens_from_transformer_summary` for the Diffusion Transformer Head
        # Paper: "c denotes the context vector obtained by summating the time step embedding and the conditional tokens"
        # This `c` is a single vector per batch element (B, D) that controls the AdaLN-Zero parameters.
        # We derive this from the *context tokens* from the Hi-MAR Transformer output for phase 2,
        # as they typically summarize the overall conditions.
        # If there are context tokens (class/text), we can take their aggregated representation.
        # For simplicity, let's take the output of the first context token if available, otherwise mean of transformer output.
        if num_context_tokens > 0:
            # Take the output of the first context token (e.g., class token) from the Transformer output
            dt_head_context_for_c = x_high_transformer_input[:, 0] # (B, embed_dim)
        else:
            # If no explicit context tokens, mean of the conditional high-res tokens themselves
            dt_head_context_for_c = conditional_tokens_high_phase2_output.mean(dim=1) # (B, embed_dim)

        # The `x_noise_corrupted` input to DiffusionTransformerHead (y^i in paper's equation)
        # This is the actual noise-corrupted high-resolution VAE latent `x_high_latent_in`.
        # The output of `diff_head2` should be `in_chans_vae_latent` to predict noise in latent space.
        predicted_noise_high_epsilon = self.diff_head2(x_high_latent_in, dt_head_context_for_c, timestep) # (B, N_high, in_chans_vae_latent)

        return predicted_noise_low, predicted_noise_high_epsilon

END_OF_PYTHON_CODE'

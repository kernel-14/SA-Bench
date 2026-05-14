import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from modules import TimestepEmbedder, LabelEmbedder, HiMARTransformer, MLPDiffusionHead, DiffusionTransformerHead
from layers import get_2d_sincos_pos_embed

from typing import Optional


class HiMAR(nn.Module):
    def __init__(
        self,
        vae_path: str,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        num_classes: int,
        conditioning_dim: int,
        transformer_layers: int,
        transformer_hidden_size: int,
        num_heads: int,
        diff_head1_layers: int,
        diff_head1_hidden_size: int,
        diff_head2_layers: int,
        diff_head2_hidden_size: int,
        low_res_image_size: int,
        image_size: int,
        dropout_prob: float = 0.1, # Common default
        text_conditioning: bool = False, # Flag to enable text conditioning
        clip_model_name: str = "openai/clip-vit-large-patch14",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.conditioning_dim = conditioning_dim # This is the dimension of the final conditioning vector for DiTBlocks
        self.text_conditioning = text_conditioning

        # 1. VAE for image tokenization
        self.vae = AutoencoderKL.from_pretrained(vae_path)
        self.latent_channels = self.vae.config.latent_channels # Typically 4
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) # Typically 8

        # Calculate latent grid sizes
        self.low_res_latent_size = low_res_image_size // self.vae_scale_factor
        self.high_res_latent_size = image_size // self.vae_scale_factor

        # Input projection for visual tokens
        self.x_embedder = nn.Linear(self.latent_channels, transformer_hidden_size)

        # Timestep embedding
        self.t_embedder = TimestepEmbedder(transformer_hidden_size)

        # Class/Text embedding
        if text_conditioning:
            self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
            self.text_encoder = CLIPTextModel.from_pretrained(clip_model_name)
            # Adjust conditioning_dim if CLIP output dim is different
            self.y_embedder = nn.Linear(self.text_encoder.config.hidden_size, transformer_hidden_size)
        else:
            self.y_embedder = LabelEmbedder(num_classes, transformer_hidden_size, dropout_prob)

        # Scale embedding (learnable vector for each resolution)
        # The paper states: "learnable scale vector for each resolution, which is injected into the Transformer backbone"
        self.scale_embed_low_res = nn.Parameter(torch.randn(transformer_hidden_size))
        self.scale_embed_high_res = nn.Parameter(torch.randn(transformer_hidden_size))
        
        # Hi-MAR Transformer backbone
        self.transformer = HiMARTransformer(
            num_layers=transformer_layers,
            hidden_size=transformer_hidden_size,
            num_heads=num_heads,
            cond_dim=transformer_hidden_size, # Cond_dim for AdaLNZero
            img_size=self.high_res_latent_size,
            low_res_img_size=self.low_res_latent_size,
        )

        # Diffusion Head 1 (for low-resolution phase) - MLP-based
        self.diffusion_head1 = MLPDiffusionHead(
            in_channels=self.latent_channels,
            hidden_size=diff_head1_hidden_size,
            out_channels=self.latent_channels, # Predicting epsilon for VAE latent
            cond_dim=transformer_hidden_size, # This will be t_embed for MLP head
        )

        # Diffusion Head 2 (for high-resolution phase) - Diffusion Transformer Head
        self.diffusion_head2 = DiffusionTransformerHead(
            num_layers=diff_head2_layers,
            hidden_size=diff_head2_hidden_size,
            num_heads=num_heads, # Using same num_heads as main transformer
            out_channels=self.latent_channels, # Predicting epsilon for VAE latent
            cond_dim=transformer_hidden_size, # This will be t_embed + pooled_c_tokens
        )
        
        # Helper for scale embedding, if paper means explicitly separate injection
        self.scale_mlp = nn.Sequential(
            nn.Linear(transformer_hidden_size, transformer_hidden_size),
            nn.SiLU(),
            nn.Linear(transformer_hidden_size, transformer_hidden_size)
        )


    def forward(
        self, 
        x_low_res: Optional[torch.Tensor] = None, 
        x_high_res: Optional[torch.Tensor] = None, 
        t: torch.Tensor, 
        y: torch.Tensor, 
        phase: int,
        low_res_pivots: Optional[torch.Tensor] = None # Z^s from phase 1
    ):
        """
        x_low_res: low-resolution input (B, C, H_lr, W_lr) for Phase 1.
        x_high_res: high-resolution input (B, C, H_hr, W_hr) for Phase 2.
        t: timestep (B,)
        y: class labels (B,) or text embeddings (B, text_seq_len, text_embed_dim). This is class/text condition.
        phase: 1 for low-res prediction, 2 for high-res prediction
        low_res_pivots: Conditional tokens from phase 1 (Z^s) to guide phase 2. (B, N_low, transformer_hidden_size)
        """
        t_embed = self.t_embedder(t) # (B, transformer_hidden_size)

        if self.text_conditioning:
            # y is text embeddings (B, seq_len, dim)
            # Need to project/pool to (B, transformer_hidden_size) for self.y_embedder
            # For simplicity, we'll pool y_embed to (B, dim)
            if y.ndim == 3: # If y is sequence of tokens
                y_embed = self.y_embedder(y.mean(dim=1)) # (B, transformer_hidden_size)
            else: # If y is already pooled
                y_embed = self.y_embedder(y)
        else:
            y_embed = self.y_embedder(y) # (B, transformer_hidden_size)

        # Phase 1: Predict low-resolution tokens (pivots)
        if phase == 1:
            # Tokenize low-res input
            # x_low_res is already latent representation (B, C, H_lr, W_lr)
            # Reshape to (B, N_low, latent_channels)
            B, C, H_lr, W_lr = x_low_res.shape
            x_low_res = x_low_res.view(B, C, H_lr * W_lr).permute(0, 2, 1) # (B, N_low, C)
            
            x_low_res = self.x_embedder(x_low_res) # (B, N_low, transformer_hidden_size)
            
            # Add positional embedding
            x_low_res = x_low_res + self.transformer.pos_embed_low_res_precomputed.to(x_low_res.device)

            # Context tokens for low-res phase: t_embed, y_embed, and scale_embed_low_res
            s_cond = self.scale_mlp(self.scale_embed_low_res.unsqueeze(0)).expand(B, -1) # (B, hidden_size)
            conditional_tokens = self.transformer(x_low_res, t_embed, y_embed, s_cond) # (B, N_low, transformer_hidden_size)
            
            # Use MLP-based diffusion head for phase 1
            # The MLPDiffusionHead expects `t_cond` for AdaLN and `x` as conditional tokens
            # We use the output of the transformer as the conditional tokens for the diffusion head
            output = self.diffusion_head1(conditional_tokens, t_embed) # (B, N_low, latent_channels)
            
            # Reshape back to latent image format (B, C, H_lr, W_lr)
            output = output.permute(0, 2, 1).view(B, self.latent_channels, H_lr, W_lr)
            return output, conditional_tokens # Also return conditional_tokens for phase 2

        # Phase 2: Predict high-resolution tokens, conditioned on predicted low-res pivots
        elif phase == 2:
            # Tokenize high-res input
            # x_high_res is already latent representation (B, C, H_hr, W_hr)
            # Reshape to (B, N_high, latent_channels)
            B, C, H_hr, W_hr = x_high_res.shape
            x_high_res = x_high_res.view(B, C, H_hr * W_hr).permute(0, 2, 1) # (B, N_high, C)
            
            x_high_res = self.x_embedder(x_high_res) # (B, N_high, transformer_hidden_size)
            
            # Add positional embedding
            x_high_res = x_high_res + self.transformer.pos_embed_high_res_precomputed.to(x_high_res.device)

            # `y` here would be the `conditional_tokens_from_phase1` which are the low-res pivots.
            # We need to concatenate `y_embed` (class/text) with the `low_res_pivots`.
            # The paper states: "Transformer takes the concatenation of context tokens, small scale conditional tokens
            # and the masked dense visual tokens as input to generate dense conditional tokens"
            # This implies the input `x` to the transformer should be (context_tokens + low_res_pivots + masked_tokens).
            # Then the `cond` to AdaLN is just (t_embed + scale_embed_high_res).
            
            # Let's clarify the "concatenation" for HiMARTransformer input and DiTBlocks' cond.
            # Figure 2(b) shows:
            # Phase 1: masked low-res tokens + context tokens -> Transformer -> conditional tokens (Z^s)
            # Phase 2: masked high-res tokens + predicted conditional tokens (Z^s) + context tokens -> Transformer -> conditional tokens (Z^l)
            
            # The 'context tokens' here could be y_embed (class/text).
            # The output of phase 1 is `conditional_tokens_low_res` (Z^s) which are (B, N_low, hidden_size).
            
            # Input to HiMARTransformer in phase 2:
            # Concatenate `y_embed` (class/text) and `conditional_tokens_low_res`
            # `conditional_tokens_low_res` is the second return value of a previous call to this `forward` function with phase=1.
            # Let's pass this `conditional_tokens_low_res` as `y_low_res_pivots` in the forward.
            
            # The "concatenation" part for the Transformer input:
            # It seems more logical that `y_embed` (class/text) is combined with `t_embed` to form the `cond` for AdaLN,
            # and `conditional_tokens_low_res` are treated as additional tokens concatenated to `x_high_res`.
            # However, the diagram shows context tokens separate from the low-res pivots, both feeding into the Transformer.
            
            # Based on "transformer takes the concatenation of context tokens, small scale conditional tokens and the masked dense visual tokens as input"
            # let's assume `y` contains `conditional_tokens_low_res` when phase=2.
            # So `y` (B, N_low, transformer_hidden_size) contains the low-res pivots.
            
            # Concatenate context tokens (y_embed) and low_res_pivots (y_low_res_pivots) to `x_high_res`
            # For simplicity, let's treat `y_embed` as a single token.
            # x_input_phase2 = torch.cat([y_embed.unsqueeze(1), y_low_res_pivots, x_high_res], dim=1)
            # This would change the sequence length and require attention to handle different types of tokens.
            # The `cond_dim` of DiTBlock is a single vector, not a sequence.

            # Re-evaluating Section 3.2: "In the second phase, the Transformer takes the concatenation of context tokens, small scale conditional tokens and the masked dense visual tokens as input to generate dense conditional tokens"
            # This means `x` itself will be a longer sequence.
            # And then: "which are further fed into Diffusion Transformer head for token prediction."
            
            # `y` from forward function input is class labels/text, not low-res pivots.
            # We need to modify `forward` signature or how `y` is used for phase 2.
            # Let's assume for phase 2, `y` argument is replaced by `low_res_pivots`.
            # So, `forward(..., y_low_res_pivots=conditional_tokens_from_phase1_output, ...)`
            
            # For phase 2, y_embed should be the class/text embedding.
            # The `conditional_tokens` (Z^s) from phase 1 are the pivots.
            # The conditioning for the Transformer blocks' AdaLN is (t_embed + y_embed + scale_embed_high_res).
            # The *input sequence* to the transformer `x` is:
            # masked_high_res_tokens + Z^s (low-res pivots) + (optionally) class/text tokens
            
            # Let's assume the current `y` input to forward is `conditional_tokens_low_res` for simplicity.
            # So, `y` is (B, N_low, transformer_hidden_size).
            
            # Cond for AdaLN in HiMARTransformer: t_embed + y_embed (class/text) + scale_embed_high_res
            # This `y_embed` comes from `self.y_embedder(y_class_or_text)`
            # The input sequence `x` to HiMARTransformer is then `x_high_res` (masked tokens) + `low_res_pivots`
            # This means we need two different `y` inputs to the forward function or clarify what `y` means.
            
            # For clarity, let's redefine the `forward` to accept `low_res_pivots` explicitly.
            # For now, I'll assume `y` when `phase=2` is `conditional_tokens_from_phase1`.
            # This means `y_embed` here becomes `y` directly, not embedded by `self.y_embedder`.
            
            # Re-evaluating Figure 2(b) and text:
            # "In the second phase, the masked high-resolution tokens and the predicted conditional tokens in first phase are fed into the Transformer with the Diffusion Transformer head predicting the full high-resolution token sequence."
            # This implies the *input* to the HiMARTransformer is `masked_high_res_tokens` and `predicted_conditional_tokens_from_phase1`.
            # The `cond` to the AdaLN of HiMARTransformer would be `t_embed` + class/text_embed + scale_embed_high_res.
            
            # Let's assume `y` is class/text for `y_embedder`.
            # We need to pass `conditional_tokens_low_res` separately.
            
            # For simplification in this reproduction, we will pass `conditional_tokens_low_res` (from phase 1)
            # as part of the `y` argument to forward, but distinguish its use.
            
            # Conditional tokens for HiMARTransformer (for AdaLNZero): t_embed, y_embed (class/text), scale_embed_high_res
            s_cond = self.scale_mlp(self.scale_embed_high_res.unsqueeze(0)).expand(B, -1) # (B, hidden_size)
            
            # Concatenate low_res_pivots with high_res_tokens before feeding to Transformer
            # We assume y is already the low_res_pivots (Z^s) of shape (B, N_low, hidden_size)
            low_res_pivots = y # Renaming for clarity in phase 2
            
            # Input sequence to Transformer is a concatenation: masked high-res tokens + low-res pivots
            # This implies the initial positional embedding for concatenated tokens needs careful handling.
            # Assuming `x_high_res` already has its positional embedding.
            # And `low_res_pivots` also implicitly carry their spatial info from phase 1.
            # This is complex; the paper states `HiMARTransformer` uses `adaln_zero` so conditioning is a single vector.
            # "Transformer takes the concatenation of context tokens, small scale conditional tokens and the masked dense visual tokens as input to generate dense conditional tokens"
            # This implies *the input sequence to the transformer* should be concatenated, not just the conditioning vector.
            # So, `x_high_res` must be concatenated with `low_res_pivots`.
            
            # To reconcile, the `y_embed` (class/text) is part of `cond` for AdaLN.
            # The `low_res_pivots` are concatenated as part of the input sequence `x`.
            
            # For simplicity, let's consider `x_high_res` as only the *masked* high-res tokens.
            # The conditional tokens (`y`) should be interpreted as the `low_res_pivots` from Phase 1 output.
            
            # Input to the Transformer in Phase 2 for producing Z^l
            # Concat low_res_pivots (y) with high_res_masked_tokens (x_high_res)
            # Let's add the y_embed (class/text) as a single token as well.
            
            # Input to Transformer for phase 2:
            # `x_high_res` is the masked high-resolution visual tokens.
            # `y` argument will be the `conditional_tokens` from phase 1. (B, N_low, hidden_size)
            # `y_embed_class_text` is `self.y_embedder` output. (B, hidden_size)

            # Combine y_embed (class/text) with low_res_pivots (Z^s) to form a richer conditional sequence if needed
            # For now, let's treat `y_embed` (class/text) and `low_res_pivots` as separate entities
            # for the Transformer input concatenation.
            
            # The `HiMARTransformer` (which uses AdaLNZero) receives a single conditioning vector.
            # This vector is `t_embed + y_embed (class/text) + s_cond (scale)`.
            # The input sequence `x` to this transformer is `x_high_res` (masked tokens) and it outputs `Z^l`.
            # The `low_res_pivots` (Z^s) are then passed to `DiffusionTransformerHead`.
            
            # This interpretation contradicts "Transformer takes the concatenation of context tokens, small scale conditional tokens and the masked dense visual tokens as input"
            # and Figure 2(b) where Z^s is directly fed into the Transformer alongside masked high-res tokens.
            
            # Let's assume the paper implies that `y_embed` and `low_res_pivots` are simply extra tokens appended
            # to the `x_high_res` sequence for the HiMARTransformer's input `x`.
            # This means the `cond` for HiMARTransformer's DiTBlocks should only be `t_embed + scale_embed_high_res`.
            # And `y_embed` and `low_res_pivots` are prepended to `x_high_res` sequence.
            
            # This is tricky without explicit architecture for token combining.
            # Simplest interpretation that aligns:
            # `cond` for HiMARTransformer's AdaLNZero = `t_embed` + `y_embed` (class/text) + `scale_embed_high_res`
            # Input `x` to HiMARTransformer = `x_high_res` (masked tokens)
            # The `low_res_pivots` (Z^s) are passed as an additional argument to DiffusionTransformerHead.
            
            # The ablation study mentions "conditional tokens output from the Hi-MAR Transformer of low-resolution visual tokens for the second phase instead".
            # This confirms `Z^s` are fed as conditional tokens, not concatenated into the sequence for `HiMARTransformer`.
            # Thus, the `HiMARTransformer` for Phase 2 processes `x_high_res`.
            # Its output (Z^l) is then fed to `DiffusionTransformerHead` along with `Z^s` (low-res pivots) and `t_embed`.
            
            # For phase 2, let `y_low_res_pivots` be the `conditional_tokens` from phase 1.
            # `x_high_res` is already latent representation (B, C, H_hr, W_hr) -> (B, N_high, C)
            x_high_res_emb = self.x_embedder(x_high_res.view(B, C, H_hr * W_hr).permute(0, 2, 1))
            x_high_res_emb = x_high_res_emb + self.transformer.pos_embed_high_res_precomputed.to(x_high_res_emb.device)
            
            # Cond for HiMARTransformer: t_embed, y_embed (class/text), scale_embed_high_res
            s_cond = self.scale_mlp(self.scale_embed_high_res.unsqueeze(0)).expand(B, -1)
            
            # Generate Z^l (dense conditional tokens) from HiMARTransformer
            dense_conditional_tokens = self.transformer(x_high_res_emb, t_embed, y_embed, s_cond) # (B, N_high, transformer_hidden_size)
            
            # Use Diffusion Transformer Head for phase 2
            # It expects `x` (noise-corrupted tokens, in this case `x_high_res_emb` potentially with mask information applied),
            # `t_emb` and `c_tokens` (which would be `low_res_pivots` and/or `dense_conditional_tokens`).
            
            # The paper says: "dense conditional tokens, which are further fed into Diffusion Transformer head for token prediction."
            # and Figure 2(b) shows `Z^l` going into the Diffusion Transformer Head, along with `Z^s` (low-res pivots).
            # This implies the `c_tokens` argument of DiffusionTransformerHead should be a combination of `Z^l` and `Z^s`.
            # Let's concatenate `Z^l` (dense_conditional_tokens) and `low_res_pivots` (y) as `c_tokens` for the DiffusionTransformerHead.
            
            # This means `DiffusionTransformerHead` needs to handle a combined `c_tokens` sequence.
            # And `x` input to `DiffusionTransformerHead` is the noise-corrupted version of `x_high_res` (latent).
            
            # Revisit DiffusionTransformerHead's forward: `x, t_emb, c_tokens`
            # `x`: noise-corrupted version of high-res image latent (B, N_high, latent_channels) -> embedded to hidden_size
            # `c_tokens`: combined from `dense_conditional_tokens` and `low_res_pivots`.
            
            # The output of HiMARTransformer for phase 2 (`dense_conditional_tokens`) are the conditional tokens for the high-res phase (`Z^l`).
            # These tokens `Z^l` are supposed to *predict* the original high-res tokens.
            # The noise-corrupted input to the diffusion head is usually `x_high_res_latent` + noise.
            # Let `x_diffusion_head` be the noise-corrupted high-res tokens (after embedding).
            
            # For simplicity, let's assume the `x` input to `diffusion_head2` is the embedded high-res tokens,
            # and `c_tokens` passed to `diffusion_head2` is the `dense_conditional_tokens` (`Z^l`) from the HiMARTransformer.
            # This aligns with the first part of "Z^l ... further fed into Diffusion Transformer head for token prediction."
            # The inclusion of `Z^s` into `DiffusionTransformerHead` then would be part of `cond` to AdaLN or
            # concatenated to `Z^l` to form `c_tokens`.
            
            # Let's make `c_tokens_for_diff_head2` be `dense_conditional_tokens` from the Transformer.
            # The paper says "Diffusion Transformer head considers all the masked and unmasked conditional tokens".
            # This points to `dense_conditional_tokens` which are derived from a transformer trained on masked and unmasked tokens.
            
            # Let's also pass `low_res_pivots` (y) as part of `c_tokens` for `diffusion_head2` by concatenating.
            # `c_tokens` in `DiffusionTransformerHead` expects (B, M, hidden_size).
            
            # c_tokens_for_diff_head2 = torch.cat([dense_conditional_tokens, y], dim=1) # y is low_res_pivots
            # This would increase M.
            # The `cond_dim` of `DiffusionTransformerHead` is `transformer_hidden_size`.
            # The `cond` to AdaLN is `t_emb + c_tokens_pooled`.
            # So, `c_tokens` must be pooled to match `cond_dim`.

            # Given the `DiffusionTransformerHead` init takes `cond_dim`, it expects a single vector for AdaLN conditioning.
            # This means `c_tokens` in its forward should be processed/pooled into a single vector before combining with `t_emb`.
            # This also means `x` in its forward should be the `Z^l` or `x_high_res_emb` itself.
            
            # Let's re-read Figure 2(b) on DiffusionTransformerHead:
            # input: `noise_corrupted_vector` + `c`.
            # `c` is `time embedding + conditional tokens`.
            # The `conditional tokens` for the DiffusionTransformerHead should be `Z^l` (dense_conditional_tokens).
            # The `noise_corrupted_vector` should be `x_high_res_emb`.
            
            output = self.diffusion_head2(x_high_res_emb, t_embed, dense_conditional_tokens)
            output = output.permute(0, 2, 1).view(B, self.latent_channels, H_hr, W_hr)
            return output
        else:
            raise ValueError(f"Invalid phase: {phase}")

    @torch.no_grad()
    def encode_vae(self, x: torch.Tensor, low_res: bool = False):
        """Encode image to VAE latent representation."""
        if low_res:
            x = F.interpolate(x, size=(self.low_res_image_size, self.low_res_image_size), mode='bicubic', align_corners=False)
        else:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode='bicubic', align_corners=False)
        
        # Normalize VAE input to [-1, 1] if not already
        # self.vae.config.scaling_factor is typically used for outputs, not inputs.
        # Images are usually normalized to [-1, 1] before VAE encoding.
        # Assuming input x is already in [-1, 1].
        
        latent_dist = self.vae.encode(x).latent_dist
        latent = latent_dist.sample() * self.vae.config.scaling_factor # Scale latents as per diffusers VAE
        return latent

    @torch.no_grad()
    def decode_vae(self, latents: torch.Tensor):
        """Decode VAE latent representation to image."""
        latents = latents / self.vae.config.scaling_factor
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1) # Denormalize to [0, 1]
        return image
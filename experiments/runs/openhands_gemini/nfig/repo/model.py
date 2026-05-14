
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from nfig.modules import Encoder, Decoder, FrequencyGuidedResidualQuantizer, NFIGTransformer
from nfig.layers import fft_2d, ifft_2d, FrequencyMask, SpatialResampler
from nfig.config import get_config

class FR_VAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(
            in_channels=3,
            nf=config.fr_vae_embedding_dim, # Using embedding_dim as base feature for VAE
            num_res_blocks=config.fr_vae_n_residual_layers,
            ch_mult=(1, 2, 4) # Example, usually this depends on downsampling
        )
        self.decoder = Decoder(
            out_channels=3,
            nf=config.fr_vae_embedding_dim,
            num_res_blocks=config.fr_vae_n_residual_layers,
            ch_mult=(1, 2, 4)
        )
        
        # Determine latent feature map size (H', W')
        # This needs to be consistent with the encoder's output.
        # Assuming an input image of 256x256 and ch_mult=(1,2,4) leads to 3 downsampling steps
        # 256 -> 128 -> 64 -> 32. So H', W' = 32.
        # This should be calculated dynamically or passed from config if encoder changes.
        sample_input = torch.randn(1, 3, config.image_size, config.image_size)
        with torch.no_grad():
            latent_f = self.encoder(sample_input)
            self.base_H, self.base_W = latent_f.shape[-2:]
            self.latent_channels = latent_f.shape[1] # C

        self.quantizer = FrequencyGuidedResidualQuantizer(
            codebook_size=config.fr_vae_codebook_size,
            embedding_dim=self.latent_channels, # Embedding dim should match latent channels
            freq_scaling_factors=config.frequency_scaling_factors,
            base_H=self.base_H,
            base_W=self.base_W
        )
        self.frequency_masker = FrequencyMask(
            height=self.base_H,
            width=self.base_W,
            channels=self.latent_channels,
            band_dims=self.quantizer.band_dims, # Pass band_dims from quantizer
            device=config.device
        )
        self.spatial_resampler = SpatialResampler()

    def forward(self, x):
        # x: Input image (B, 3, H, W)
        
        # 1. Encode image to latent feature f
        f = self.encoder(x) # (B, C, H', W')

        # 2. Frequency-guided Decomposition
        fft_f = fft_2d(f) # (B, H', W'//2+1, C, 2)
        
        hat_f_bands = []
        for i in range(len(self.frequency_masker.masks)):
            masked_fft_f = self.frequency_masker(fft_f, i)
            hat_f_i = ifft_2d(masked_fft_f, output_size=(self.base_H, self.base_W))
            hat_f_bands.append(hat_f_i) # List of (B, C, H', W')

        # 3. Frequency-guided Residual Quantization
        quantized_hat_f_bands, q_losses, token_indices_bands = self.quantizer(hat_f_bands)

        # 4. Frequency-guided Composer (Reconstruction of latent feature tilde_f)
        tilde_f = torch.zeros_like(f)
        for q_band in quantized_hat_f_bands:
            tilde_f += q_band # Summing up all quantized frequency components

        # 5. Decode tilde_f back to image
        x_recon = self.decoder(tilde_f) # (B, 3, H, W)

        return x_recon, q_losses, token_indices_bands, f, tilde_f

class NFIGModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fr_vae = FR_VAE(config)
        
        # The total number of tokens for the transformer is the sum of tokens from all frequency bands
        # Each band_dim (h_i, w_i) provides h_i * w_i tokens.
        total_transformer_tokens = sum([h * w for h, w in self.fr_vae.quantizer.band_dims])

        self.nfig_transformer = NFIGTransformer(
            num_tokens=config.fr_vae_codebook_size, # Codebook size is the vocabulary size for tokens
            max_seq_len=total_transformer_tokens,
            dim=config.transformer_dim,
            depth=config.transformer_depth,
            heads=config.transformer_heads,
            dim_head=config.transformer_dim // config.transformer_heads,
            mlp_dim=config.transformer_dim * 4, # Standard practice
            num_classes=config.num_classes
        )

    def forward(self, x, class_label):
        # For training, we mainly need the FR-VAE to get tokens
        # and then train the transformer on these tokens.
        # During inference, the transformer generates tokens, and FR-VAE decodes them.
        
        # This forward pass is primarily for training the FR-VAE or for combined training
        # where FR-VAE encodes and transformer predicts
        x_recon, q_losses, token_indices_bands, f_latent, tilde_f_latent = self.fr_vae(x)
        
        # Flatten token_indices_bands into a single sequence for the transformer
        # (B, N_1), (B, N_2), ..., (B, N_n) -> (B, N_1 + N_2 + ... + N_n)
        # N_i = h_i * w_i
        flattened_tokens = []
        for tokens_in_band in token_indices_bands:
            flattened_tokens.append(rearrange(tokens_in_band, 'b h w -> b (h w)'))
        
        # Concatenate along the sequence dimension
        transformer_input_tokens = torch.cat(flattened_tokens, dim=1) # (B, total_transformer_tokens)

        # Transformer forward pass (e.g., for token prediction loss)
        transformer_logits = self.nfig_transformer(transformer_input_tokens[:, :-1], class_label) # Predict next token
        
        return x_recon, q_losses, transformer_logits, transformer_input_tokens, f_latent, tilde_f_latent

    @torch.no_grad()
    def generate_image(self, num_samples, class_label_input, temperature=1.0, cfg_scale=1.0, top_k=None, device='cuda'):
        # This method outlines the inference process as described in the paper:
        # "first synthesizes the low-frequency components of an image, then iteratively
        # incorporates higher-frequency details, progressively refining the generated output at each step"
        
        # Start with an empty sequence for the transformer
        generated_token_sequences = torch.zeros((num_samples, 0), dtype=torch.long, device=device)
        
        # Accumulate tokens for each band
        all_generated_tokens_per_band = []

        total_tokens_to_generate = self.nfig_transformer.pos_emb.num_embeddings

        # Prepare class labels for CFG
        conditional_class_label = class_label_input.to(device)
        unconditional_class_label = torch.full_like(conditional_class_label, self.nfig_transformer.num_classes).to(device) # num_classes is null class index

        # Generate tokens sequentially
        for _ in range(total_tokens_to_generate):
            # Conditional logits
            cond_logits = self.nfig_transformer(generated_token_sequences, conditional_class_label)
            cond_logits = cond_logits[:, -1, :] # Logits for the last token

            # Unconditional logits
            uncond_logits = self.nfig_transformer(generated_token_sequences, unconditional_class_label)
            uncond_logits = uncond_logits[:, -1, :] # Logits for the last token

            # Apply CFG
            logits = uncond_logits + cfg_scale * (cond_logits - uncond_logits)
            logits = logits / temperature

            if top_k is not None:
                # Apply top-k sampling
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated_token_sequences = torch.cat((generated_token_sequences, next_token), dim=1)

        # After generating all tokens, reshape them back into band-specific tokens
        # and use FR-VAE decoder
        current_idx = 0
        for h, w in self.fr_vae.quantizer.band_dims:
            band_tokens = generated_token_sequences[:, current_idx:current_idx + (h*w)]
            all_generated_tokens_per_band.append(rearrange(band_tokens, 'b (h w) -> b h w', h=h, w=w))
            current_idx += (h*w)
        
        # Decode the tokens using FR-VAE's codebook and decoder
        # This involves reversing the quantization process to get feature maps and then decoding.
        
        # 1. Convert token indices back to quantized feature maps
        reconstructed_latent_bands = []
        for i, (h, w) in enumerate(self.fr_vae.quantizer.band_dims):
            tokens = all_generated_tokens_per_band[i].long()
            # Lookup embeddings from the corresponding quantizer's codebook
            quantized_embedding = self.fr_vae.quantizer.quantizers[i].embedding(tokens) # B, H, W, C
            quantized_embedding = rearrange(quantized_embedding, 'b h w c -> b c h w') # B, C, H, W
            reconstructed_latent_bands.append(quantized_embedding)
        
        # 2. Sum up reconstructed latent bands (similar to Frequency-guided Composer)
        # Need to ensure all are upsampled to the base_H, base_W before summing
        composed_latent_feature = torch.zeros(num_samples, self.fr_vae.latent_channels, self.fr_vae.base_H, self.fr_vae.base_W, device=device)
        for band_feat in reconstructed_latent_bands:
            upsampled_feat = self.fr_vae.spatial_resampler(band_feat, self.fr_vae.base_H, self.fr_vae.base_W)
            composed_latent_feature += upsampled_feat
        
        # 3. Decode to image
        generated_image = self.fr_vae.decoder(composed_latent_feature)
        
        return generated_image

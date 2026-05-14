import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange # For potential reshaping, although manual view/permute might be sufficient
from typing import Optional, List, Tuple

# Assuming Config class is available from config.py
# Assuming FRVAE class is available from models.fr_vae.py
from config import Config
from models.fr_vae import FRVAE # Used in generate method for decoding tokens


class NFIGTransformer(nn.Module):
    """
    NFIG: Next-Frequency Image Generation Transformer.
    A decoder-only Transformer that autoregressively generates frequency tokens.
    """

    def __init__(
        self,
        config: Config,
        vocab_size: int,
        num_classes: int,
        total_sequence_length: int,
        freq_band_token_lengths: List[int],
    ):
        """
        Initializes the NFIGTransformer model components.

        Args:
            config: Configuration object.
            vocab_size: The size of the codebook (number of distinct tokens).
            num_classes: The number of classes for conditional generation.
            total_sequence_length: The total length of the token sequence (sum of all freq bands).
            freq_band_token_lengths: A list where each element is the number of tokens
                                     in a specific frequency band, in order from low to high.
        """
        super().__init__()
        self.config = config

        # --- Configuration Loading ---
        nfig_cfg = config.nfig_transformer
        
        self.embed_dim: int = nfig_cfg.embed_dim
        self.num_heads: int = nfig_cfg.num_heads
        self.ffn_dim: int = nfig_cfg.ffn_dim
        self.depth: int = nfig_cfg.depth

        self.vocab_size: int = vocab_size # Codebook size
        self.num_classes: int = num_classes

        # Special tokens:
        # A start-of-sequence token is needed to initiate generation.
        # It should be outside the range of valid codebook indices [0, vocab_size-1].
        # The token embedding layer needs to accommodate codebook tokens + SOS token
        self.start_of_sequence_token_id: int = self.vocab_size
        self.total_token_embeddings_count: int = self.vocab_size + 1

        # Class embedding: num_classes + 1 for a special null token for CFG
        self.null_class_token_idx: int = self.num_classes

        self.total_sequence_length: int = total_sequence_length
        self.freq_band_token_lengths: List[int] = freq_band_token_lengths

        # --- Embeddings Initialization ---
        self.token_embedding = nn.Embedding(self.total_token_embeddings_count, self.embed_dim)
        # Positional embedding for total_sequence_length.
        self.pos_embedding = nn.Embedding(self.total_sequence_length, self.embed_dim)
        self.class_embedding = nn.Embedding(self.num_classes + 1, self.embed_dim)

        # --- Transformer Decoder Stack ---
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ffn_dim,
            batch_first=True,  # Input/Output tensors are (Batch, Sequence, Features)
            activation="gelu", # Common activation in modern transformers
            norm_first=True, # Pre-LN architecture (common in recent transformers)
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=self.depth)

        # --- Output Head ---
        self.output_head = nn.Linear(self.embed_dim, self.vocab_size)


    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """
        Generates a standard causal mask for sequence processing,
        where attention is blocked to future positions.
        An upper triangular matrix of -inf, with 0s on the diagonal and lower triangle.

        Args:
            sz: The sequence length for which to generate the mask.
            device: The device on which to create the mask tensor.

        Returns:
            A `torch.Tensor` of shape (sz, sz) with -inf for masked positions and 0 for unmasked.
        """
        # (sz, sz) mask, where mask[i, j] is True if j > i (future tokens)
        mask = torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)
        return mask


    def forward(
        self,
        token_indices: torch.Tensor,
        class_labels: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Performs a forward pass through the NFIG Transformer to predict next token logits.

        Args:
            token_indices: Input token sequence (B, S), where S is the current sequence length.
                           These are indices from 0 to `vocab_size-1` (codebook entries).
                           Does not include the `start_of_sequence_token_id`.
            class_labels: Class labels for conditioning (B,).
            attn_mask: Optional attention mask (S, S) for `torch.nn.TransformerDecoder`.
                       If None, a standard causal mask is generated.

        Returns:
            Logits for the next token prediction (B, S, vocab_size).
        """
        if token_indices.ndim != 2:
            raise ValueError(f"token_indices must be 2D (B, S), but got {token_indices.ndim}D.")
        if class_labels.ndim != 1:
            raise ValueError(f"class_labels must be 1D (B,), but got {class_labels.ndim}D.")
        
        batch_size, seq_len = token_indices.shape
        device = token_indices.device

        # --- Embed Tokens ---
        # Ensure token_indices are within the expected range for token_embedding
        if torch.any(token_indices >= self.total_token_embeddings_count) or torch.any(token_indices < 0):
             raise ValueError("token_indices contains values outside the range of token_embedding.")
        token_embeddings = self.token_embedding(token_indices) # (B, S, E)

        # --- Embed Positions ---
        if seq_len > self.total_sequence_length:
            raise ValueError(f"Current sequence length {seq_len} exceeds maximum positional embedding length {self.total_sequence_length}.")
        position_ids = torch.arange(seq_len, device=device) # (S,)
        pos_embeddings = self.pos_embedding(position_ids) # (S, E)

        # --- Embed Class Labels and Condition ---
        if torch.any(class_labels > self.num_classes) or torch.any(class_labels < 0):
             raise ValueError("class_labels contains values outside the range of class_embedding.")
        class_embed = self.class_embedding(class_labels) # (B, E)
        class_cond = class_embed.unsqueeze(1) # (B, 1, E)

        # Combine embeddings: token + position + class condition (class_cond broadcasts over S)
        combined_embeddings = token_embeddings + pos_embeddings + class_cond

        # --- Prepare Attention Mask ---
        if attn_mask is None:
            attn_mask = self._generate_square_subsequent_mask(seq_len, device)
        else:
            # Ensure the provided mask is on the correct device and matches expected size
            if attn_mask.shape != (seq_len, seq_len):
                raise ValueError(f"Provided attn_mask shape {attn_mask.shape} does not match expected ({seq_len}, {seq_len}).")
            attn_mask = attn_mask.to(device)


        # --- Apply Transformer Decoder ---
        transformer_output = self.transformer(tgt=combined_embeddings, tgt_mask=attn_mask) # (B, S, E)

        # --- Output Logits ---
        logits = self.output_head(transformer_output) # (B, S, vocab_size)

        return logits

    @torch.no_grad()
    def generate(
        self,
        class_label: torch.Tensor, # (B,)
        fr_vae: FRVAE,
        cfg_weight: float,
        top_k: int,
        num_generation_steps: int, # This should be total_sequence_length
    ) -> torch.Tensor:
        """
        Generates a batch of full images autoregressively using the NFIG Transformer and FR-VAE decoder.

        Args:
            class_label: A tensor of class labels for conditioning, shape (B,).
            fr_vae: An instance of the trained FRVAE model for decoding tokens to images.
            cfg_weight: Classifier-Free Guidance weight.
            top_k: Number of top-k tokens to sample from. If 0, uses argmax sampling.
            num_generation_steps: The total number of tokens to generate, which should be
                                  equal to `self.total_sequence_length`.

        Returns:
            Generated images (B, 3, H, W), normalized to [-1, 1].
        """
        self.eval() # Set model to evaluation mode
        fr_vae.eval() # Ensure FRVAE is also in eval mode
        device = class_label.device
        batch_size = class_label.shape[0]

        if num_generation_steps != self.total_sequence_length:
            raise ValueError(f"num_generation_steps must match self.total_sequence_length ({self.total_sequence_length}), but got {num_generation_steps}.")

        generated_tokens_list: List[torch.Tensor] = []
        
        # Initialize an empty sequence. The first token will be predicted for this empty sequence.
        current_sequence = torch.empty((batch_size, 0), dtype=torch.long, device=device)

        # Unconditional class label for CFG
        uncond_class_label = torch.full(
            (batch_size,), self.null_class_token_idx, dtype=torch.long, device=device
        )

        for _ in range(num_generation_steps):
            # The current sequence length to input to the transformer.
            # If current_sequence is empty, seq_len will be 0.
            seq_len = current_sequence.shape[1]

            # Generate causal mask for the current sequence length.
            # This mask is applied to the self-attention layer of the TransformerDecoder.
            # Since we are predicting the *next* token, the input sequence is `current_sequence`.
            # The mask should prevent attention to future tokens *within* `current_sequence`.
            # For a sequence of length `S`, the mask is `S x S`.
            attn_mask = self._generate_square_subsequent_mask(seq_len, device)

            # Conditional forward pass
            logits_cond_full = self.forward(current_sequence, class_label, attn_mask=attn_mask) # (B, S, V)
            
            # We are interested in the logits for the *next* token, which are produced
            # at the last position of the output sequence.
            # If seq_len is 0 (first token generation), this will be the output at position 0.
            logits_cond = logits_cond_full[:, -1, :]  # (B, V)

            # Unconditional forward pass (for CFG)
            logits_uncond_full = self.forward(current_sequence, uncond_class_label, attn_mask=attn_mask) # (B, S, V)
            logits_uncond = logits_uncond_full[:, -1, :]  # (B, V)
            
            # Apply Classifier-Free Guidance
            mixed_logits = logits_uncond + cfg_weight * (logits_cond - logits_uncond)

            # Apply Top-k sampling
            if top_k > 0:
                # Get top-k values and their indices
                top_k_values, top_k_indices = torch.topk(mixed_logits, top_k, dim=-1)
                
                # Convert top-k logits to probabilities
                probs = F.softmax(top_k_values, dim=-1)
                
                # Sample from the multinomial distribution
                next_token_choice = torch.multinomial(probs, num_samples=1) # (B, 1)
                
                # Retrieve the actual token ID from the top_k_indices using the sampled choice
                next_token = top_k_indices.gather(-1, next_token_choice) # (B, 1)
            else: # Argmax sampling if top_k is 0 or less
                next_token = torch.argmax(mixed_logits, dim=-1, keepdim=True) # (B, 1)

            generated_tokens_list.append(next_token)
            # Append the newly generated token to the sequence for the next iteration
            current_sequence = torch.cat([current_sequence, next_token], dim=1)
        
        all_generated_tokens = torch.cat(generated_tokens_list, dim=1) # (B, total_sequence_length)

        # Decode tokens back into an image using FR-VAE
        reconstructed_images = fr_vae.decode_from_tokens(all_generated_tokens)

        return reconstructed_images


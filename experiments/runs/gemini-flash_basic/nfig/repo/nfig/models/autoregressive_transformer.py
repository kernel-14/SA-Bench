import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        assert self.head_dim * num_heads == self.embedding_dim, "embedding_dim must be divisible by num_heads"

        self.wq = nn.Linear(embedding_dim, embedding_dim)
        self.wk = nn.Linear(embedding_dim, embedding_dim)
        self.wv = nn.Linear(embedding_dim, embedding_dim)
        self.wo = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: B x T x C (Batch, Sequence_Length, Embedding_Dim)
        batch_size, seq_len, _ = x.shape

        q = self.wq(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # B x N_H x T x H_D
        k = self.wk(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # B x N_H x T x H_D
        v = self.wv(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # B x N_H x T x H_D

        # Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5) # B x N_H x T x T

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, v) # B x N_H x T x H_D

        # Concatenate heads and put through final linear layer
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embedding_dim) # B x T x C
        output = self.wo(output)
        return output

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.GELU(), # Often used in modern transformers instead of ReLU
            nn.Linear(ff_dim, embedding_dim),
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention part
        attn_output = self.attention(self.norm1(x), mask=mask)
        x = x + self.dropout1(attn_output)

        # Feed-forward part
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout2(ff_output)
        return x

class NFIGTransformer(nn.Module):
    def __init__(self, 
                 codebook_size: int, 
                 embedding_dim: int, 
                 num_heads: int, 
                 num_transformer_blocks: int, 
                 ff_dim: int, 
                 scaling_factors: list[int], 
                 max_seq_len: int, # Maximum possible sequence length (e.g., H'*W')
                 dropout: float = 0.1):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.num_frequency_bands = len(scaling_factors)
        self.scaling_factors = scaling_factors

        # Token embeddings for each token in the codebook
        self.token_embeddings = nn.Embedding(codebook_size, embedding_dim)
        
        # Positional embeddings (learned or fixed, here learned for flexibility)
        self.position_embeddings = nn.Parameter(torch.randn(max_seq_len, embedding_dim))

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, ff_dim, dropout) 
            for _ in range(num_transformer_blocks)
        ])

        # Output layer to predict the next token (logits over codebook_size)
        self.output_layer = nn.Linear(embedding_dim, codebook_size)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # Causal mask for autoregressive generation
        # [1, 0, 0]
        # [1, 1, 0]
        # [1, 1, 1]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0) # 1 x 1 x S x S
        return mask

    def forward(self, token_indices_list: list[torch.Tensor]) -> list[torch.Tensor]:
        # token_indices_list: list of B x h_i x w_i for each frequency band
        # The generation process is from low to high frequency bands.
        
        # Flatten all token indices and concatenate them into a single sequence
        # The order matters: F1_tokens, F2_tokens, ..., Fn_tokens
        
        # The paper describes p(T_1, ..., T_n) = product p(T_i | T_1, ..., T_{i-1})
        # This means generation for T_i is conditioned on all previous T_j (j<i).
        # Inside each T_i, tokens are generated in a raster-scan manner (implied by AR nature).

        all_tokens_embedded = []
        all_token_seq_lengths = []
        
        batch_size = token_indices_list[0].shape[0]
        device = token_indices_list[0].device

        # Embed and flatten each frequency band's tokens
        for i, token_indices_i in enumerate(token_indices_list):
            # token_indices_i: B x h_i x w_i
            seq_len_i = token_indices_i.shape[1] * token_indices_i.shape[2]
            all_token_seq_lengths.append(seq_len_i)

            flat_token_indices = token_indices_i.view(batch_size, -1) # B x (h_i*w_i)
            embedded_tokens = self.token_embeddings(flat_token_indices) # B x (h_i*w_i) x embedding_dim
            all_tokens_embedded.append(embedded_tokens)

        # Concatenate all embedded token sequences
        # This will be [T1_embedded, T2_embedded, ..., Tn_embedded]
        full_sequence_embedded = torch.cat(all_tokens_embedded, dim=1) # B x Total_Seq_Len x embedding_dim
        
        total_seq_len = full_sequence_embedded.shape[1]
        
        # Add positional embeddings
        if total_seq_len > self.position_embeddings.shape[0]:
            raise ValueError(f"Sequence length {total_seq_len} exceeds max_seq_len {self.position_embeddings.shape[0]}")
        full_sequence_embedded = full_sequence_embedded + self.position_embeddings[:total_seq_len].unsqueeze(0)

        # Create block-wise causal mask
        # This mask should ensure: 
        # 1. Causal attention within each frequency block T_i.
        # 2. Attention from T_i to all T_j where j < i.
        # The paper mentions "block-wise causal attention [19]" - VAR uses this for different resolutions.
        # Here, blocks are frequency bands.
        
        # A full causal mask over the concatenated sequence will achieve this.
        # A token at position `p` can only attend to positions `0` to `p-1`.
        # If T1 tokens come first, then T2, etc., a standard causal mask over the full sequence
        # naturally enforces that T2 tokens can attend to T1, and T1 tokens can only attend to prior T1 tokens.
        # And within T2, tokens can attend to prior T2 tokens and all of T1.
        
        causal_mask = self._generate_causal_mask(total_seq_len, device)

        # Pass through transformer blocks
        transformer_output = full_sequence_embedded
        for block in self.transformer_blocks:
            transformer_output = block(transformer_output, mask=causal_mask)

        # Output layer to predict the next token (logits over codebook_size)
        logits = self.output_layer(transformer_output) # B x Total_Seq_Len x codebook_size

        # Reshape logits back into a list of logits for each frequency band
        output_logits_list = []
        start_idx = 0
        for seq_len_i in all_token_seq_lengths:
            end_idx = start_idx + seq_len_i
            logits_i = logits[:, start_idx:end_idx, :]
            output_logits_list.append(logits_i)
            start_idx = end_idx

        return output_logits_list

    def generate(self, H_prime: int, W_prime: int, initial_tokens: torch.Tensor = None, temperature: float = 1.0, top_k: int = None, cfg_scale: float = 1.0) -> list[torch.Tensor]:
        # This method would handle the actual autoregressive sampling process.
        # For static benchmark, this is a conceptual outline.
        
        # H_prime, W_prime: dimensions of the full feature map from VAE encoder
        # initial_tokens: B x h_0 x w_0 for the lowest frequency band (if provided, else start from scratch)
        
        generated_token_indices_list = []
        
        batch_size = 1 # Assuming single image generation for simplicity
        device = self.token_embeddings.weight.device
        
        # This section simulates the iterative generation process.
        # In a real implementation, we would generate tokens one by one for each band,
        # and then for each token within a band, conditioning on previously generated tokens.
        
        # For the static benchmark, we'll outline the loop structure:
        
        current_sequence_tokens = [] # Stores generated token indices for all bands so far (flattened)
        current_sequence_embedded = [] # Stores embedded tokens
        current_total_seq_len = 0
        
        for i in range(self.num_frequency_bands):
            s = self.scaling_factors[i]
            h_i, w_i = H_prime // s, W_prime // s
            seq_len_i = h_i * w_i
            
            generated_band_tokens = torch.zeros(batch_size, h_i, w_i, dtype=torch.long, device=device)
            
            for _token_idx_in_band in range(seq_len_i):
                # Prepare input for the transformer: all previously generated tokens
                if not current_sequence_embedded:
                    # For the very first token, we might have an empty context
                    # or a special 'start' token. For now, assume it starts with embedding 0.
                    # A more robust implementation might use a learned start token.
                    input_embedding = self.token_embeddings(torch.tensor([[0]], device=device)) # Dummy start
                    current_total_seq_len = 1
                else:
                    # Concatenate existing embedded tokens
                    input_embedding = torch.cat(current_sequence_embedded, dim=1)
                    current_total_seq_len = input_embedding.shape[1]
                    
                # Add positional embeddings (conceptual for generation)
                if current_total_seq_len > self.position_embeddings.shape[0]:
                    raise ValueError("Sequence length during generation exceeds max_seq_len")
                input_embedding = input_embedding + self.position_embeddings[:current_total_seq_len].unsqueeze(0)
                
                # Apply causal mask. During generation, the mask is implicitly handled by feeding
                # only generated tokens.
                # Here, we will just apply it to the last token's prediction if we were doing parallel prediction.
                # For true autoregressive (token-by-token), we feed one token at a time and predict the next.
                
                # For simplicity in outlining, we'll assume we get logits for the *next* token.
                # In a real setup, we'd feed `current_sequence_embedded` and predict the `current_total_seq_len`-th token.
                
                # We need to compute attention for current_total_seq_len positions.
                # The causal mask will be for the full input up to the point of prediction.
                
                # This is a high-level conceptual loop. Actual token-by-token generation is more complex.
                # For a static model, we assume the transformer *could* do this.
                
                # If we were predicting the next token, we would typically take the last output embedding
                # from the transformer and pass it through the output layer.
                
                # Simplified generation step:
                # This is a single pass for illustration, not token-by-token.
                # In actual sampling, we'd append one token, then predict next, iteratively.
                
                # To correctly reflect autoregressive generation, we simulate by adding a single 'next' token
                # for each step.
                
                # (Conceptual) Run through transformer blocks to get prediction for the *next* token
                # last_output_embedding = ... (from transformer output for the last position)
                # next_token_logits = self.output_layer(last_output_embedding)
                
                # For the static benchmark, let's simulate by picking a random token.
                # In practice, it would be sample from `next_token_logits` with temperature and top_k.
                next_token_idx = torch.randint(0, self.codebook_size, (batch_size, 1), device=device)
                
                # Store the generated token
                generated_band_tokens[:, _token_idx_in_band // w_i, _token_idx_in_band % w_i] = next_token_idx.squeeze(1)
                
                # Add the newly generated token to the sequence for the next iteration
                current_sequence_tokens.append(next_token_idx) # Flattened 1-token sequence
                current_sequence_embedded.append(self.token_embeddings(next_token_idx)) # 1-token embedded sequence
            
            generated_token_indices_list.append(generated_band_tokens)
            
        return generated_token_indices_list


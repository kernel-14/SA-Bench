import torch
import torch.nn as nn

class ARMModel(nn.Module):
    """
    Conceptual Autoregressive Model (ARM).
    This model predicts the next token in a sequence given the preceding tokens.
    """
    def __init__(self, vocab_size, sequence_length, hidden_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim

        # Similar to MDM, a Transformer-based architecture is implied.
        # Key difference is the use of causal attention.
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(sequence_length, hidden_dim) # Learnable positional embeddings

        # Transformer decoder layers with causal masking
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim*4, batch_first=True)
        # For an ARM, we usually use a TransformerEncoder with a causal mask, or a TransformerDecoder
        # if there's an encoder-decoder setup, but for a standard LM, it's typically a decoder-only architecture
        # with a causal attention mask.
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim*4, batch_first=True)
        self.transformer_decoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=6) # Using Encoder with causal mask

        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        """
        Forward pass for the ARM model.
        Args:
            x (torch.Tensor): Input sequence. Shape: (batch_size, sequence_length).
        Returns:
            torch.Tensor: Log probabilities for the next token at each position.
                          Shape: (batch_size, sequence_length, vocab_size).
        """
        batch_size, seq_len = x.shape

        token_embed = self.token_embedding(x)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_embed = self.position_embedding(positions)

        input_embed = token_embed + pos_embed

        # Causal mask for autoregressive generation
        # Subsequent positions cannot attend to current or future positions.
        causal_mask = self.generate_causal_mask(seq_len).to(x.device)

        # In TransformerEncoder, src_mask is applied to the self-attention layer.
        encoded_features = self.transformer_decoder(input_embed, mask=causal_mask)

        logits = self.output_layer(encoded_features)

        return torch.log_softmax(logits, dim=-1)

    def generate_causal_mask(self, seq_len):
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

import torch
import torch.nn as nn

class AutoregressiveImageGenerator(nn.Module):
    def __init__(self, num_frequency_bands, embedding_dim, num_transformer_layers):
        super(AutoregressiveImageGenerator, self).__init__()
        self.num_frequency_bands = num_frequency_bands
        self.embedding_dim = embedding_dim
        self.num_transformer_layers = num_transformer_layers

        # Positional embedding
        self.positional_embedding = nn.Parameter(torch.randn(1, num_frequency_bands, embedding_dim))

        # Transformer Decoder
        self.decoder = nn.Transformer(
            d_model=embedding_dim,
            nhead=8,
            num_decoder_layers=num_transformer_layers,
            batch_first=True,
        )

        # Embedding projection for tokens
        self.token_projection = nn.Linear(embedding_dim, num_frequency_bands)

    def forward(self, x):
        # Initial embeddings
        x = x + self.positional_embedding

        # Transformer decoding
        decoded_tokens = self.decoder(x, x)

        # Token projection to frequency bands
        generated_tokens = self.token_projection(decoded_tokens)

        return generated_tokens


# model.py

from modules import DenoisingNetwork

class MaskedDiffusionModel:
    def __init__(self, vocab_size: int, sequence_length: int, hidden_dim: int):
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.denoising_network = DenoisingNetwork(vocab_size, hidden_dim)

    def forward(self, x_t, t):
        return self.denoising_network(x_t, t)
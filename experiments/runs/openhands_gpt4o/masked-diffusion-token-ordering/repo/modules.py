# modules.py

import torch
import torch.nn as nn

class DenoisingNetwork(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int):
        super(DenoisingNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, t):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.output_layer(lstm_out)
        return output
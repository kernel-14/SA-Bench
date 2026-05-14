import torch
import torch.nn as nn
from layers import AdaLNZero, SinusoidalEmbedding

class ScaleAwareTransformer(nn.Module):
    def __init__(self, config):
        super(ScaleAwareTransformer, self).__init__()
        self.layers = nn.ModuleList([
            ScaleAwareTransformerBlock(config) for _ in range(config['num_layers'])
        ])

    def forward(self, tokens, *context):
        for layer in self.layers:
            tokens = layer(tokens, *context)
        return tokens

class DiffusionTransformerHead(nn.Module):
    def __init__(self, config):
        super(DiffusionTransformerHead, self).__init__()
        self.layers = nn.ModuleList([
            DiffusionTransformerBlock(config) for _ in range(config['num_layers'])
        ])

    def forward(self, tokens):
        for layer in self.layers:
            tokens = layer(tokens)
        return tokens

class ScaleAwareTransformerBlock(nn.Module):
    def __init__(self, config):
        super(ScaleAwareTransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(config['hidden_size'], config['num_heads'])
        self.feed_forward = nn.Sequential(
            nn.Linear(config['hidden_size'], config['ffn_size']),
            nn.ReLU(),
            nn.Linear(config['ffn_size'], config['hidden_size'])
        )
        self.ada_ln = AdaLNZero(config['hidden_size'])

    def forward(self, tokens, *context):
        tokens = self.ada_ln(tokens)
        tokens, _ = self.attention(tokens, tokens, tokens)
        tokens = self.feed_forward(tokens)
        return tokens

class DiffusionTransformerBlock(nn.Module):
    def __init__(self, config):
        super(DiffusionTransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(config['hidden_size'], config['num_heads'])
        self.feed_forward = nn.Sequential(
            nn.Linear(config['hidden_size'], config['ffn_size']),
            nn.ReLU(),
            nn.Linear(config['ffn_size'], config['hidden_size'])
        )

    def forward(self, tokens):
        tokens, _ = self.attention(tokens, tokens, tokens)
        tokens = self.feed_forward(tokens)
        return tokens
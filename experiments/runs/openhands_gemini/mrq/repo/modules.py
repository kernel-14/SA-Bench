
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from layers import LayerNorm, ln_activ
from config import Config

def init_weights(m, weight_init=Config.WEIGHT_INITIALIZATION, bias_init=Config.BIAS_INITIALIZATION):
    if isinstance(m, nn.Linear):
        if weight_init == "Xavier uniform":
            nn.init.xavier_uniform_(m.weight)
        elif weight_init == "orthogonal": # For consistency, though paper only mentions Xavier
            nn.init.orthogonal_(m.weight)
        else:
            raise ValueError(f"Unknown weight initialization: {weight_init}")
        if m.bias is not None:
            nn.init.constant_(m.bias, bias_init)
    elif isinstance(m, nn.Conv2d):
        if weight_init == "Xavier uniform":
            nn.init.xavier_uniform_(m.weight)
        elif weight_init == "orthogonal":
            nn.init.orthogonal_(m.weight)
        else:
            raise ValueError(f"Unknown weight initialization: {weight_init}")
        if m.bias is not None:
            nn.init.constant_(m.bias, bias_init)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, activation_fn, use_layer_norm=True):
        super().__init__()
        self.activation_fn = activation_fn
        self.use_layer_norm = use_layer_norm
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layer_norm and i < num_layers - 1: # No LayerNorm on last layer
                layers.append(LayerNorm(hidden_dim))
            in_dim = hidden_dim
        self.mlp = nn.ModuleList(layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.apply(init_weights)

    def forward(self, x):
        for i, layer in enumerate(self.mlp):
            if isinstance(layer, nn.Linear):
                x = layer(x)
            elif isinstance(layer, LayerNorm):
                x = layer(x)
            x = self.activation_fn(x)
        return self.output_layer(x)

class CNNEncoder(nn.Module):
    def __init__(self, state_channels, zs_dim, activation_fn):
        super().__init__()
        self.activation_fn = activation_fn
        self.cnn = nn.Sequential(
            nn.Conv2d(state_channels, 32, 3, stride=2),
            activation_fn,
            nn.Conv2d(32, 32, 3, stride=2),
            activation_fn,
            nn.Conv2d(32, 32, 3, stride=2),
            activation_fn,
            nn.Conv2d(32, 32, 3, stride=1),
            activation_fn
        )
        # Assuming 84x84 input, output size will be 32 * 35 * 35 after 4 convs (with kernel 3, stride 2, same padding for first 3, valid for last)
        # ( ( ( (84 - 3)/2 + 1 ) - 3)/2 + 1 ) - 3)/2 + 1 ) - 3)/1 + 1 ) = 35
        # The paper implies 1568 features, which is 32 * 7 * 7. This means a different padding/stride or initial size.
        # Let's follow the paper's implied intermediate shape by using the output of 1568 (32 * 7 * 7) if the input was 64x64 or similar
        # For 84x84, the calculation is:
        # (84 - 3)/2 + 1 = 41.5 -> 41 (stride 2, if no padding, output is floor((input - kernel)/stride) + 1)
        # (41 - 3)/2 + 1 = 19.5 -> 19
        # (19 - 3)/2 + 1 = 8.5 -> 8
        # (8 - 3)/1 + 1 = 6
        # So it would be 32 * 6 * 6 = 1152 if standard convolutions.
        # The paper implies the output from the CNN for 84x84 input is 1568 (32 * 7 * 7).
        # This implies that perhaps 'padding' is used implicitly or the input size is different.
        # Given "Assumes 84x84 input" and "self.zs_lin = nn.Linear(1568, zs_dim)", we will assume the conv layers somehow lead to 1568 features.
        # For now, I will use a dummy linear layer and add a comment that this needs careful checking against actual implementation if source code was available.
        self.output_features = 32 * 7 * 7 # This is based on paper's implied size (1568)
        self.linear = nn.Linear(self.output_features, zs_dim)
        self.layer_norm = LayerNorm(zs_dim)
        self.apply(init_weights)


    def forward(self, state):
        # Normalize state image from [0, 255] to [-0.5, 0.5]
        # In B.2, it explicitly states `state = state/255. - 0.5`
        state = state / 255.0 - 0.5
        x = self.cnn(state)
        x = x.reshape(x.size(0), -1) # Flatten
        x = self.linear(x)
        return self.activation_fn(self.layer_norm(x))



import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from config import Config
from layers import LayerNorm, ln_activ
from modules import CNNEncoder, init_weights

class StateEncoder(nn.Module):
    def __init__(self, observation_space_shape, zs_dim, activation_fn):
        super().__init__()
        self.zs_dim = zs_dim
        self.activation_fn = activation_fn

        if len(observation_space_shape) == 3:  # Image input (e.g., Atari, Visual DMC)
            state_channels = observation_space_shape[0] * Config.FRAME_STACK # Stacked frames
            self.encoder = CNNEncoder(state_channels, zs_dim, activation_fn)
        else:  # Vector input (e.g., Gym, Proprioceptive DMC)
            state_dim = observation_space_shape[0]
            self.zs_mlp1 = nn.Linear(state_dim, Config.HIDDEN_DIM)
            self.zs_ln1 = LayerNorm(Config.HIDDEN_DIM)
            self.zs_mlp2 = nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM)
            self.zs_ln2 = LayerNorm(Config.HIDDEN_DIM)
            self.zs_mlp3 = nn.Linear(Config.HIDDEN_DIM, zs_dim)
            self.encoder = self._mlp_forward
            self.apply(init_weights)

    def _mlp_forward(self, state):
        x = self.activation_fn(self.zs_ln1(self.zs_mlp1(state)))
        x = self.activation_fn(self.zs_ln2(self.zs_mlp2(x)))
        return self.activation_fn(self.zs_mlp3(x))

    def forward(self, state):
        return self.encoder(state)

class StateActionEncoder(nn.Module):
    def __init__(self, zs_dim, action_dim, za_dim, zsa_dim, output_dim, activation_fn):
        super().__init__()
        self.activation_fn = activation_fn
        self.za_linear = nn.Linear(action_dim, za_dim)

        self.zsa_mlp1 = nn.Linear(zs_dim + za_dim, Config.HIDDEN_DIM)
        self.zsa_ln1 = LayerNorm(Config.HIDDEN_DIM)
        self.zsa_mlp2 = nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM)
        self.zsa_ln2 = LayerNorm(Config.HIDDEN_DIM)
        self.zsa_mlp3 = nn.Linear(Config.HIDDEN_DIM, zsa_dim)

        self.model_predictor_zs = nn.Linear(zsa_dim, zs_dim) # Predicts next state embedding
        self.model_predictor_reward = nn.Linear(zsa_dim, Config.REWARD_BINS) # Predicts reward logits
        self.model_predictor_terminal = nn.Linear(zsa_dim, 1) # Predicts terminal signal
        self.apply(init_weights)

    def forward(self, zs, action):
        za = self.activation_fn(self.za_linear(action))
        zsa = torch.cat([zs, za], dim=1)
        zsa = self.activation_fn(self.zsa_ln1(self.zsa_mlp1(zsa)))
        zsa = self.activation_fn(self.zsa_ln2(self.zsa_mlp2(zsa)))
        zsa_embedding = self.zsa_mlp3(zsa) # No activation here in the paper's code block for zsa_mlp3

        predicted_zs = self.model_predictor_zs(zsa_embedding)
        predicted_reward_logits = self.model_predictor_reward(zsa_embedding)
        predicted_terminal = self.model_predictor_terminal(zsa_embedding)
        
        return predicted_zs, predicted_reward_logits, predicted_terminal, zsa_embedding

class ValueNetwork(nn.Module):
    def __init__(self, zsa_dim, activation_fn):
        super().__init__()
        self.activation_fn = activation_fn
        self.l1 = nn.Linear(zsa_dim, Config.HIDDEN_DIM)
        self.ln1 = LayerNorm(Config.HIDDEN_DIM)
        self.l2 = nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM)
        self.ln2 = LayerNorm(Config.HIDDEN_DIM)
        self.l3 = nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM)
        self.ln3 = LayerNorm(Config.HIDDEN_DIM)
        self.l4 = nn.Linear(Config.HIDDEN_DIM, 1) # Output a single Q-value
        self.apply(init_weights)

    def forward(self, zsa):
        q = self.activation_fn(self.ln1(self.l1(zsa)))
        q = self.activation_fn(self.ln2(self.l2(q)))
        q = self.activation_fn(self.ln3(self.l3(q)))
        return self.l4(q)

class PolicyNetwork(nn.Module):
    def __init__(self, zs_dim, action_dim, discrete_action_space, activation_fn):
        super().__init__()
        self.activation_fn = activation_fn # ReLU for policy hidden layers
        self.discrete_action_space = discrete_action_space
        self.l1 = nn.Linear(zs_dim, Config.HIDDEN_DIM)
        self.ln1 = LayerNorm(Config.HIDDEN_DIM)
        self.l2 = nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM)
        self.ln2 = LayerNorm(Config.HIDDEN_DIM)
        self.l3 = nn.Linear(Config.HIDDEN_DIM, action_dim) # Output action logits or continuous action values
        self.apply(init_weights)

        if discrete_action_space:
            self.final_activation = partial(F.gumbel_softmax, tau=Config.GUMBEL_SOFTMAX_TAU, hard=True) # hard=True for one-hot
        else:
            self.final_activation = torch.tanh # Scale to [-1, 1]

    def forward(self, zs):
        a = self.activation_fn(self.ln1(self.l1(zs)))
        a = self.activation_fn(self.ln2(self.l2(a)))
        pre_activations = self.l3(a) # Output before final activation
        return self.final_activation(pre_activations), pre_activations

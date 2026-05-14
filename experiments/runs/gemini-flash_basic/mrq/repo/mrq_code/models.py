import torch
import torch.nn as nn
import torch.nn.functional as F

from mrq_code.utils import Identity, LinearNormalizedActivation, init_weights
from mrq_code.config import MRQConfig

class StateEncoder(nn.Module):
    def __init__(self, state_dim, image_observation_space=False, state_channels=3):
        super().__init__()
        self.image_observation_space = image_observation_space
        self.activation_fn = self._get_activation_function(MRQConfig.ACTIVATION_FUNCTION)

        if self.image_observation_space:
            # For image inputs: four convolutional layers
            # Assuming 84x84 input as mentioned in paper
            self.cnn1 = nn.Conv2d(state_channels, 32, 3, stride=2)
            self.cnn2 = nn.Conv2d(32, 32, 3, stride=2)
            self.cnn3 = nn.Conv2d(32, 32, 3, stride=2)
            self.cnn4 = nn.Conv2d(32, 32, 3, stride=1)
            
            # After CNN, a linear layer followed by LayerNorm and activation
            self.linear_layer = nn.Linear(32 * 7 * 7, MRQConfig.ZS_DIM) # 32*7*7 for 84x84 input
            self.ln_activ_final = LinearNormalizedActivation(MRQConfig.ZS_DIM, self.activation_fn)

        else:
            # For vector inputs: three-layer MLP
            self.mlp1 = nn.Linear(state_dim, MRQConfig.HIDDEN_DIM)
            self.ln_activ1 = LinearNormalizedActivation(MRQConfig.HIDDEN_DIM, self.activation_fn)
            self.mlp2 = nn.Linear(MRQConfig.HIDDEN_DIM, MRQConfig.HIDDEN_DIM)
            self.ln_activ2 = LinearNormalizedActivation(MRQConfig.HIDDEN_DIM, self.activation_fn)
            self.mlp3 = nn.Linear(MRQConfig.HIDDEN_DIM, MRQConfig.ZS_DIM)
            self.ln_activ3 = LinearNormalizedActivation(MRQConfig.ZS_DIM, self.activation_fn)

        self.apply(lambda m: init_weights(m, MRQConfig.WEIGHT_INITIALIZATION, MRQConfig.BIAS_INITIALIZATION))

    def _get_activation_function(self, name):
        if name == "ELU":
            return nn.ELU()
        elif name == "ReLU":
            return nn.ReLU()
        else:
            raise ValueError(f"Unknown activation function: {name}")

    def forward(self, state):
        if self.image_observation_space:
            state = state / 255.0 - 0.5 # Normalize input image
            features = self.activation_fn(self.cnn1(state))
            features = self.activation_fn(self.cnn2(features))
            features = self.activation_fn(self.cnn3(features))
            features = self.activation_fn(self.cnn4(features))
            features = features.reshape(features.size(0), -1) # Flatten
            zs = self.linear_layer(features)
            return self.ln_activ_final(zs)
        else:
            zs = self.ln_activ1(self.mlp1(state))
            zs = self.ln_activ2(self.mlp2(zs))
            zs = self.ln_activ3(self.mlp3(zs))
            return zs

class StateActionEncoder(nn.Module):
    def __init__(self, action_dim, zs_dim=MRQConfig.ZS_DIM):
        super().__init__()
        self.activation_fn = self._get_activation_function(MRQConfig.ACTIVATION_FUNCTION)

        self.action_linear = nn.Linear(action_dim, MRQConfig.ZA_DIM)
        
        # MLP layers with Linear then LinearNormalizedActivation (except final)
        self.mlp1 = nn.Linear(zs_dim + MRQConfig.ZA_DIM, MRQConfig.HIDDEN_DIM)
        self.ln_activ1 = LinearNormalizedActivation(MRQConfig.HIDDEN_DIM, self.activation_fn)
        self.mlp2 = nn.Linear(MRQConfig.HIDDEN_DIM, MRQConfig.HIDDEN_DIM)
        self.ln_activ2 = LinearNormalizedActivation(MRQConfig.HIDDEN_DIM, self.activation_fn)
        self.mlp3 = nn.Linear(MRQConfig.HIDDEN_DIM, MRQConfig.ZSA_DIM) # Final output is zsa_dim

        # Linear MDP Predictor - output_dim needs to predict reward (bins), next state embedding (zs_dim), terminal (1)
        self.model = nn.Linear(MRQConfig.ZSA_DIM, MRQConfig.REWARD_BINS + zs_dim + 1) 

        self.apply(lambda m: init_weights(m, MRQConfig.WEIGHT_INITIALIZATION, MRQConfig.BIAS_INITIALIZATION))
    
    def _get_activation_function(self, name):
        if name == "ELU":
            return nn.ELU()
        elif name == "ReLU":
            return nn.ReLU()
        else:
            raise ValueError(f"Unknown activation function: {name}")

    def forward(self, zs, action):
        # Process action input
        processed_action = self.activation_fn(self.action_linear(action))
        
        # Concatenate state embedding and processed action
        zsa_input = torch.cat([zs, processed_action], dim=-1)
        
        # Process through MLP
        zsa = self.ln_activ1(self.mlp1(zsa_input))
        zsa = self.ln_activ2(self.mlp2(zsa))
        zsa = self.mlp3(zsa) # No ln_activ after the last linear layer as per value/policy

        # Predict reward, next state embedding, and terminal signal
        predictions = self.model(zsa)

        # Split predictions into reward logits, next_zs, and terminal_logits
        reward_logits = predictions[..., :MRQConfig.REWARD_BINS]
        next_zs = predictions[..., MRQConfig.REWARD_BINS:-1]
        terminal_logits = predictions[..., -1]
        
        return reward_logits, next_zs, terminal_logits, zsa


class ValueNetwork(nn.Module):
    def __init__(self, zsa_dim=MRQConfig.ZSA_DIM):
        super().__init__()
        self.activation_fn = self._get_activation_function(MRQConfig.ACTIVATION_FUNCTION)

        # 4-layer MLP with Linear then LinearNormalizedActivation (except final output)
        self.l1 = nn.Linear(zsa_dim, MRQConfig.HIDDEN_DIM)
        self.ln_activ1 = LinearNormalizedActivation(MRQConfig.HIDDEN_DIM, self.activation_fn)
        self.l2 = nn.Linear(MRQConfig.HIDDEN_DIM, MRQConfig.HIDDEN_DIM)
        self.ln_activ2 = LinearNormalizedActivation(MRQConfig.HIDDEN_DIM, self.activation_fn)
        self.l3 = nn.Linear(MRQConfig.HIDDEN_DIM, MRQConfig.HIDDEN_DIM)
        self.ln_activ3 = LinearNormalizedActivation(MRQConfig.HIDDEN_DIM, self.activation_fn)
        self.l4 = nn.Linear(MRQConfig.HIDDEN_DIM, 1) # Output is a single Q-value

        self.apply(lambda m: init_weights(m, MRQConfig.WEIGHT_INITIALIZATION, MRQConfig.BIAS_INITIALIZATION))
    
    def _get_activation_function(self, name):
        if name == "ELU":
            return nn.ELU()
        elif name == "ReLU":
            return nn.ReLU()
        else:
            raise ValueError(f"Unknown activation function: {name}")

    def forward(self, zsa):
        q = self.ln_activ1(self.l1(zsa))
        q = self.ln_activ2(self.l2(q))
        q = self.ln_activ3(self.l3(q))
        return self.l4(q)


class PolicyNetwork(nn.Module):
    def __init__(self, action_dim, zs_dim=MRQConfig.ZS_DIM, is_discrete=False):
        super().__init__()
        self.is_discrete = is_discrete
        self.activation_fn = self._get_activation_function("ReLU") # Policy uses ReLU as per paper

        # 3-layer MLP with Linear then LinearNormalizedActivation (except final output)
        self.mlp1 = nn.Linear(zs_dim, MRQConfig.HIDDEN_DIM)
        self.ln_activ1 = LinearNormalizedActivation(MRQConfig.HIDDEN_DIM, self.activation_fn)
        self.mlp2 = nn.Linear(MRQConfig.HIDDEN_DIM, MRQConfig.HIDDEN_DIM)
        self.ln_activ2 = LinearNormalizedActivation(MRQConfig.HIDDEN_DIM, self.activation_fn)
        self.mlp3 = nn.Linear(MRQConfig.HIDDEN_DIM, action_dim) # Output logits for discrete or raw actions for continuous
        
        self.apply(lambda m: init_weights(m, MRQConfig.WEIGHT_INITIALIZATION, MRQConfig.BIAS_INITIALIZATION))
    
    def _get_activation_function(self, name):
        if name == "ELU":
            return nn.ELU()
        elif name == "ReLU":
            return nn.ReLU()
        else:
            raise ValueError(f"Unknown activation function: {name}")

    def forward(self, zs):
        pre_activations = self.ln_activ1(self.mlp1(zs))
        pre_activations = self.ln_activ2(self.mlp2(pre_activations))
        pre_activations = self.mlp3(pre_activations)
        return pre_activations


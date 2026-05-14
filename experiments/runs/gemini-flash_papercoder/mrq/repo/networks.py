import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from typing import Callable, Optional, Tuple, Union

from config import Config
from utils import LayerNormActivation


def _init_weights(m: nn.Module) -> None:
    """Initializes weights with Xavier uniform and biases with zeros."""
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class BaseMLP(nn.Module):
    """A reusable MLP block with LayerNorm and activation after intermediate linear layers.
    The final linear layer does not apply LayerNorm or activation internally.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int, # This is the output dimension of the entire MLP
        hidden_dim: int,
        num_hidden_layers: int, # Number of hidden linear layers before the final linear layer
        activation_fn_factory: Callable[[], nn.Module],
    ) -> None:
        """Initializes the BaseMLP module.

        Args:
            input_dim: The input dimension for the first linear layer.
            output_dim: The output dimension for the final linear layer.
            hidden_dim: The dimension of all hidden layers.
            num_hidden_layers: The number of linear layers that are followed by LayerNorm and activation.
                               A value of 0 means a single linear layer from input_dim to output_dim.
            activation_fn_factory: A callable that returns an instance of the activation function (e.g., nn.ELU).
        """
        super().__init__()
        
        layers = []
        in_dim = input_dim

        # Hidden layers: Linear -> LayerNormActivation
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(LayerNormActivation(hidden_dim, activation_fn_factory()))
            in_dim = hidden_dim
        
        # Final linear layer, no LN+Activation applied internally
        layers.append(nn.Linear(in_dim, output_dim))

        self.model = nn.Sequential(*layers)
        self.apply(_init_weights) # Apply weight initialization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass through the MLP."""
        return self.model(x)


class ImageStateEncoder(nn.Module):
    """CNN-based state encoder for image observations (f_omega).
    Converts raw image observations into a state embedding (zs).
    """
    def __init__(
        self,
        obs_shape: Tuple[int, int, int], # (channels, height, width)
        zs_dim: int,
        activation_fn_factory: Callable[[], nn.Module], # Factory for activation e.g. nn.ELU
    ) -> None:
        """Initializes the ImageStateEncoder.

        Args:
            obs_shape: The shape of the input image observations (C, H, W).
            zs_dim: The dimension of the output state embedding.
            activation_fn_factory: A callable that returns an instance of the activation function.
        """
        super().__init__()
        channels, height, width = obs_shape
        
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=2),
            activation_fn_factory(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            activation_fn_factory(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            activation_fn_factory(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            activation_fn_factory(), # Last conv also has activation before flattening
        )
        
        # Calculate flattened output size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_shape)
            flattened_size = self.cnn_layers(dummy_input).view(1, -1).shape[1]
        
        # Paper states 1568 for 84x84 input (assuming it after 4 convs with given params).
        # We use dynamic calculation but log a warning if it deviates from paper's assumption.
        if height == 84 and width == 84:
            # Paper's Appendix B.2 indicates 1568 for 84x84 input.
            self.flattened_size = 1568
            if flattened_size != self.flattened_size:
                print(f"Warning: Calculated flattened size {flattened_size} for 84x84 input "
                      f"does not match paper's {self.flattened_size}. "
                      "Check CNN layer definitions or input shape assumptions.")
        else:
            self.flattened_size = flattened_size
            print(f"Info: Input image size is not 84x84 ({obs_shape}). Dynamically calculated flattened size: {self.flattened_size}.")

        self.zs_lin = nn.Linear(self.flattened_size, zs_dim)
        # The paper specifies LayerNorm followed by ELU activation after the final linear layer
        self.ln_activ_final = LayerNormActivation(zs_dim, activation_fn_factory())

        self.apply(_init_weights)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass for image state encoding.

        Args:
            state: Input tensor of raw image observations (batch_size, C, H, W).

        Returns:
            The state embedding (batch_size, zs_dim).
        """
        # Normalize image input (from 0-255 to -0.5 to 0.5 range)
        state = state / 255.0 - 0.5
        
        cnn_output = self.cnn_layers(state)
        # Flatten the output from convolutional layers
        flattened_output = cnn_output.view(state.shape[0], -1)
        
        zs_lin_output = self.zs_lin(flattened_output)
        zs = self.ln_activ_final(zs_lin_output)
        return zs


class VectorStateEncoder(nn.Module):
    """MLP-based state encoder for vector observations (f_omega).
    Converts raw vector observations into a state embedding (zs).
    """
    def __init__(
        self,
        obs_dim: int,
        zs_dim: int,
        hidden_dim: int, # config.network.hidden_dim
        activation_fn_factory: Callable[[], nn.Module], # e.g. nn.ELU
    ) -> None:
        """Initializes the VectorStateEncoder.

        Args:
            obs_dim: The dimension of the input vector observations.
            zs_dim: The dimension of the output state embedding.
            hidden_dim: The dimension of the hidden layers in the MLP.
            activation_fn_factory: A callable that returns an instance of the activation function.
        """
        super().__init__()
        # Paper snippet for vector input:
        # zs = self.ln_activ(self.zs_mlp1(state))
        # zs = self.ln_activ(self.zs_mlp2(zs))
        # return self.ln_activ(self.zs_mlp3(zs))
        # This implies 3 linear layers, each followed by LayerNorm and Activation.
        self.mlp1 = nn.Linear(obs_dim, hidden_dim)
        self.ln_activ1 = LayerNormActivation(hidden_dim, activation_fn_factory())
        self.mlp2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln_activ2 = LayerNormActivation(hidden_dim, activation_fn_factory())
        self.mlp3 = nn.Linear(hidden_dim, zs_dim)
        self.ln_activ3 = LayerNormActivation(zs_dim, activation_fn_factory())
        
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass for vector state encoding.

        Args:
            x: Input tensor of raw vector observations (batch_size, obs_dim).

        Returns:
            The state embedding (batch_size, zs_dim).
        """
        x = self.ln_activ1(self.mlp1(x))
        x = self.ln_activ2(self.mlp2(x))
        zs = self.ln_activ3(self.mlp3(x)) # Final LN+Activ as per paper snippet
        return zs


class StateActionEncoder(nn.Module):
    """Implements the state-action encoder (g_omega) and the linear MDP predictor (m).
    Takes a state embedding and action, produces a state-action embedding, and predicts
    the next state embedding, categorical reward logits, and terminal logits.
    """
    def __init__(
        self,
        zs_dim: int,
        action_dim: int,
        za_dim: int, # config.network.za_dim
        zsa_dim: int, # config.network.zsa_dim
        hidden_dim: int, # config.network.hidden_dim
        reward_bins: int, # config.reward_processing.reward_bins
        activation_fn_factory: Callable[[], nn.Module], # e.g. nn.ELU
    ) -> None:
        """Initializes the StateActionEncoder.

        Args:
            zs_dim: Dimension of the input state embedding.
            action_dim: Dimension of the input action.
            za_dim: Dimension of the processed action embedding.
            zsa_dim: Dimension of the output state-action embedding.
            hidden_dim: Dimension of hidden layers in the MLP.
            reward_bins: Number of bins for categorical reward prediction.
            activation_fn_factory: A callable that returns an instance of the activation function.
        """
        super().__init__()
        self.zs_dim = zs_dim
        self.reward_bins = reward_bins
        self.activation_fn = activation_fn_factory() # Store the instantiated activation function

        # Action processing (za)
        self.za_linear = nn.Linear(action_dim, za_dim)

        # MLP to generate zsa_embedding
        # Paper snippet for g_omega MLP:
        # zsa = self.ln_activ(self.zsa1(zsa))
        # zsa = self.ln_activ(self.zsa2(zsa))
        # zsa = self.zsa3(zsa)
        # This matches BaseMLP with num_hidden_layers=2.
        self.zsa_mlp = BaseMLP(
            input_dim=zs_dim + za_dim,
            output_dim=zsa_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=2, # Corresponds to zsa1 and zsa2 being followed by LN+Act
            activation_fn_factory=activation_fn_factory,
        )

        # Linear MDP Predictor (self.model as per paper snippet)
        # Predicts next_zs_embedding (zs_dim), reward_logits (reward_bins), terminal_logit (1)
        # Output dimension: zs_dim (for next state) + reward_bins (for categorical reward) + 1 (for terminal)
        self.mdp_predictor = nn.Linear(zsa_dim, zs_dim + reward_bins + 1)

        self.apply(_init_weights)

    def forward(self, zs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass for state-action encoding and MDP prediction.

        Args:
            zs: State embedding (batch_size, zs_dim).
            action: Action (batch_size, action_dim).

        Returns:
            A tuple containing:
            - predicted_zs_prime: Predicted next state embedding (batch_size, zs_dim).
            - predicted_reward_logits: Logits for categorical reward (batch_size, reward_bins).
            - predicted_terminal_logits: Logit for terminal signal (batch_size, 1).
            - zsa_embedding: The state-action embedding itself (batch_size, zsa_dim).
        """
        # Process action with activation
        za = self.activation_fn(self.za_linear(action))
        
        # Concatenate state embedding and processed action
        z_sa_input = torch.cat([zs, za], dim=-1)
        
        # Generate state-action embedding
        zsa_embedding = self.zsa_mlp(z_sa_input)

        # Predict MDP components
        predicted_all = self.mdp_predictor(zsa_embedding)
        
        predicted_zs_prime = predicted_all[..., :self.zs_dim]
        predicted_reward_logits = predicted_all[..., self.zs_dim:self.zs_dim + self.reward_bins]
        predicted_terminal_logits = predicted_all[..., self.zs_dim + self.reward_bins:]

        return predicted_zs_prime, predicted_reward_logits, predicted_terminal_logits, zsa_embedding


class ValueNetwork(nn.Module):
    """Implements a Q-value network (Q_theta).
    Predicts a scalar Q-value from a state-action embedding. Two such networks are used.
    """
    def __init__(
        self,
        zsa_dim: int,
        hidden_dim: int, # config.network.hidden_dim
        activation_fn_factory: Callable[[], nn.Module], # e.g. nn.ELU
    ) -> None:
        """Initializes the ValueNetwork.

        Args:
            zsa_dim: Dimension of the input state-action embedding.
            hidden_dim: Dimension of hidden layers in the MLP.
            activation_fn_factory: A callable that returns an instance of the activation function.
        """
        super().__init__()
        # Paper snippet for Value Q Networks:
        # q = self.ln_activ(self.l1(zsa))
        # q = self.ln_activ(self.l2(q))
        # q = self.ln_activ(self.l3(q))
        # return self.l4(q)
        # This matches BaseMLP with num_hidden_layers=3.
        self.mlp = BaseMLP(
            input_dim=zsa_dim,
            output_dim=1, # Scalar Q-value
            hidden_dim=hidden_dim,
            num_hidden_layers=3, # Corresponds to l1, l2, l3 being followed by LN+Act
            activation_fn_factory=activation_fn_factory,
        )
        self.apply(_init_weights)

    def forward(self, zsa: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass to output a scalar Q-value.

        Args:
            zsa: Input state-action embedding (batch_size, zsa_dim).

        Returns:
            A tensor of scalar Q-values (batch_size, 1).
        """
        return self.mlp(zsa)


class PolicyNetwork(nn.Module):
    """Implements the policy network (pi_phi).
    Predicts actions from a state embedding, supporting both discrete and continuous action spaces.
    Includes logic for exploration noise.
    """
    def __init__(
        self,
        zs_dim: int,
        action_dim: int,
        hidden_dim: int, # config.network.hidden_dim
        discrete_actions: bool,
        gumbel_tau: float, # config.policy.gumbel_tau
        activation_fn_factory: Callable[[], nn.Module], # e.g. nn.ReLU
    ) -> None:
        """Initializes the PolicyNetwork.

        Args:
            zs_dim: Dimension of the input state embedding.
            action_dim: Dimension of the output action space.
            hidden_dim: Dimension of hidden layers in the MLP.
            discrete_actions: Boolean indicating if the action space is discrete.
            gumbel_tau: Temperature parameter for Gumbel-Softmax (for discrete actions).
            activation_fn_factory: A callable that returns an instance of the activation function for hidden layers.
        """
        super().__init__()
        self.action_dim = action_dim
        self.discrete_actions = discrete_actions
        self.gumbel_tau = gumbel_tau

        # MLP to generate pre-activations (logits)
        # Paper snippet for Policy pi Network:
        # a = self.ln_activ(self.l1(zs))
        # a = self.ln_activ(self.l2(a))
        # return self.final_activ(self.l3(a))
        # This matches BaseMLP with num_hidden_layers=2.
        self.mlp = BaseMLP(
            input_dim=zs_dim,
            output_dim=action_dim, # Output logits for actions (pre-activations)
            hidden_dim=hidden_dim,
            num_hidden_layers=2, # Corresponds to l1 and l2 being followed by LN+Act
            activation_fn_factory=activation_fn_factory,
        )

        if self.discrete_actions:
            self.final_activ = functools.partial(F.gumbel_softmax, tau=self.gumbel_tau, hard=False)
        else: # Continuous actions
            self.final_activ = torch.tanh
        
        self.apply(_init_weights)

    def forward(self, zs: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass to output pre-activations (logits) for actions.

        Args:
            zs: Input state embedding (batch_size, zs_dim).

        Returns:
            A tensor of action pre-activations/logits (batch_size, action_dim).
        """
        pre_activations = self.mlp(zs)
        return pre_activations

    def act(self, zs: torch.Tensor, add_noise: bool, std: float, policy_noise_clip: Optional[float] = None) -> torch.Tensor:
        """
        Selects an action based on the state embedding, with optional exploration noise.
        This method is used for collecting experience and for the target policy in value updates.

        Args:
            zs: State embedding (batch_size, zs_dim).
            add_noise: Whether to add exploration noise.
            std: Standard deviation for Gaussian noise.
            policy_noise_clip: Clipping value for continuous action noise. Used for target policy noise.

        Returns:
            Selected action (batch_size, action_dim for continuous, or batch_size for discrete index).
        """
        pre_activations = self.forward(zs) # Get logits/pre-activations

        if self.discrete_actions:
            # Paper: "For exploration, Gaussian noise is added to each dimension of the action
            # (or one-hot encoding of the action). ... For discrete actions, the final action is
            # determined by the argmax operation."
            # Also for target policy noise: "Discrete actions are represented by a one-hot encoding,
            # where the Gaussian noise is added to each dimension."
            # This implies adding noise to the logits and then taking argmax.
            if add_noise:
                noise = torch.randn_like(pre_activations, device=zs.device) * std
                noisy_pre_activations = pre_activations + noise
                return torch.argmax(noisy_pre_activations, dim=-1)
            else:
                return torch.argmax(pre_activations, dim=-1)
        else: # Continuous actions
            raw_action = self.final_activ(pre_activations) # Tanh activation maps to [-1, 1]
            if add_noise:
                noise = torch.randn_like(raw_action, device=zs.device) * std
                if policy_noise_clip is not None:
                    noise = torch.clamp(noise, -policy_noise_clip, policy_noise_clip)
                noisy_action = raw_action + noise
                # Clip continuous actions to [-1, 1] range after adding noise
                action = torch.clamp(noisy_action, -1.0, 1.0)
                return action
            else:
                return raw_action


class Models:
    """A container class managing all neural networks (main and target) and their optimizers."""
    def __init__(
        self,
        config: Config,
        obs_space_info: dict, # Contains 'obs_dim' for vector or 'obs_shape' for image
        action_space_info: dict, # Contains 'action_dim' and 'discrete_actions' boolean
        device: torch.device,
    ) -> None:
        """Initializes the Models container, creating all networks and optimizers.

        Args:
            config: Configuration object holding all hyperparameters.
            obs_space_info: Dictionary with observation space details (e.g., shape/dim).
            action_space_info: Dictionary with action space details (e.g., dim, discrete flag).
            device: The PyTorch device (e.g., 'cuda', 'cpu') to place the models on.
        """
        self.config = config
        self.device = device

        # Map activation function names from config to PyTorch modules
        activation_map = {"ELU": nn.ELU, "ReLU": nn.ReLU}
        
        # Determine activation function factory for encoders and value networks
        encoder_value_activation_fn_factory = activation_map.get(
            config.network.encoder_value_activation, nn.ELU
        )
        if config.network.encoder_value_activation not in activation_map:
             print(f"Warning: Unknown encoder_value_activation '{config.network.encoder_value_activation}'. Using nn.ELU as default.")

        # Determine activation function factory for policy network
        policy_activation_fn_factory = activation_map.get(
            config.network.policy_activation, nn.ReLU
        )
        if config.network.policy_activation not in activation_map:
            print(f"Warning: Unknown policy_activation '{config.network.policy_activation}'. Using nn.ReLU as default.")


        # 1. State Encoders (f_omega and f_omega_target)
        # Dynamically choose between ImageStateEncoder and VectorStateEncoder based on config.
        self.is_image_obs = config.environment.image_obs
        if self.is_image_obs:
            self.state_encoder: Union[ImageStateEncoder, VectorStateEncoder] = ImageStateEncoder(
                obs_shape=obs_space_info["obs_shape"],
                zs_dim=config.network.zs_dim,
                activation_fn_factory=encoder_value_activation_fn_factory,
            )
            self.target_state_encoder: Union[ImageStateEncoder, VectorStateEncoder] = ImageStateEncoder(
                obs_shape=obs_space_info["obs_shape"],
                zs_dim=config.network.zs_dim,
                activation_fn_factory=encoder_value_activation_fn_factory,
            )
        else: # Vector observations
            self.state_encoder = VectorStateEncoder(
                obs_dim=obs_space_info["obs_dim"],
                zs_dim=config.network.zs_dim,
                hidden_dim=config.network.hidden_dim,
                activation_fn_factory=encoder_value_activation_fn_factory,
            )
            self.target_state_encoder = VectorStateEncoder(
                obs_dim=obs_space_info["obs_dim"],
                zs_dim=config.network.zs_dim,
                hidden_dim=config.network.hidden_dim,
                activation_fn_factory=encoder_value_activation_fn_factory,
            )
        
        # 2. State-Action Encoder (g_omega and m)
        # As per the design and analysis in the paper, g_omega (StateActionEncoder) does NOT have a separate target network.
        # Its parameters are part of 'omega' which is updated by the encoder optimizer.
        self.state_action_encoder = StateActionEncoder(
            zs_dim=config.network.zs_dim,
            action_dim=action_space_info["action_dim"],
            za_dim=config.network.za_dim,
            zsa_dim=config.network.zsa_dim,
            hidden_dim=config.network.hidden_dim,
            reward_bins=config.reward_processing.reward_bins,
            activation_fn_factory=encoder_value_activation_fn_factory,
        )

        # 3. Value Networks (Q_theta and Q_theta_target) - Two Q-networks as in TD3
        self.value_net1 = ValueNetwork(
            zsa_dim=config.network.zsa_dim,
            hidden_dim=config.network.hidden_dim,
            activation_fn_factory=encoder_value_activation_fn_factory,
        )
        self.value_net2 = ValueNetwork(
            zsa_dim=config.network.zsa_dim,
            hidden_dim=config.network.hidden_dim,
            activation_fn_factory=encoder_value_activation_fn_factory,
        )
        self.target_value_net1 = ValueNetwork(
            zsa_dim=config.network.zsa_dim,
            hidden_dim=config.network.hidden_dim,
            activation_fn_factory=encoder_value_activation_fn_factory,
        )
        self.target_value_net2 = ValueNetwork(
            zsa_dim=config.network.zsa_dim,
            hidden_dim=config.network.hidden_dim,
            activation_fn_factory=encoder_value_activation_fn_factory,
        )

        # 4. Policy Network (pi_phi and pi_phi_target)
        self.policy_net = PolicyNetwork(
            zs_dim=config.network.zs_dim,
            action_dim=action_space_info["action_dim"],
            hidden_dim=config.network.hidden_dim,
            discrete_actions=action_space_info["discrete_actions"],
            gumbel_tau=config.policy.gumbel_tau,
            activation_fn_factory=policy_activation_fn_factory,
        )
        self.target_policy_net = PolicyNetwork(
            zs_dim=config.network.zs_dim,
            action_dim=action_space_info["action_dim"],
            hidden_dim=config.network.hidden_dim,
            discrete_actions=action_space_info["discrete_actions"],
            gumbel_tau=config.policy.gumbel_tau,
            activation_fn_factory=policy_activation_fn_factory,
        )

        # Initialize target networks by copying main network parameters
        self.update_targets(initial_copy=True)

        # Optimizers
        # Encoder optimizer: parameters of f_omega (self.state_encoder) and g_omega (self.state_action_encoder)
        encoder_params = list(self.state_encoder.parameters()) + \
                         list(self.state_action_encoder.parameters())
        self.encoder_optimizer = torch.optim.AdamW(
            encoder_params,
            lr=config.optimizer.learning_rate_encoders,
            weight_decay=config.optimizer.weight_decay,
        )

        # Value optimizer: parameters of Q_theta1 and Q_theta2 (self.value_net1, self.value_net2)
        value_params = list(self.value_net1.parameters()) + \
                       list(self.value_net2.parameters())
        self.value_optimizer = torch.optim.AdamW(
            value_params,
            lr=config.optimizer.learning_rate_rl,
            weight_decay=config.optimizer.weight_decay,
        )

        # Policy optimizer: parameters of pi_phi (self.policy_net)
        policy_params = list(self.policy_net.parameters())
        self.policy_optimizer = torch.optim.AdamW(
            policy_params,
            lr=config.optimizer.learning_rate_rl,
            weight_decay=config.optimizer.weight_decay,
        )
        
        self.optimizers = {
            "encoder": self.encoder_optimizer,
            "value": self.value_optimizer,
            "policy": self.policy_optimizer,
        }

        # Move all networks to the specified device
        self.to(self.device)

    def to(self, device: torch.device) -> None:
        """Moves all networks to the specified device."""
        self.state_encoder.to(device)
        self.target_state_encoder.to(device)
        self.state_action_encoder.to(device)
        self.value_net1.to(device)
        self.value_net2.to(device)
        self.target_value_net1.to(device)
        self.target_value_net2.to(device)
        self.policy_net.to(device)
        self.target_policy_net.to(device)

    def update_targets(self, initial_copy: bool = False) -> None:
        """
        Updates target network parameters by performing a hard copy from their
        corresponding main networks. This synchronizes target network weights.

        Args:
            initial_copy: A boolean flag, typically True for the first copy at initialization.
                          It doesn't change behavior for hard updates but can be used for logging.
        """
        # Hard update (copy parameters directly)
        self.target_state_encoder.load_state_dict(self.state_encoder.state_dict())
        self.target_value_net1.load_state_dict(self.value_net1.state_dict())
        self.target_value_net2.load_state_dict(self.value_net2.state_dict())
        self.target_policy_net.load_state_dict(self.policy_net.state_dict())


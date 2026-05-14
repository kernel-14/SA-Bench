
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Union

from modules import MLP, CNNEncoder, UNetDiffusionModel, ValueNetwork, FeatureEncoder, ForwardDynamicsModel, RandomNetworkDistillation, CTSDensityModel, ECOModule
from config import cfg

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class Critic(nn.Module):
    """
    Critic network (Q-function) for SAC/REDQ.
    Implements an ensemble of Q-networks as typically used in REDQ.
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, num_layers: int, num_q_networks: int = 2):
        super().__init__()
        self.num_q_networks = num_q_networks
        self.q_networks = nn.ModuleList()
        for _ in range(num_q_networks):
            self.q_networks.append(ValueNetwork(obs_dim, action_dim, hidden_dim, num_layers))

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # Returns all Q-values from the ensemble
        return torch.stack([q_net(obs, action) for q_net in self.q_networks], dim=0) # (num_q_networks, batch_size, 1)

    def q_min(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # Returns the minimum Q-value across the ensemble (used in REDQ for TD target)
        return self.forward(obs, action).min(dim=0).values # (batch_size, 1)


class Actor(nn.Module):
    """
    Actor network for SAC/REDQ.
    Outputs mean and log_std for a Gaussian policy.
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, num_layers: int, max_action: float):
        super().__init__()
        self.log_std_min = LOG_STD_MIN
        self.log_std_max = LOG_STD_MAX
        self.max_action = max_action

        # Base MLP for feature extraction
        self.net = MLP(obs_dim, hidden_dim, hidden_dim, num_layers)

        # Output layers for mean and log_std
        self.mu_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor, deterministic: bool = False,
                with_logprob: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        h = self.net(obs)
        mu = self.mu_layer(h)
        log_std = self.log_std_layer(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        pi_distribution = torch.distributions.Normal(mu, std)

        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample() # Reparameterization trick

        if with_logprob:
            log_prob = pi_distribution.log_prob(pi_action).sum(axis=-1, keepdim=True)
            # Apply correction for Tanh squashing
            log_prob -= torch.log(self.max_action * (1 - torch.tanh(pi_action).pow(2)) + 1e-6).sum(axis=-1, keepdim=True)
        else:
            log_prob = None

        # Scale action to environment bounds
        squashed_action = self.max_action * torch.tanh(pi_action)
        return squashed_action, log_prob


class PolicyAgent(nn.Module):
    """
    The main policy agent that interacts with the environment.
    Can use either state-based or pixel-based observations.
    """
    def __init__(self, observation_space: Any, action_space: Any, device: torch.device):
        super().__init__()
        self.device = device
        self.obs_type = cfg.OBS_TYPE
        self.max_action = action_space.high[0]

        if self.obs_type == "pixel":
            self.encoder = CNNEncoder(observation_space.shape, cfg.CNN_ENCODER_OUT_DIM).to(device)
            obs_dim = cfg.CNN_ENCODER_OUT_DIM
        else:
            self.encoder = nn.Identity() # Placeholder for state-based, no separate encoder needed
            obs_dim = observation_space.shape[0]

        self.action_dim = action_space.shape[0]

        self.actor = Actor(obs_dim, self.action_dim, cfg.POLICY_HIDDEN_DIM, cfg.POLICY_HIDDEN_LAYERS, self.max_action).to(device)
        self.critic = Critic(obs_dim, self.action_dim, cfg.POLICY_HIDDEN_DIM, cfg.POLICY_HIDDEN_LAYERS).to(device)
        self.critic_target = Critic(obs_dim, self.action_dim, cfg.POLICY_HIDDEN_DIM, cfg.POLICY_HIDDEN_LAYERS).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for param in self.critic_target.parameters():
            param.requires_grad = False

        self.log_alpha = nn.Parameter(torch.zeros(1, requires_grad=True, device=device))
        self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(device)).item()

    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            if self.obs_type == "pixel":
                obs_tensor = self.encoder(obs_tensor.unsqueeze(0)) # Add batch dim, encode
            else:
                obs_tensor = obs_tensor.unsqueeze(0) # Add batch dim

            action, _ = self.actor(obs_tensor, deterministic, with_logprob=False)
        return action.cpu().numpy().flatten()

    def update_critic_target(self):
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(cfg.TAU * param.data + (1 - cfg.TAU) * target_param.data)


class ConditionalDiffusionModel(nn.Module):
    """
    Wrapper for the UNetDiffusionModel to handle training and sampling.
    Implements DDPM (Denoising Diffusion Probabilistic Models) logic.
    """
    def __init__(self,
                 transition_dim: int, # Dimension of (s, a, s', r) flattened
                 time_embed_dim: int,
                 cond_embed_dim: int,
                 diffusion_steps: int = 1000,
                 model_capacity_factor: float = 1.0,
                 device: torch.device = "cpu"):
        super().__init__()
        self.denoise_model = UNetDiffusionModel(
            input_dim=transition_dim,
            output_dim=transition_dim,
            time_embed_dim=time_embed_dim,
            cond_embed_dim=cond_embed_dim,
            model_capacity_factor=model_capacity_factor
        ).to(device)
        self.diffusion_steps = diffusion_steps
        self.device = device

        # Linear schedule for beta (noise variance)
        betas = torch.linspace(1e-4, 0.02, diffusion_steps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0) # alpha_0_bar is 1
        
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # Calculations for diffusion process
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu()).to(self.device)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start: torch.Tensor, conditions: torch.Tensor, t: Optional[torch.Tensor] = None,
                 uncond_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        batch_size = x_start.shape[0]
        if t is None:
            t = torch.randint(0, self.diffusion_steps, (batch_size,), device=self.device).long()
        
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        # Predict noise
        predicted_noise = self.denoise_model(x_noisy, t, conditions, uncond_mask)
        
        loss = F.mse_loss(noise, predicted_noise)
        return loss

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: torch.Tensor, conditions: torch.Tensor,
                 guidance_scale: float = 1.0) -> torch.Tensor:
        
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas_cumprod, t, x.shape)

        # Classifier-free guidance
        if guidance_scale == 1.0: # No guidance, standard conditional generation
            model_mean = self.denoise_model(x, t, conditions)
        else: # Apply guidance
            uncond_pred = self.denoise_model(x, t, torch.zeros_like(conditions)) # Predict with null condition
            cond_pred = self.denoise_model(x, t, conditions) # Predict with actual condition
            model_output = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
            # The model_output is the predicted noise.
            # We use this predicted noise to estimate x_0, then the mean for reverse step
            # Note: DDPM predicts noise, not x_0 directly
            
            # Reconstruct x_0 from x_t and predicted noise
            # x_0 = (x_t - sqrt(1-alpha_cumprod_t) * noise) / sqrt(alpha_cumprod_t)
            x_recon = sqrt_recip_alphas_t * x - sqrt_recipm1_alphas_cumprod_t * model_output
            
            # Use estimated x_0 to compute the mean of the reverse process
            # mean = posterior_mean_coef1 * x_0 + posterior_mean_coef2 * x_t
            posterior_mean_coef1_t = self._extract(self.posterior_mean_coef1, t, x.shape)
            posterior_mean_coef2_t = self._extract(self.posterior_mean_coef2, t, x.shape)
            
            model_mean = posterior_mean_coef1_t * x_recon + posterior_mean_coef2_t * x

        posterior_log_variance_t = self._extract(self.posterior_log_variance_clipped, t, x.shape)
        
        if t[0] == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.exp(0.5 * posterior_log_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape: Tuple[int, ...], conditions: torch.Tensor,
                      guidance_scale: float = 1.0) -> torch.Tensor:
        img = torch.randn(shape, device=self.device)
        for i in reversed(range(0, self.diffusion_steps)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            img = self.p_sample(img, t, conditions, guidance_scale)
        return img

    @torch.no_grad()
    def sample(self, num_samples: int, conditions: torch.Tensor,
               guidance_scale: float = 1.0) -> torch.Tensor:
        batch_size = conditions.shape[0]
        assert batch_size == num_samples, "Number of samples must match number of conditions"
        
        # The input_dim of the diffusion model is the flattened dimension of (s,a,s',r)
        # The shape for sampling should be (batch_size, transition_dim)
        return self.p_sample_loop(
            (batch_size, self.denoise_model.input_dim),
            conditions,
            guidance_scale
        )

# ================================================
# Relevance Functions (F)
# ================================================

class RelevanceFunction(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, transitions: Dict[str, torch.Tensor], agent: PolicyAgent) -> torch.Tensor:
        raise NotImplementedError

class ReturnRelevance(RelevanceFunction):
    """
    Relevance function based on the Q-value estimate of the current policy.
    F(s, a, s', r) = Q(s, pi(s))
    """
    def __init__(self):
        super().__init__()

    def forward(self, transitions: Dict[str, torch.Tensor], agent: PolicyAgent) -> torch.Tensor:
        with torch.no_grad():
            obs = transitions["observations"]
            # If pixel-based, encode observations
            if cfg.OBS_TYPE == "pixel":
                obs = agent.encoder(obs)
            
            # Get action from current policy for Q-value estimation
            policy_action, _ = agent.actor(obs, deterministic=True, with_logprob=False)
            
            # Use one of the Q-networks (e.g., the first one)
            q_values = agent.critic.q_networks[0](obs, policy_action)
        return q_values.squeeze(-1) # Return a scalar for each transition

class TDErrorRelevance(RelevanceFunction):
    """
    Relevance function based on Temporal Difference (TD) error.
    F = r + gamma * Q_target(s', a') - Q(s, a)
    """
    def __init__(self):
        super().__init__()

    def forward(self, transitions: Dict[str, torch.Tensor], agent: PolicyAgent) -> torch.Tensor:
        with torch.no_grad():
            obs, actions, rewards, next_obs, dones = (
                transitions["observations"],
                transitions["actions"],
                transitions["rewards"],
                transitions["next_observations"],
                transitions["dones"]
            )

            if cfg.OBS_TYPE == "pixel":
                obs = agent.encoder(obs)
                next_obs = agent.encoder(next_obs)

            # Calculate target Q-value
            next_actions, next_log_probs = agent.actor(next_obs)
            target_q_values = agent.critic_target.q_min(next_obs, next_actions)
            target_q = rewards + (1 - dones) * cfg.DISCOUNT * (target_q_values - agent.log_alpha.exp() * next_log_probs)

            # Current Q-value (using one of the Q-networks)
            current_q = agent.critic.q_networks[0](obs, actions)

            td_error = (target_q - current_q).abs()
        return td_error.squeeze(-1)

class CuriosityRelevance(RelevanceFunction):
    """
    Relevance function based on Intrinsic Curiosity Module (ICM) prediction error.
    F(s, a, s', r) = 0.5 * ||g(h(s), a) - h(s')||^2
    """
    def __init__(self, observation_space: Any, action_dim: int, device: torch.device):
        super().__init__()
        self.feature_encoder = FeatureEncoder(observation_space, cfg.ICM_FEATURE_DIM).to(device) # h in paper
        self.forward_dynamics_model = ForwardDynamicsModel(cfg.ICM_FEATURE_DIM, action_dim, cfg.ICM_HIDDEN_DIM).to(device) # g in paper
        self.optimizer = torch.optim.Adam(
            list(self.feature_encoder.parameters()) + list(self.forward_dynamics_model.parameters()),
            lr=cfg.ICM_LR
        )

    def forward(self, transitions: Dict[str, torch.Tensor], agent: Optional[PolicyAgent] = None) -> torch.Tensor:
        # agent parameter is not used here but kept for consistent API
        obs, actions, next_obs = transitions["observations"], transitions["actions"], transitions["next_observations"]

        # Encode observations to latent space
        with torch.no_grad(): # No gradient updates when just computing relevance
            latent_obs = self.feature_encoder(obs)
            latent_next_obs = self.feature_encoder(next_obs)
            
            # Predict next latent state using forward dynamics model
            predicted_latent_next_obs = self.forward_dynamics_model(latent_obs, actions)
            
            # Calculate prediction error
            curiosity_loss = F.mse_loss(predicted_latent_next_obs, latent_next_obs, reduction='none').sum(dim=-1, keepdim=True)
        
        return curiosity_loss.squeeze(-1) # Return scalar error for each transition

    def update(self, transitions: Dict[str, torch.Tensor]):
        obs, actions, next_obs = transitions["observations"], transitions["actions"], transitions["next_observations"]

        self.optimizer.zero_grad()

        latent_obs = self.feature_encoder(obs)
        latent_next_obs = self.feature_encoder(next_obs)
        predicted_latent_next_obs = self.forward_dynamics_model(latent_obs, actions)
        
        loss = F.mse_loss(predicted_latent_next_obs, latent_next_obs)
        loss.backward()
        self.optimizer.step()
        return loss.item()

class RNDRelevance(RelevanceFunction):
    """
    Relevance function based on Random Network Distillation (RND).
    F(s, a, s', r) = 0.5 * ||predictor(s') - target(s')||^2
    """
    def __init__(self, observation_space: Any, device: torch.device):
        super().__init__()
        self.rnd = RandomNetworkDistillation(observation_space, cfg.RND_FEATURE_DIM, cfg.RND_LATENT_DIM).to(device)
        self.optimizer = torch.optim.Adam(self.rnd.predictor_net.parameters(), lr=cfg.RND_LR)

    def forward(self, transitions: Dict[str, torch.Tensor], agent: Optional[PolicyAgent] = None) -> torch.Tensor:
        # agent parameter is not used here but kept for consistent API
        next_obs = transitions["next_observations"]
        
        with torch.no_grad():
            predicted_features, target_features = self.rnd(next_obs)
            rnd_error = F.mse_loss(predicted_features, target_features, reduction='none').sum(dim=-1, keepdim=True)
        return rnd_error.squeeze(-1)

    def update(self, transitions: Dict[str, torch.Tensor]):
        next_obs = transitions["next_observations"]
        self.optimizer.zero_grad()
        predicted_features, target_features = self.rnd(next_obs)
        loss = F.mse_loss(predicted_features, target_features)
        loss.backward()
        self.optimizer.step()
        return loss.item()

class CTSRelevance(RelevanceFunction):
    """
    Relevance function based on Context-Tree Switching (CTS) Density Model.
    F(s, a, s', r) = (N_hat(s, a) + 0.01)^(-0.5)
    Note: This is a conceptual implementation as CTS is not a neural network.
    It would typically involve a separate, specialized library.
    For this reproduction, it will return a placeholder value for novelty.
    """
    def __init__(self, image_size: int, context_bins: int):
        super().__init__()
        self.cts_model = CTSDensityModel(image_size, context_bins)
        # No optimizer as it's not a neural network being optimized by gradient descent.

    def forward(self, transitions: Dict[str, torch.Tensor], agent: Optional[PolicyAgent] = None) -> torch.Tensor:
        # For simplicity, we assume observation is a state, and we combine with action
        # In a real CTS implementation, the image_size and context_bins would be used.
        # This is a placeholder that always returns 1.0 (indicating high novelty for all)
        batch_size = transitions["observations"].shape[0]
        # In a full implementation, you would iterate through the batch,
        # convert obs, action to appropriate format, query CTS for N_hat, and compute F.
        
        # Placeholder: Assume all transitions are equally novel for now
        # until a concrete CTS implementation is integrated.
        return torch.ones(batch_size, device=transitions["observations"].device)

    def update(self, transitions: Dict[str, torch.Tensor]):
        # In a full implementation, you would update the CTS model here
        # by feeding it the new transitions to update counts/densities.
        return 0.0 # Placeholder loss, as no NN update

class ECORelevance(RelevanceFunction):
    """
    Relevance function based on Episodic Curiosity (ECO).
    F(s, a, s', r) = alpha * (beta - F_percentile(C(E(s), E(s_i)))) for all s_i in M
    """
    def __init__(self, observation_space: Any, device: torch.device):
        super().__init__()
        self.eco_module = ECOModule(
            observation_space,
            cfg.ECO_EMBEDDER_OUT_DIM,
            cfg.ECO_MLP_LAYERS,
            cfg.ECO_MLP_DIM,
            cfg.ECO_MEMORY_SIZE
        ).to(device)
        self.optimizer = torch.optim.Adam(self.eco_module.parameters(), lr=cfg.ECO_LR)
        self.alpha = cfg.ECO_ALPHA
        self.beta = cfg.ECO_BETA
        self.percentile = cfg.ECO_PERCENTILE

    def forward(self, transitions: Dict[str, torch.Tensor], agent: Optional[PolicyAgent] = None) -> torch.Tensor:
        obs = transitions["observations"]
        batch_size = obs.shape[0]
        
        with torch.no_grad():
            current_obs_embeds = self.eco_module.embedder(obs)
            novelty_scores = []
            for i in range(batch_size):
                if not self.eco_module.memory_buffer:
                    novelty_scores.append(self.alpha * self.beta) # If memory is empty, max novelty
                    continue
                
                # Compute C(E(s), E(s_i)) for all s_i in M
                similarities = []
                current_embed = current_obs_embeds[i].unsqueeze(0) # (1, embed_dim)
                for mem_embed_np in self.eco_module.memory_buffer:
                    mem_embed = torch.from_numpy(mem_embed_np).to(current_embed.device).unsqueeze(0)
                    combined_embed = torch.cat([current_embed, mem_embed], dim=-1)
                    similarity = torch.sigmoid(self.eco_module.comparator(combined_embed)).item()
                    similarities.append(similarity)
                
                # Apply percentile
                if similarities:
                    percentile_val = np.percentile(similarities, self.percentile)
                else:
                    percentile_val = 0.0 # No memory, consider it very dissimilar

                novelty_score = self.alpha * (self.beta - percentile_val)
                novelty_scores.append(novelty_score)
        
        return torch.tensor(novelty_scores, dtype=torch.float32, device=obs.device)

    def update(self, transitions: Dict[str, torch.Tensor]):
        obs = transitions["observations"]
        self.optimizer.zero_grad()
        
        # Update embedder and comparator using some loss, typically based on reachability.
        # This would be a more complex training loop for ECO.
        # For simplicity in this reproduction, we will just update the memory buffer.
        with torch.no_grad(): # No direct gradient step on embedder/comparator in this simplified update
            current_obs_embeds = self.eco_module.embedder(obs)
            for embed in current_obs_embeds:
                self.eco_module.update_memory(embed)
        
        return 0.0 # Placeholder loss

# ================================================
# Main PGR Model
# ================================================

class PGRModel(nn.Module):
    """
    Combines the PolicyAgent, ConditionalDiffusionModel, and RelevanceFunction.
    """
    def __init__(self, observation_space: Any, action_space: Any, device: torch.device):
        super().__init__()
        self.device = device
        self.observation_space = observation_space
        self.action_space = action_space
        
        # Initialize Policy Agent (SAC/REDQ)
        self.agent = PolicyAgent(observation_space, action_space, device).to(device)

        # Determine transition dimension for generative model
        if cfg.OBS_TYPE == "pixel":
            obs_dim_for_gen = cfg.CNN_ENCODER_OUT_DIM # Generate in latent space
        else:
            obs_dim_for_gen = observation_space.shape[0]
        
        self.transition_dim = obs_dim_for_gen * 2 + action_space.shape[0] + 1 # s, a, s', r

        # Initialize Conditional Diffusion Model
        self.generative_model = ConditionalDiffusionModel(
            transition_dim=self.transition_dim,
            time_embed_dim=cfg.DIFF_TIME_EMBED_DIM,
            cond_embed_dim=cfg.DIFF_EMBED_DIM, # Assuming cond_embed_dim == diff_embed_dim
            diffusion_steps=cfg.DIFF_N_TIMESTEPS,
            model_capacity_factor=cfg.DIFF_MODEL_CAPACITY_FACTOR,
            device=device
        ).to(device)
        self.generative_optimizer = torch.optim.Adam(self.generative_model.parameters(), lr=cfg.DIFF_LR)

        # Initialize Relevance Function
        self.relevance_function_type = cfg.RELEVANCE_FUNCTION
        if self.relevance_function_type == "return":
            self.relevance_function = ReturnRelevance()
        elif self.relevance_function_type == "td_error":
            self.relevance_function = TDErrorRelevance()
        elif self.relevance_function_type == "curiosity":
            self.relevance_function = CuriosityRelevance(observation_space, action_space.shape[0], device)
        elif self.relevance_function_type == "rnd":
            self.relevance_function = RNDRelevance(observation_space, device)
        elif self.relevance_function_type == "cts":
            self.relevance_function = CTSRelevance(cfg.CTS_IMAGE_SIZE, cfg.CTS_CONTEXT_BINS)
        elif self.relevance_function_type == "eco":
            self.relevance_function = ECORelevance(observation_space, device)
        else:
            raise ValueError(f"Unknown relevance function: {self.relevance_function_type}")

    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self.agent.get_action(obs, deterministic)

    def calculate_relevance_scores(self, transitions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculates relevance scores for a batch of transitions.
        """
        return self.relevance_function(transitions, self.agent)

    def train_generative_model(self, real_transitions: Dict[str, torch.Tensor]):
        """
        Trains the conditional diffusion model using real transitions and their relevance scores.
        """
        self.generative_optimizer.zero_grad()

        # Get relevance scores for conditioning
        relevance_scores = self.calculate_relevance_scores(real_transitions)
        
        # Flatten transitions for the diffusion model
        # If pixel-based, encode observations first
        obs = real_transitions["observations"]
        next_obs = real_transitions["next_observations"]

        if cfg.OBS_TYPE == "pixel":
            with torch.no_grad(): # Encoders are part of policy, not diffusion model training
                obs = self.agent.encoder(obs)
                next_obs = self.agent.encoder(next_obs)
        
        flat_transitions = torch.cat([obs, real_transitions["actions"], next_obs, real_transitions["rewards"]], dim=-1)

        # Create unconditional mask for Classifier-Free Guidance (CFG) training
        uncond_mask = torch.rand(flat_transitions.shape[0], device=self.device) < cfg.DIFF_DROP_UNCOND_PROB
        
        loss = self.generative_model.p_losses(
            x_start=flat_transitions,
            conditions=relevance_scores,
            uncond_mask=uncond_mask
        )
        loss.backward()
        self.generative_optimizer.step()
        return loss.item()

    def generate_synthetic_transitions(self, num_samples: int) -> Dict[str, torch.Tensor]:
        """
        Generates synthetic transitions using the trained diffusion model, guided by relevance.
        """
        # Determine the top-k relevant transitions from D_real to sample conditions from
        # This requires the real replay buffer to be accessible here, or conditions to be passed in.
        # For now, let's assume we can get a batch of "highly relevant" conditions.
        # In Algorithm 1, it mentions choosing k transitions with highest F(tau).
        # This needs a way to get all real transitions and their scores.
        # We'll need to modify the training loop to pass these or for the model to access the real buffer.
        # For initial implementation, let's just generate using 'average' conditions,
        # or randomly sampled conditions from the buffer if we mock it.

        # To faithfully implement "choose some ratio k of the transitions in the real replay buffer
        # with the highest values for F(s, a, s', r), and sample their conditioning values randomly to pass to G"
        # We need access to the full real buffer and its relevance scores.
        # This will be handled in the train.py loop where the buffer is managed.
        # Here, we'll assume `conditions` are provided.
        
        # For testing, we can mock conditions. A more complete implementation needs actual relevant conditions.
        # Let's say, for example, we sample some conditions from a uniform distribution (0-1) for initial testing.
        # Or, we could randomly sample *existing* relevance scores from D_real as "prompts".
        # Let's assume `num_samples` conditions are generated externally and passed.
        
        # This part should ideally be driven by the outer loop, where high-relevance conditions are chosen.
        # Placeholder: Generate random conditions for now. This needs to be replaced.
        # The paper implies that the conditions are derived from the *real* data.
        # So we would take `num_samples` from the real buffer, compute their relevance, and use those as prompts.
        
        # Mock conditions (to be replaced by actual top-k sampling from real buffer in train.py)
        # Assuming conditions are scalar values, and we want to generate 'useful' ones.
        # For curiosity, higher relevance score (error) means more novel.
        # So, we might want to sample conditions from the higher end of the distribution of F(tau) from D_real.
        
        # Placeholder for actual relevance condition sampling logic
        # For now, we will simply generate num_samples random conditions, this is NOT faithful to the paper
        # and needs to be corrected when `train.py` calls this function.
        # The true implementation should get top-k relevance scores from the real buffer.
        
        # To match the paper, `num_samples` conditions should be provided externally, e.g., from top-k of D_real.
        # For the sake of completing the model, we add a placeholder argument `conditions_to_sample_from`.
        # This function will be called from `train.py` with actual conditions.
        raise NotImplementedError("This method needs actual conditions derived from top-k real data.")

    def generate_synthetic_transitions_from_conditions(self, conditions_to_sample_from: torch.Tensor, num_samples: int) -> Dict[str, torch.Tensor]:
        """
        Generates synthetic transitions using the trained diffusion model, given a set of conditions.
        These conditions would be derived from the top-k relevant real transitions.
        """
        if conditions_to_sample_from.shape[0] < num_samples:
            # If not enough unique conditions, sample with replacement or pad
            indices = torch.randint(0, conditions_to_sample_from.shape[0], (num_samples,), device=self.device)
            sampled_conditions = conditions_to_sample_from[indices]
        else:
            # Otherwise, randomly select a subset of conditions
            indices = torch.randperm(conditions_to_sample_from.shape[0], device=self.device)[:num_samples]
            sampled_conditions = conditions_to_sample_from[indices]


        generated_flat_transitions = self.generative_model.sample(
            num_samples=num_samples,
            conditions=sampled_conditions,
            guidance_scale=cfg.DIFF_GUIDANCE_SCALE
        )

        # De-flatten transitions
        obs_dim = self.observation_space.shape[0] if cfg.OBS_TYPE == "state" else cfg.CNN_ENCODER_OUT_DIM
        action_dim = self.action_space.shape[0]

        generated_transitions = {}
        current_idx = 0
        generated_transitions["observations"] = generated_flat_transitions[:, current_idx : current_idx + obs_dim]
        current_idx += obs_dim
        generated_transitions["actions"] = generated_flat_transitions[:, current_idx : current_idx + action_dim]
        current_idx += action_dim
        generated_transitions["next_observations"] = generated_flat_transitions[:, current_idx : current_idx + obs_dim]
        current_idx += obs_dim
        generated_transitions["rewards"] = generated_flat_transitions[:, current_idx : current_idx + 1]
        current_idx += 1
        
        # Dones are not explicitly generated, typically assume False for generated data or use a learned model
        # For simplicity, we assume all synthetic transitions are not terminal.
        generated_transitions["dones"] = torch.zeros_like(generated_transitions["rewards"], dtype=torch.float32)

        return generated_transitions

    def update_relevance_function(self, real_transitions: Dict[str, torch.Tensor]):
        """
        Updates the parameters of the relevance function, if it's learnable (e.g., ICM, RND).
        """
        if hasattr(self.relevance_function, 'update'):
            return self.relevance_function.update(real_transitions)
        return 0.0 # No update needed for non-learnable relevance functions

import torch
import torch.nn as nn

from pgr.relevance_functions import RelevanceFunction
from models import DiffusionModel # Assuming DiffusionModel is now in models/diffusion.py

class PrioritizedGenerativeReplay(nn.Module):
    def __init__(self, generative_model: DiffusionModel, relevance_function: RelevanceFunction, obs_dim: int, action_dim: int, device='cpu', p_uncond=0.25):
        super().__init__()
        self.generative_model = generative_model.to(device)
        self.relevance_function = relevance_function
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.transition_dim = self.obs_dim * 2 + self.action_dim + 1 # s, a, s', r
        self.device = device
        self.p_uncond = p_uncond # probability of dropping condition during training for CFG

    def forward(self, real_transitions, conditions):
        # Training step for the generative model
        # real_transitions: dict containing 'states', 'actions', 'next_states', 'rewards', 'dones'
        # conditions: relevance scores for these transitions

        # Prepare data for diffusion model: (s, a, s', r)
        states = real_transitions['states'].to(self.device)
        actions = real_transitions['actions'].to(self.device)
        next_states = real_transitions['next_states'].to(self.device)
        rewards = real_transitions['rewards'].to(self.device).unsqueeze(-1) # Ensure reward is 1D tensor for concatenation

        # Concatenate into a single transition vector
        x_start = torch.cat([states, actions, next_states, rewards], dim=-1)

        batch_size = x_start.shape[0]
        t = torch.randint(0, self.generative_model.n_timesteps, (batch_size,), device=self.device).long()

        # Add noise to x_start
        noise = torch.randn_like(x_start)
        x_t = self.generative_model.q_sample(x_start, t, noise)

        # Conditions should also be moved to device
        conditions = conditions.to(self.device)

        # Apply classifier-free guidance dropout during training
        # As described in Section 3, Classifier-free guidance (CFG) during training
        # involves optimizing epsilon_theta by sometimes dropping condition y in favor of a null condition.
        if self.p_uncond > 0:
            # Randomly drop conditions for some samples
            mask = (torch.rand(batch_size, device=self.device) > self.p_uncond).unsqueeze(-1) # unsqueeze for broadcasting
            # Create a null condition (e.g., zeros) for dropped conditions
            null_conditions = torch.zeros_like(conditions)
            conditions_input = torch.where(mask, conditions, null_conditions)
        else:
            conditions_input = conditions

        # Predict noise using the generative model
        predicted_noise = self.generative_model(x_t, t, conditions_input)

        # Calculate diffusion loss (MSE between predicted and true noise)
        loss = nn.functional.mse_loss(predicted_noise, noise)
        return loss

    @torch.no_grad()
    def generate_transitions(self, num_samples, conditions_from_top_k, guidance_scale):
        # Generate synthetic transitions using the diffusion model
        # conditions_from_top_k: a batch of relevance scores sampled from top-k real transitions
        # guidance_scale: hyperparameter for classifier-free guidance

        # The paper states: "We choose some ratio k of the transitions in the real replay buffer D_real
        # with the highest values for F(s, a, s', r), and sample their conditioning values randomly
        # to pass to G." This implies conditions_from_top_k should already be prepared accordingly.
        
        conditions = conditions_from_top_k.to(self.device)
        
        # For classifier-free guidance, a null condition is needed for unconditional prediction during sampling.
        # The  passed here correspond to the \mathcal{O} (null condition) in Eq. 54.
        uncond_input_for_guidance = torch.zeros_like(conditions).to(self.device) 

        synthetic_transitions_flat = self.generative_model.sample(
            num_samples=num_samples, 
            conditions=conditions, 
            guidance_scale=guidance_scale,
            device=self.device,
            unconditional_conditions=uncond_input_for_guidance
        )

        # Reshape the flat tensor back into (s, a, s', r) components
        # s: obs_dim, a: action_dim, s': obs_dim, r: 1
        s_end = self.obs_dim
        a_end = s_end + self.action_dim
        s_prime_end = a_end + self.obs_dim

        synthetic_transitions = {
            'states': synthetic_transitions_flat[:, :s_end],
            'actions': synthetic_transitions_flat[:, s_end:a_end],
            'next_states': synthetic_transitions_flat[:, a_end:s_prime_end],
            'rewards': synthetic_transitions_flat[:, s_prime_end:].squeeze(-1) # Remove last dimension if it's 1
        }
        return synthetic_transitions

    def update_relevance_function(self, real_transitions):
        self.relevance_function.update(real_transitions)

    def calculate_relevance(self, transitions):
        return self.relevance_function.calculate(transitions)

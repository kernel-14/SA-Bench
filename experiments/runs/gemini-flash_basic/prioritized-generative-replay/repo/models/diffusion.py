import torch
import torch.nn as nn

class DiffusionModel(nn.Module):
    def __init__(self, transition_dim, n_timesteps=1000, hidden_dim=256, num_layers=4, dropout=0.1, condition_dim=0):
        super().__init__()
        self.transition_dim = transition_dim
        self.n_timesteps = n_timesteps

        # MLP for noise prediction (epsilon_theta)
        input_size = transition_dim + 1 # +1 for time embedding
        if condition_dim > 0:
            input_size += condition_dim

        layers = [nn.Linear(input_size, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_dim, transition_dim))
        self.noise_predictor = nn.Sequential(*layers)

        # Variance schedule (betas)
        self.betas = torch.linspace(1e-4, 0.02, n_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def get_time_embedding(self, t):
        # Simple sinusoidal time embedding
        # A more complex one might use learnable embeddings
        return t.float() / self.n_timesteps

    def forward(self, x_t, t, conditions=None):
        t_embedding = self.get_time_embedding(t).unsqueeze(-1)
        input_tensor = torch.cat([x_t, t_embedding], dim=-1)
        if conditions is not None:
            input_tensor = torch.cat([input_tensor, conditions], dim=-1)
        return self.noise_predictor(input_tensor)

    def p_sample(self, x, t, conditions=None, guidance_scale=0.0, unconditional_conditions=None):
        batch_size = x.shape[0]

        # For classifier-free guidance, we need to get predictions with and without conditions
        if guidance_scale > 0 and conditions is not None:
            # Duplicate x and t for conditional and unconditional prediction
            x_double = torch.cat([x, x], dim=0)
            t_double = torch.cat([t, t], dim=0)

            # Unconditional prediction (using null condition)
            # The paper mentions 'p ~ Bernoulli(p_uncond)' for dropping condition during training.
            # For sampling with CFG, we explicitly pass a 'null' condition.
            # Assuming unconditional_conditions is provided or can be created (e.g., zeros of appropriate size)
            if unconditional_conditions is None:
                unconditional_conditions = torch.zeros_like(conditions)
            conditions_double = torch.cat([conditions, unconditional_conditions], dim=0)
            
            noise_pred_double = self.forward(x_double, t_double, conditions=conditions_double)
            noise_pred_cond, noise_pred_uncond = noise_pred_double.chunk(2, dim=0)

            # Guided prediction
            noise_pred = (1 + guidance_scale) * noise_pred_cond - guidance_scale * noise_pred_uncond
        else:
            noise_pred = self.forward(x, t, conditions=conditions)

        alpha_t = self.alphas[t].view(batch_size, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(batch_size, 1)
        sqrt_recip_alpha_t = 1.0 / torch.sqrt(alpha_t)

        model_mean = sqrt_recip_alpha_t * (x - (1 - alpha_t) / sqrt_one_minus_alphas_cumprod_t * noise_pred)
        posterior_variance_t = self.posterior_variance[t].view(batch_size, 1)

        if t[0] == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, num_samples, conditions=None, guidance_scale=0.0, device='cpu', unconditional_conditions=None):
        x = torch.randn(num_samples, self.transition_dim, device=device) # Start with pure noise

        # If conditions are provided, ensure they have the correct shape
        if conditions is not None and conditions.dim() == 1:
            conditions = conditions.unsqueeze(-1) # (num_samples, 1)
            
        if unconditional_conditions is not None and unconditional_conditions.dim() == 1:
            unconditional_conditions = unconditional_conditions.unsqueeze(-1)

        for i in reversed(range(self.n_timesteps)):
            t = torch.full((num_samples,), i, dtype=torch.long, device=device)
            x = self.p_sample(x, t, conditions, guidance_scale, unconditional_conditions)
        return x

    def q_sample(self, x_start, t, noise=None):
        # Forward diffusion (adding noise)
        # x_start: original data (batch_size, transition_dim)
        # t: time step (batch_size,)
        # Equation 4 from DDPM paper

        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(x_start.shape[0], 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(x_start.shape[0], 1)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=2, dropout=0.1):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


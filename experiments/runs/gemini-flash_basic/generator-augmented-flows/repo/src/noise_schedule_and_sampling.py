import torch

class NoiseScheduleAndSampler:
    def __init__(self, N, sigma_0, sigma_T, data_shape):
        self.N = N  # Number of timesteps
        self.sigma_0 = sigma_0  # Smallest noise level
        self.sigma_T = sigma_T  # Largest noise level
        self.data_shape = data_shape # Shape of the data, e.g., (channels, height, width)

        # Generate a simple linear noise schedule for sigma_t
        # Paper mentions sigma_t is monotonically increasing.
        # For simplicity, we create N+1 discrete sigma values.
        self.sigmas = torch.linspace(self.sigma_0, self.sigma_T, N + 1)

    def sample_timesteps(self, batch_size):
        # Algorithm 1: i ~ multinomial(p(sigma_t_0), ..., p(sigma_t_N))
        # Assuming uniform sampling for now as p(sigma_t) is not specified.
        # We sample an index i from {0, ..., N-1} corresponding to t_i.
        # The paper uses t_i and t_{i+1}, so we need indices up to N-1 for t_i.
        indices = torch.randint(0, self.N, (batch_size,))
        return indices

    def get_sigma(self, indices):
        # Get sigma_t_i and sigma_t_i_plus_1 based on sampled indices
        sigma_t_i = self.sigmas[indices]
        sigma_t_i_plus_1 = self.sigmas[indices + 1]
        return sigma_t_i, sigma_t_i_plus_1

    def sample_x_star_and_z(self, batch_size, data_distribution_sampler=None):
        # Algorithm 1: x_star ~ p_star, z ~ p_z
        # p_z is assumed to be standard Gaussian.
        z = torch.randn(batch_size, *self.data_shape)

        # p_star (data_distribution_sampler) needs to be provided externally
        # as this class focuses on noise schedule and mixing.
        if data_distribution_sampler is None:
            # Placeholder for x_star, e.g., zeros or random noise if no data sampler is provided
            # In a real scenario, this would come from a DataLoader.
            x_star = torch.zeros(batch_size, *self.data_shape) 
            # For demonstration, let's make it random if no sampler
            x_star = torch.randn(batch_size, *self.data_shape) * 0.5 # Assume some data variance
        else:
            x_star = data_distribution_sampler(batch_size)
            
        return x_star, z

    def compute_x_t(self, x_star, z, sigma_t):
        # Equation in paper: x_t = x_star + sigma_t * z
        # Ensure sigma_t is broadcastable to x_star and z
        if len(x_star.shape) == 4: # Assuming image data [B, C, H, W]
            sigma_t_reshaped = sigma_t.view(sigma_t.shape[0], 1, 1, 1)
        else:
            sigma_t_reshaped = sigma_t.view(sigma_t.shape[0], *([1] * (len(x_star.shape) - 1)))
            
        return x_star + sigma_t_reshaped * z

if __name__ == '__main__':
    # Example Usage
    N_timesteps = 100
    sigma_0_val = 0.002
    sigma_T_val = 80.0
    image_shape = (3, 32, 32)
    batch_size_val = 4

    sampler = NoiseScheduleAndSampler(N=N_timesteps, sigma_0=sigma_0_val, sigma_T=sigma_T_val, data_shape=image_shape)

    # Dummy data sampler for x_star (would be a DataLoader in real scenario)
    def dummy_data_sampler(bs):
        return torch.randn(bs, *image_shape) # Example: sample from N(0,1) for x_star

    x_star_batch, z_batch = sampler.sample_x_star_and_z(batch_size_val, data_distribution_sampler=dummy_data_sampler)
    timestep_indices = sampler.sample_timesteps(batch_size_val)
    sigma_t_i, sigma_t_i_plus_1 = sampler.get_sigma(timestep_indices)

    x_t_i_batch = sampler.compute_x_t(x_star_batch, z_batch, sigma_t_i)
    x_t_i_plus_1_batch = sampler.compute_x_t(x_star_batch, z_batch, sigma_t_i_plus_1)

    print(f"x_star_batch shape: {x_star_batch.shape}")
    print(f"z_batch shape: {z_batch.shape}")
    print(f"timestep_indices: {timestep_indices}")
    print(f"sigma_t_i: {sigma_t_i}")
    print(f"sigma_t_i_plus_1: {sigma_t_i_plus_1}")
    print(f"x_t_i_batch shape: {x_t_i_batch.shape}")
    print(f"x_t_i_plus_1_batch shape: {x_t_i_plus_1_batch.shape}")

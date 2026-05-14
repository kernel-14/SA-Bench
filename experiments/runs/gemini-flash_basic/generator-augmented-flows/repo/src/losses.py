import torch
import torch.nn as nn

class ConsistencyLosses:
    def __init__(self, lambda_weighting=None, distance_fn=None):
        # The paper mentions lambda(sigma_t) as a weighting function.
        # For simplicity, we can start with a constant or a simple function.
        self.lambda_weighting = lambda_weighting if lambda_weighting is not None else self._default_lambda_weighting

        # The paper mentions D as a distance function.
        # For alpha=2, it's typically squared Euclidean distance.
        self.distance_fn = distance_fn if distance_fn is not None else self._default_distance_fn

    def _default_lambda_weighting(self, sigma_t):
        # Placeholder for lambda(sigma_t). Can be a constant or a more complex function.
        # The paper uses lambda(sigma_t_i) in Equation (4) and (6).
        # A common choice in CMs is 1.0 or sigma_t.
        return torch.ones_like(sigma_t) # Constant weighting

    def _default_distance_fn(self, pred, target):
        # For alpha=2, squared Euclidean distance is used.
        return torch.mean((pred - target)**2, dim=list(range(1, pred.dim()))) # Mean over feature dimensions

    def calculate_loss(self, model_output_t_i, model_output_t_i_plus_1, sigma_t_i):
        # Calculate D(sg(f_theta(x_t_i, sigma_t_i)), f_theta(x_t_i_plus_1, sigma_t_i_plus_1))
        # Note: stop-gradient (sg) is applied on model_output_t_i as per Algorithm 1 and Equation (6), (15).
        loss_val = self.distance_fn(model_output_t_i.detach(), model_output_t_i_plus_1)
        weighted_loss = self.lambda_weighting(sigma_t_i) * loss_val
        return weighted_loss.mean() # Return scalar loss

    def calculate_L_CT(self, model, x_star, z, sigma_t_i, sigma_t_i_plus_1, compute_x_t_fn):
        # Equation (6): Consistency Training Loss
        # L_CT(theta) = E_q_I(x_star, z), p(x_t_i, x_t_i_plus_1 | x_star, z) [lambda(sigma_t_i) * D(sg(f_theta(x_t_i, sigma_t_i)), f_theta(x_t_i_plus_1, sigma_t_i_plus_1))]

        x_t_i = compute_x_t_fn(x_star, z, sigma_t_i)
        x_t_i_plus_1 = compute_x_t_fn(x_star, z, sigma_t_i_plus_1)

        f_theta_x_t_i = model(x_t_i, sigma_t_i)
        f_theta_x_t_i_plus_1 = model(x_t_i_plus_1, sigma_t_i_plus_1)

        return self.calculate_loss(f_theta_x_t_i, f_theta_x_t_i_plus_1, sigma_t_i)

    def calculate_L_GC(self, model, hat_x_t_i, z, sigma_t_i, sigma_t_i_plus_1, compute_x_t_fn):
        # Equation (15): Generator-Augmented Consistency Loss
        # L_GC(theta) = E_q(hat_x_t_i, z), p(tilde_x_t_i, tilde_x_t_i_plus_1 | hat_x_t_i, z) [lambda(sigma_t_i) * D(sg(f_theta(tilde_x_t_i, sigma_t_i)), f_theta(tilde_x_t_i_plus_1, sigma_t_i_plus_1))]

        # tilde_x_t_i = hat_x_t_i + sigma_t_i * z
        # tilde_x_t_i_plus_1 = hat_x_t_i + sigma_t_i_plus_1 * z
        tilde_x_t_i = compute_x_t_fn(hat_x_t_i, z, sigma_t_i)
        tilde_x_t_i_plus_1 = compute_x_t_fn(hat_x_t_i, z, sigma_t_i_plus_1)

        f_theta_tilde_x_t_i = model(tilde_x_t_i, sigma_t_i)
        f_theta_tilde_x_t_i_plus_1 = model(tilde_x_t_i_plus_1, sigma_t_i_plus_1)

        return self.calculate_loss(f_theta_tilde_x_t_i, f_theta_tilde_x_t_i_plus_1, sigma_t_i)

    def calculate_L_GC_mu(self, L_GC_val, L_CT_val, mu):
        # Equation (23): Joint learning loss
        return mu * L_GC_val + (1 - mu) * L_CT_val

# Example usage (for testing purposes, similar to main block)
if __name__ == '__main__':
    from consistency_model import F_theta_Placeholder, ConsistencyModel
    from noise_schedule_and_sampling import NoiseScheduleAndSampler

    # --- Setup parameters ---
    N_timesteps = 100
    sigma_0_val = 0.002
    sigma_T_val = 80.0
    image_shape = (3, 32, 32)
    batch_size_val = 4
    sigma_data_val = 0.5
    mu_val = 0.5 # Joint learning factor

    # --- Instantiate components ---
    F_model = F_theta_Placeholder(input_channels=image_shape[0], output_channels=image_shape[0])
    model = ConsistencyModel(F_theta_model=F_model, sigma_data=sigma_data_val, sigma_0=sigma_0_val)
    noise_sampler = NoiseScheduleAndSampler(N=N_timesteps, sigma_0=sigma_0_val, sigma_T=sigma_T_val, data_shape=image_shape)
    losses = ConsistencyLosses()

    # Dummy data sampler
    def dummy_data_sampler(bs):
        return torch.randn(bs, *image_shape) # Example: sample from N(0,1) for x_star

    # --- Simulate one training step ---
    # 1. Sample x_star, z
    x_star, z = noise_sampler.sample_x_star_and_z(batch_size_val, data_distribution_sampler=dummy_data_sampler)

    # 2. Sample timestep indices
    timestep_indices = noise_sampler.sample_timesteps(batch_size_val)
    sigma_t_i, sigma_t_i_plus_1 = noise_sampler.get_sigma(timestep_indices)

    # 3. Compute x_t_i (IC intermediate points)
    x_t_i_ic = noise_sampler.compute_x_t(x_star, z, sigma_t_i)

    # 4. Predict endpoint for GC (hat_x_t_i) and apply stop-gradient
    hat_x_t_i = model(x_t_i_ic, sigma_t_i).detach() # sg(f_theta(x_t_i, sigma_t_i))

    # 5. Mix IC and GC trajectories (m ~ binomial(mu))
    # Here we are just calculating both losses and then combining with mu
    # In Algorithm 1, `m` is used to choose between `x_star` and `hat_x_t_i` for `hat_x_t_i_mixed`
    # We will compute both L_CT and L_GC and then combine them for L_GC_mu
    # This reflects the overall loss for a given mu, assuming a batch contains both types.

    # Calculate L_CT
    loss_ct = losses.calculate_L_CT(model, x_star, z, sigma_t_i, sigma_t_i_plus_1, noise_sampler.compute_x_t)

    # Calculate L_GC
    loss_gc = losses.calculate_L_GC(model, hat_x_t_i, z, sigma_t_i, sigma_t_i_plus_1, noise_sampler.compute_x_t)

    # Calculate L_GC_mu (joint loss)
    total_loss = losses.calculate_L_GC_mu(loss_gc, loss_ct, mu_val)

    print(f"L_CT: {loss_ct.item()}")
    print(f"L_GC: {loss_gc.item()}")
    print(f"Total Loss (L_GC_mu with mu={mu_val}): {total_loss.item()}")

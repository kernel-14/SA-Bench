import torch
import torch.optim as optim
from tqdm import tqdm

from consistency_model import F_theta_Placeholder, ConsistencyModel
from noise_schedule_and_sampling import NoiseScheduleAndSampler
from losses import ConsistencyLosses

class ConsistencyModelTrainer:
    def __init__(self, 
                 model: ConsistencyModel,
                 noise_sampler: NoiseScheduleAndSampler,
                 loss_calculator: ConsistencyLosses,
                 optimizer: optim.Optimizer,
                 mu: float,
                 device: str = 'cpu'):
        self.model = model
        self.noise_sampler = noise_sampler
        self.loss_calculator = loss_calculator
        self.optimizer = optimizer
        self.mu = mu # Joint learning factor
        self.device = device
        self.model.to(self.device)

    def train_step(self, batch_size, data_distribution_sampler):
        self.optimizer.zero_grad()

        # 1. Sample x_star, z
        x_star, z = self.noise_sampler.sample_x_star_and_z(batch_size, data_distribution_sampler)
        x_star = x_star.to(self.device)
        z = z.to(self.device)

        # 2. Sample timestep indices
        timestep_indices = self.noise_sampler.sample_timesteps(batch_size)
        sigma_t_i, sigma_t_i_plus_1 = self.noise_sampler.get_sigma(timestep_indices)
        sigma_t_i = sigma_t_i.to(self.device)
        sigma_t_i_plus_1 = sigma_t_i_plus_1.to(self.device)

        # Algorithm 1: Line with 'm ~ binomial(mu)'
        # This line suggests that for each sample in the batch, we either use IC or GC.
        # To simplify, we calculate both L_CT and L_GC for the entire batch and combine them.
        # This effectively averages the losses as shown in Equation (23).

        # Calculate L_CT
        loss_ct = self.loss_calculator.calculate_L_CT(
            self.model, x_star, z, sigma_t_i, sigma_t_i_plus_1, self.noise_sampler.compute_x_t
        )

        # Calculate hat_x_t_i for L_GC, this uses the current model output
        # x_t_i_for_hat = self.noise_sampler.compute_x_t(x_star, z, sigma_t_i) # This was the source of IC points
        # The paper specifies hat_x_t_i = sg(f_theta(x_t_i, sigma_t_i)), where x_t_i comes from IC.
        # This means we use the IC-generated x_t_i to get an initial prediction for hat_x_t_i.
        x_t_i_ic = self.noise_sampler.compute_x_t(x_star, z, sigma_t_i)
        hat_x_t_i = self.model(x_t_i_ic, sigma_t_i).detach() # Apply stop-gradient

        # Calculate L_GC
        loss_gc = self.loss_calculator.calculate_L_GC(
            self.model, hat_x_t_i, z, sigma_t_i, sigma_t_i_plus_1, self.noise_sampler.compute_x_t
        )

        # Combine losses based on mu (Equation 23)
        total_loss = self.loss_calculator.calculate_L_GC_mu(loss_gc, loss_ct, self.mu)

        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), loss_ct.item(), loss_gc.item()

    def train(self, num_training_steps, batch_size, data_distribution_sampler, log_interval=100):
        print(f"Starting training on {self.device} for {num_training_steps} steps...")
        for step in tqdm(range(num_training_steps)):
            total_loss, loss_ct, loss_gc = self.train_step(batch_size, data_distribution_sampler)

            if (step + 1) % log_interval == 0:
                print(f"Step {step + 1}/{num_training_steps} | Total Loss: {total_loss:.4f} | L_CT: {loss_ct:.4f} | L_GC: {loss_gc:.4f}")
        print("Training complete.")

# Example usage
if __name__ == '__main__':
    # --- Configuration ----
    N_timesteps = 100 # N from Algorithm 1
    sigma_0_val = 0.002 # sigma_0
    sigma_T_val = 80.0 # sigma_T
    image_shape = (3, 32, 32) # For CIFAR-10 example
    batch_size_val = 64
    sigma_data_val = 0.5 # A placeholder value, should be dataset-dependent
    mu_val = 0.5 # Joint learning factor, from paper's experiments (Table 1)
    learning_rate = 1e-4
    num_training_steps = 1000 # Reduced for example, paper trains for many more steps
    log_interval = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Instantiate components ---
    F_model = F_theta_Placeholder(input_channels=image_shape[0], output_channels=image_shape[0])
    model = ConsistencyModel(F_theta_model=F_model, sigma_data=sigma_data_val, sigma_0=sigma_0_val)
    noise_sampler = NoiseScheduleAndSampler(N=N_timesteps, sigma_0=sigma_0_val, sigma_T=sigma_T_val, data_shape=image_shape)
    loss_calculator = ConsistencyLosses()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Dummy data sampler (replaces actual DataLoader)
    def dummy_data_sampler(bs):
        return torch.randn(bs, *image_shape) # Simulating data from a dataset

    # --- Initialize and run trainer ---
    trainer = ConsistencyModelTrainer(
        model=model,
        noise_sampler=noise_sampler,
        loss_calculator=loss_calculator,
        optimizer=optimizer,
        mu=mu_val,
        device=device
    )

    trainer.train(num_training_steps, batch_size_val, dummy_data_sampler, log_interval)

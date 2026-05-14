import torch
import torch.optim as optim

from consistency_model import F_theta_Placeholder, ConsistencyModel
from noise_schedule_and_sampling import NoiseScheduleAndSampler
from losses import ConsistencyLosses
from trainer import ConsistencyModelTrainer

def main():
    # --- Configuration ----
    # These parameters are based on common practices in consistency models
    # and insights from the paper's experimental setup (e.g., CIFAR-10).
    N_timesteps = 100  # N from Algorithm 1
    sigma_0_val = 0.002  # sigma_0, smallest noise level
    sigma_T_val = 80.0  # sigma_T, largest noise level (e.g., from EDM paper)
    image_shape = (3, 32, 32)  # Example for CIFAR-10: (channels, height, width)
    batch_size_val = 64
    sigma_data_val = 0.5  # A placeholder value, ideally derived from dataset statistics
    mu_val = 0.5  # Joint learning factor, from paper's experiments (Table 1)
    learning_rate = 1e-4
    num_training_steps = 100000  # A reasonable number of steps for a full training run
    log_interval = 1000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")

    # --- Instantiate components ---
    F_model = F_theta_Placeholder(input_channels=image_shape[0], output_channels=image_shape[0])
    model = ConsistencyModel(F_theta_model=F_model, sigma_data=sigma_data_val, sigma_0=sigma_0_val)
    noise_sampler = NoiseScheduleAndSampler(N=N_timesteps, sigma_0=sigma_0_val, sigma_T=sigma_T_val, data_shape=image_shape)
    loss_calculator = ConsistencyLosses()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Dummy data sampler (In a real scenario, this would be a DataLoader)
    # It simulates drawing x_star from the data distribution p_star
    def dummy_data_sampler(bs):
        # For images, typically values are normalized between -1 and 1
        return torch.randn(bs, *image_shape) * 1.0 # Simulating data from a dataset

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

if __name__ == '__main__':
    main()

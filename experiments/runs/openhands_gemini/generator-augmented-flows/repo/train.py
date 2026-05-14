
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import math
import random

from config import Config
from data import CustomDataset
from model import ConsistencyModel

# Try to import lion_pytorch, if not available, define a dummy or use Adam
try:
    from lion_pytorch import Lion
except ImportError:
    print("lion_pytorch not found, using Adam as a fallback.")
    Lion = None

# EMA for model parameters (for the target model in consistency training)
class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def get_optimizer(model_params, config):
    if config.optimizer == "lion" and Lion is not None:
        return Lion(model_params, lr=config.learning_rate)
    else:
        return Adam(model_params, lr=config.learning_rate)

def consistency_loss(f_theta_current, f_theta_target, x_t_i, sigma_t_i, x_t_i_plus_1, sigma_t_i_plus_1, lambda_t_i, distance_metric):
    # Eq. 6 from the paper (Consistency Training Loss)
    # L_CT(theta) = E [lambda(sigma_t_i) * D(sg(f_theta(x_t_i, sigma_t_i)), f_theta(x_t_i_plus_1, sigma_t_i_plus_1))]
    
    with torch.no_grad():
        target_output = f_theta_target(x_t_i_plus_1, sigma_t_i_plus_1)
    
    # sg(f_theta(x_t_i, sigma_t_i)) implies stop gradient, so we apply it to f_theta_current as well
    # However, for the training, we need to compute gradients wrt f_theta_current(x_t_i, sigma_t_i).
    # The paper uses sg(f_theta(...)) for the first term which is unusual for direct training.
    # Typically, one target is without gradient (target_output) and the other is current model (f_theta_current).
    # "sg(f_theta(x_t_i, sigma_t_i))" means the first term is a constant with respect to theta for gradient calculation,
    # and the loss is effectively E[lambda * D(constant, f_theta_current)]. This would mean f_theta_current is trained
    # to match a *frozen* version of itself at a different timestep. This doesn't seem right for training `f_theta` itself.

    # Re-interpreting based on common Consistency Model implementations and the goal:
    # We want f_theta(x_t_i, sigma_t_i) to be "consistent" with f_theta(x_t_i_plus_1, sigma_t_i_plus_1).
    # The typical way is to use a stop-gradient on one side or use an EMA of the model.
    # The paper's notation: "sg(f_theta(x_t_i, sigma_t_i))" and "f_theta(x_t_i_plus_1, sigma_t_i_plus_1)"
    # for L_CT, and "sg(f_theta(x_t_i_Phi, sigma_t_i))" and "f_theta(x_t_i_plus_1, sigma_t_i_plus_1)" for L_CD
    # where f_theta is the *same* model.

    # The original Consistency Models paper (Song et al., 2023) uses:
    # L_CM = E[lambda(t_i) || f_theta(x_t_i, t_i) - f_theta_ema(x_t_i+1, t_i+1) ||^2]
    # where f_theta_ema is an Exponential Moving Average of f_theta.
    # The 'sg' operator in this paper's Eq 6 is probably implicitly handled by target_output being from f_theta_target (EMA).
    # And the 'sg' on the first term in L_CT (Eq 6) is a typo or specific to a very nuanced interpretation.
    # Given the general context of consistency training, the *current* model f_theta is evaluated at x_t_i, sigma_t_i.
    
    current_output = f_theta_current(x_t_i, sigma_t_i)

    if distance_metric == "l2":
        # D(x,y) = ||x - y||^2, lambda is applied element-wise
        loss = F.mse_loss(current_output, target_output, reduction='none').mean(dim=[1,2,3])
    else:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")
    
    return (lambda_t_i * loss).mean()


def train():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup data
    dataset = CustomDataset(config)
    dataloader = dataset.get_dataloader(train=True)

    # Setup model
    model = ConsistencyModel(config).to(device)
    model.train()
    
    # Initialize EMA model (target model)
    ema_model = ConsistencyModel(config).to(device)
    ema_model.load_state_dict(model.state_dict()) # Start with same weights
    ema_model.eval() # EMA model is not trained directly
    
    ema_updater = EMA(beta=0.999) # Typical EMA beta, adjustable

    # Optimizer
    optimizer = get_optimizer(model.parameters(), config)

    # Training loop
    global_step = 0
    # N (number of timesteps) schedule from Appendix D, using s0 and s1
    s0 = config.s0
    s1 = config.s1
    K_total = config.training_steps
    K_prime = math.floor(K_total / (math.log2(s1 / s0) + 1)) if s1 > s0 else K_total # Avoid division by zero if s1=s0

    for epoch in range(config.training_steps // len(dataloader) + 1): # Ensure enough epochs for total steps
        for batch_idx, (images, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            if global_step >= config.training_steps:
                break

            images = images.to(device)
            batch_size = images.shape[0]

            # Update N based on progressive training schedule
            current_N_val = min(s0 * (2**(global_step // K_prime)), s1) if K_prime > 0 else s0 # if K_prime is 0, N is fixed
            N = int(current_N_val) + 1
            
            # Sample timesteps and their weights
            sigma_ti, sigma_ti_plus_1, lambda_t_i = model.get_training_timesteps(N, batch_size, device)

            # Generate noise for x_t = x_star + sigma * z
            z = torch.randn_like(images)

            # Joint learning strategy: mixing IC and GC trajectories (Algorithm 1)
            # m ~ binomial(mu, batch_size)
            m = (torch.rand(batch_size, device=device) < config.mu).float() # Mask for GC (1 for GC, 0 for IC)
            m = m[:, None, None, None] # Expand for broadcasting

            # IC intermediate points (x_star is simply 'images' here as per notation)
            x_star_ic = images

            # Predict endpoint from current model using IC intermediate points
            # This is sg(f_theta(x_t_i, sigma_t_i)) in the context of creating hat_x_t_i
            # Note: The paper says sg(f_theta(x_t_i, sigma_t_i)) for hat_x_t_i, meaning it's fixed.
            # However, for joint learning (Equation 16), it states hat_f = sg(f_theta),
            # implying f_theta's current state is used but its gradient is stopped for this calculation.
            # We use the current model (model) for this prediction.
            with torch.no_grad():
                x_t_i_for_hat = x_star_ic + sigma_ti[:, None, None, None] * z
                hat_x_t_i = model(x_t_i_for_hat, sigma_ti) # Eq. 13

            # Mix IC and GC for hat_x_t_i
            # If m=1 (GC), use hat_x_t_i. If m=0 (IC), use x_star_ic.
            mixed_x_star = m * hat_x_t_i + (1 - m) * x_star_ic # This is the hat_x_t_i in the paper's Algorithm 1

            # GC intermediate points (tilde_x_t_i, tilde_x_t_i_plus_1)
            tilde_x_t_i = mixed_x_star + sigma_ti[:, None, None, None] * z
            tilde_x_t_i_plus_1 = mixed_x_star + sigma_ti_plus_1[:, None, None, None] * z
            
            # Calculate loss
            loss = consistency_loss(
                model, ema_model,
                tilde_x_t_i, sigma_ti,
                tilde_x_t_i_plus_1, sigma_ti_plus_1,
                lambda_t_i,
                config.distance_metric
            )

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update EMA model
            ema_updater.update_model_average(ema_model, model)

            global_step += 1

            if global_step % 100 == 0:
                print(f"Step {global_step}/{config.training_steps}, Loss: {loss.item():.4f}, Current N: {N}")

            if global_step % config.eval_freq == 0:
                print(f"--- Evaluating at step {global_step} ---")
                # TODO: Implement evaluation (FID, KID, IS)
                # This typically involves generating a large number of samples and
                # comparing them to real samples using a pre-trained Inception model.
                # This is computationally intensive and might require specific setup.
                # For now, we just print a placeholder.
                print("Evaluation metrics (FID, KID, IS) calculation is not yet implemented.")
                print("Skipping evaluation for now.")
                
            if global_step % 1000 == 0: # Save checkpoint
                os.makedirs(config.checkpoint_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(config.checkpoint_dir, f"model_step_{global_step}.pth"))
                torch.save(ema_model.state_dict(), os.path.join(config.checkpoint_dir, f"ema_model_step_{global_step}.pth"))
                print(f"Saved checkpoint at step {global_step}")

    print("Training finished.")

if __name__ == "__main__":
    train()


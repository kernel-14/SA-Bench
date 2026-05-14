
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import math

from config import Config
from model import WDNO
from data import get_dataloader, PDEDataset, PDE_Solver_Mock, create_guidance_objective

def train(config: Config):
    # Setup device
    device = config.device
    
    # Initialize model
    model = WDNO(config).to(device)
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    # Learning rate scheduler
    if config.learning_rate_scheduler == "cosine_annealing":
        scheduler = CosineAnnealingLR(optimizer, T_max=config.training_steps)
    elif config.learning_rate_scheduler == "StepLR":
        # Placeholder values, adjust as per paper's baselines if implementing them
        scheduler = StepLR(optimizer, step_size=50000, gamma=0.1) 
    else:
        scheduler = None

    # Data loaders
    train_dataloader = get_dataloader(config, is_train=True)
    # val_dataloader = get_dataloader(config, is_train=False) # Not explicitly used for validation loop yet

    # For control tasks, initialize mock PDE solver for guidance objective
    pde_solver = PDE_Solver_Mock(config).to(device) if config.task_type == "control" else None

    # Training loop
    for step in tqdm(range(config.training_steps), desc="Training"):
        model.train()
        optimizer.zero_grad()

        # Get data batch
        u_data, a_data, f_data, target_u_data = next(iter(train_dataloader))
        u_data = u_data.to(device)
        a_data = a_data.to(device)
        f_data = f_data.to(device) if f_data is not None else None
        target_u_data = target_u_data.to(device) if target_u_data is not None else None

        # Determine x_data and condition_data for WDNO
        if config.task_type == "simulation":
            x_data = u_data # Diffuse the state trajectory
            # Condition on initial condition (u0) and force term (f)
            # For 1D Burgers: a_data (u0) is (B,1,S,C), f_data is (B,T-1,S,C)
            # Need to combine them to form the conditioning input.
            # Paper says W_a for simulation is from equation parameter a.
            # Let's assume a_data contains u0 and f (if applicable) for simplicity.
            # In a real setup, u0 would be `a` and f would be `f` as separate conditional inputs.
            # For now, let's just pass `a_data` as `condition_data` and assume the model handles it.
            # If `f_data` is part of condition, we need to concatenate it or pass separately.
            
            # Per the paper, for simulation, "we include W_a as a conditioning factor."
            # W_a represents parameters like initial conditions and boundary conditions.
            # For 1D Burgers simulation, this is u0 and f. So condition_data should be concatenated.
            
            if config.pde_type == "1d_burgers":
                # For 1D Burgers simulation, condition_data is (u0, f)
                # u0 is (B, 1, S, C), f is (B, T-1, S, C)
                # Need to resize u0 to match time dim of f, or concatenate smartly.
                # A simple approach is to repeat u0 across time dimension to match f.
                # Or, pass them as separate channels if the Unet is set up for it.
                # The Unet takes `cond` as input, which is concatenated with `x` along channel dim.
                # So if `x_data` is `u_data` (B, T, S, C), `cond` should have compatible T, S dims.
                # Reshaping and concatenating `a_data` and `f_data` here.
                
                # Assume a_data (u0) is (B, 1, S, C) and f_data is (B, T-1, S, C)
                # Concatenate u0 and f to form condition input.
                # Need to make sure the time dimension aligns. If u0 is just initial step,
                # we need to make it match the trajectory length.
                
                # For simplicity, if u0 is (B, 1, S, C) and f is (B, T-1, S, C)
                # let's assume `condition_data` refers to `a_data` and the model handles `f_data`
                # as a separate input or it is implicitly part of `a_data` for now.
                # The paper's Algorithm 1 just has `W_a`.
                
                # Let's assume for simplicity, `condition_data` for simulation is `a_data` and `f_data` concatenated and reshaped.
                # The paper implies `a` is "certain parameter functions, such as initial conditions and boundary conditions"
                # For Burgers, this is `u0` and `f`.
                # So `condition_data` needs to combine `u0` and `f`.
                # Let's construct `condition_data` as (B, T, S, 2C) assuming one channel each for u0 (repeated) and f.
                
                # The `PDEDataset` returns `a_data` as initial condition, and `f_data` as force.
                # For simulation, the target `x_data` is `u_data`.
                # The conditioning data `condition_data` should contain both initial condition `a_data` and force `f_data`.
                # `a_data` (u0) is (B, 1, S, C), `f_data` is (B, T-1, S, C).
                # `u_data` is (B, T, S, C).
                
                # Let's construct a condition tensor by repeating `a_data` (u0) for `T` steps and concatenating with `f_data` 
                # (padded with zero for first time step). This is a simplification.
                
                # If x_data is (B, T, S, C) -> input to DWT is (B, C, T, S)
                # Then wavelet_input_x is (B, C_wavelet_total, T_wavelet, S_wavelet)
                # condition_data needs to have compatible T_wavelet, S_wavelet.
                
                # For Burgers simulation: x_data = u_data. condition_data = initial_condition + force_term.
                # Initial condition is (B, 1, S, C). Force term is (B, T_f, S, C).
                # To match dimensions for wavelet transform, let's repeat u0 along time dimension.
                # This needs careful handling. The Unet expects cond to be same spatial/temporal resolution as input.
                
                # Option 1: Concatenate along feature dimension before DWT
                # If x_data.shape = (B, T, S, C_u)
                # and cond_data.shape = (B, T, S, C_cond)
                # then torch.cat([x_data, cond_data], dim=-1) for input to DWT
                # Or, as implemented, pass `cond` separately.
                
                # Let's assume `a_data` is the "a" for simulation and `f_data` is part of `a` as well.
                # For 1D Burgers simulation: `u_data` is (B, T, S, C). `a_data` is initial `u0` (B, 1, S, C). `f_data` is force (B, T-1, S, C).
                # The problem is that the condition_data (`W_a`) has to be of the same spatial/temporal dimensions after wavelet transform as `W_u`.
                # If `x_data` is `u_data` (B, T, S, C), then `condition_data` needs to be `(B, T, S, C_cond)` to match up.
                # So, initial condition `u0` needs to be extended across time and `f` needs to be combined.
                
                # For simplicity, for Burgers simulation, let's treat `a_data` as (B, 1, S, C)
                # and `f_data` (B, T-1, S, C) as part of the overall `condition_data`.
                # Let's repeat `a_data` to match the `T` dimension of `u_data`.
                repeated_u0 = a_data.repeat(1, u_data.shape[1], 1, 1) # (B, T, S, C)
                
                # Pad f_data to match T. Simplistic: assume f has T-1 length, pad first dim.
                # Assuming f_data is (B, T-1, S, C). Add a zero time step at the beginning.
                padded_f_data = torch.cat([torch.zeros_like(a_data), f_data], dim=1) # (B, T, S, C)
                
                # Concatenate repeated_u0 and padded_f_data along channel dimension
                condition_data_for_model = torch.cat([repeated_u0, padded_f_data], dim=-1) # (B, T, S, 2C)
                
                # Need to update config.data_channels for the Unet if cond_channels are used.
                # Unet's cond_channels param is for channels after DWT.
                # So, condition_data here should be the combined (u0, f) in original space.
                
                # Let's simplify: only u0 as condition for now for simulation. This might be too simple.
                # Per paper Alg 1, for simulation, `epsilon_theta(W_u_k, W_a, k)`. `W_a` is the original condition.
                # Let's use `a_data` directly and the `model.forward` will wavelet transform it.
                condition_input = a_data 
                x_input = u_data

            elif config.pde_type == "2d_fluid": # Simulation
                x_input = u_data # (B, D, H, W, C_u)
                # Condition on initial density (a_data) and control sequences (f_data)
                # a_data (B, 1, H, W, C_density), f_data (B, D, H, W, C_force)
                repeated_initial_density = a_data.repeat(1, x_input.shape[1], 1, 1, 1) # (B, D, H, W, C_density)
                condition_input = torch.cat([repeated_initial_density, f_data], dim=-1) # (B, D, H, W, C_density + C_force)
            else:
                x_input = u_data # Diffuse the state trajectory (u)
                if config.pde_type == "1d_burgers":
                    # For 1D Burgers simulation, condition is u0 (a_data) and force (f_data)
                    # a_data (B, 1, S, C), f_data (B, T-1, S, C)
                    # Need to combine to (B, T, S, 2C) where first C is repeated u0, second C is padded f
                    repeated_u0 = a_data.repeat(1, u_data.shape[1], 1, 1) # (B, T, S, C)
                    padded_f_data = torch.cat([torch.zeros_like(a_data), f_data], dim=1) # (B, T, S, C)
                    condition_input = torch.cat([repeated_u0, padded_f_data], dim=-1) # (B, T, S, 2C)
                elif config.pde_type == "2d_fluid":
                    # For 2D fluid simulation, condition is initial density (a_data) and force (f_data)
                    # a_data (B, 1, H, W, C_density), f_data (B, T, H, W, C_force)
                    repeated_initial_density = a_data.repeat(1, x_input.shape[1], 1, 1, 1) # (B, D, H, W, C_density)
                    condition_input = torch.cat([repeated_initial_density, f_data], dim=-1) # (B, D, H, W, C_density + C_force)
                else: # 1D advection, 1D Navier-Stokes, ERA5 simulation
                    x_input = u_data
                    condition_input = a_data
            
        elif config.task_type == "control":
            x_input = f_data # Diffuse the force trajectory (f)
            if config.pde_type == "1d_burgers":
                # For 1D Burgers control, condition is u0 (a_data) and u_T (target_u_data)
                # f_data is (B, T-1, S, C)
                # a_data (u0) is (B, 1, S, C)
                # target_u_data (u_T) is (B, 1, S, C)
                # Repeat u0 and u_T to match T-1 of f_data
                repeated_u0 = a_data.repeat(1, f_data.shape[1], 1, 1) # (B, T-1, S, C)
                repeated_u_T = target_u_data.repeat(1, f_data.shape[1], 1, 1) # (B, T-1, S, C)
                condition_input = torch.cat([repeated_u0, repeated_u_T], dim=-1) # (B, T-1, S, 2C)
            elif config.pde_type == "2d_fluid": # Control
                # For 2D fluid control, condition is initial density (a_data) and target smoke percentage (target_u_data)
                # x_input (f_data) is (B, D, H_f, W_f, C_f)
                # a_data (initial density) is (B, 1, H, W, C_density)
                # target_u_data (target smoke percentage) is (B, 1, 1, 1, 1)
                repeated_initial_density = a_data.repeat(1, x_input.shape[1], 1, 1, 1) # (B, D, H, W, C_density)
                repeated_target_u = target_u_data.repeat(1, x_input.shape[1], x_input.shape[2], x_input.shape[3], 1)
                condition_input = torch.cat([repeated_initial_density, repeated_target_u], dim=-1)
            else:
                raise ValueError(f"Control task not implemented for PDE type: {config.pde_type}")
        else:
            raise ValueError(f"Unknown task type: {config.task_type}")

        # Forward pass
        loss = model(x_input, condition_input)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        # Logging
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

        # Save model checkpoint
        if step % 1000 == 0 or step == config.training_steps - 1:
            os.makedirs(config.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(config.output_dir, f"model_step_{step}.pt"))
            print(f"Model saved at step {step}")

    print("Training complete.")

def evaluate(config: Config):
    device = config.device
    model = WDNO(config).to(device)
    # Load best model checkpoint
    # model.load_state_dict(torch.load(os.path.join(config.output_dir, "best_model.pt"))) # Or last model
    
    # For demonstration, load the last saved model
    latest_checkpoint = sorted([f for f in os.listdir(config.output_dir) if f.startswith("model_step_")])[-1]
    model.load_state_dict(torch.load(os.path.join(config.output_dir, latest_checkpoint)))
    model.eval()

    test_dataloader = get_dataloader(config, is_train=False)

    total_mse = 0
    num_samples = 0

    with torch.no_grad():
        for i, (u_data, a_data, f_data, target_u_data) in enumerate(tqdm(test_dataloader, desc="Evaluating")):
            u_data = u_data.to(device)
            a_data = a_data.to(device)
            f_data = f_data.to(device) if f_data is not None else None
            target_u_data = target_u_data.to(device) if target_u_data is not None else None

            # Determine input for sampling
            if config.task_type == "simulation":
                # For simulation, we want to predict u_data. `x_data` is the target output shape.
                # `shape` for sampling needs to be the wavelet domain shape.
                # Since `model.sample` reconstructs data from wavelet coeffs, we need to pass
                # the expected shape of the wavelet coefficients.
                
                # Let's calculate the wavelet output shape. This needs to be dynamic.
                # For J=1:
                # 2D DWT: (H, W) -> (H/2, W/2)
                # 3D DWT: (D, H, W) -> (D/2, H/2, W/2)
                
                if config.pde_type == "1d_burgers":
                    # Target output for u_data is (B, T, S, C)
                    T_wt = math.ceil(u_data.shape[1] / 2)
                    S_wt = math.ceil(u_data.shape[2] / 2)
                    C_total_wavelet_x = config.raw_input_channels_x * model.factor_wavelet_channels
                    sample_shape = (u_data.shape[0], C_total_wavelet_x, T_wt, S_wt)
                    
                    repeated_u0 = a_data.repeat(1, u_data.shape[1], 1, 1) # (B, T, S, C)
                    zero_f_timestep = torch.zeros(f_data.shape[0], 1, f_data.shape[2], f_data.shape[3], device=f_data.device, dtype=f_data.dtype)
                    padded_f_data = torch.cat([zero_f_timestep, f_data], dim=1) # (B, T, S, C)
                    condition_input = torch.cat([repeated_u0, padded_f_data], dim=-1) # (B, T, S, 2 * C_data)
                elif config.pde_type == "2d_fluid":
                    # Target output for u_data is (B, D, H, W, C)
                    D_wt = math.ceil(u_data.shape[1] / 2)
                    H_wt = math.ceil(u_data.shape[2] / 2)
                    W_wt = math.ceil(u_data.shape[3] / 2)
                    C_total_wavelet_x = config.raw_input_channels_x * model.factor_wavelet_channels
                    sample_shape = (u_data.shape[0], C_total_wavelet_x, D_wt, H_wt, W_wt)
                    
                    repeated_initial_density = a_data.repeat(1, u_data.shape[1], 1, 1, 1) # (B, D, H, W, C_density)
                    condition_input = torch.cat([repeated_initial_density, f_data], dim=-1) # (B, D, H, W, C_density + C_force)
                elif config.pde_type in ["1d_advection", "1d_navier_stokes", "era5"]:
                    if model.is_3d:
                        D_wt = math.ceil(u_data.shape[1] / 2)
                        H_wt = math.ceil(u_data.shape[2] / 2)
                        W_wt = math.ceil(u_data.shape[3] / 2)
                        C_total_wavelet_x = config.raw_input_channels_x * model.factor_wavelet_channels
                        sample_shape = (u_data.shape[0], C_total_wavelet_x, D_wt, H_wt, W_wt)
                    else:
                        T_wt = math.ceil(u_data.shape[1] / 2)
                        S_wt = math.ceil(u_data.shape[2] / 2)
                        C_total_wavelet_x = config.raw_input_channels_x * model.factor_wavelet_channels
                        sample_shape = (u_data.shape[0], C_total_wavelet_x, T_wt, S_wt)
                    condition_input = a_data # Use a_data as the primary condition, e.g., initial state

                predicted_u = model.sample(sample_shape, condition_data=condition_input, low_res_data=None, guidance_func=None)
                # Compare predicted_u with actual u_data
                mse = F.mse_loss(predicted_u, u_data) # This is assuming predicted_u also contains channels C
                total_mse += mse.item() * u_data.shape[0]
                num_samples += u_data.shape[0]

            elif config.task_type == "control":
                # For control, we want to predict f_data. `x_data` is the target output shape.
                # `condition_data` is u0 and u_T.
                
                if config.pde_type == "1d_burgers":
                    # f_data is (B, T-1, S, C_f)
                    T_wt = math.ceil(f_data.shape[1] / 2)
                    S_wt = math.ceil(f_data.shape[2] / 2)
                    C_total_wavelet_x = config.raw_input_channels_x * model.factor_wavelet_channels # Channels of f_data * 4
                    sample_shape = (f_data.shape[0], C_total_wavelet_x, T_wt, S_wt)
                    
                    repeated_u0 = a_data.repeat(1, f_data.shape[1], 1, 1) # (B, T-1, S, C)
                    repeated_u_T = target_u_data.repeat(1, f_data.shape[1], 1, 1) # (B, T-1, S, C)
                    condition_input = torch.cat([repeated_u0, repeated_u_T], dim=-1) # (B, T-1, S, 2 * C_data)
                elif config.pde_type == "2d_fluid": # Control
                    # Target output for f_data is (B, D, H_f, W_f, C_f)
                    D_wt = math.ceil(f_data.shape[1] / 2)
                    H_wt = math.ceil(f_data.shape[2] / 2)
                    W_wt = math.ceil(f_data.shape[3] / 2)
                    C_total_wavelet_x = config.raw_input_channels_x * model.factor_wavelet_channels # Channels of f_data * 8
                    sample_shape = (f_data.shape[0], C_total_wavelet_x, D_wt, H_wt, W_wt)

                    repeated_initial_density = a_data.repeat(1, f_data.shape[1], 1, 1, 1) # (B, D, H, W, C_density)
                    repeated_target_u = target_u_data.repeat(1, f_data.shape[1], f_data.shape[2], f_data.shape[3], 1)
                    condition_input = torch.cat([repeated_initial_density, repeated_target_u], dim=-1)

                guidance_func = create_guidance_objective(config, pde_solver, target_u_data, a_data)
                predicted_f = model.sample(sample_shape, condition_data=condition_input, 
                                           low_res_data=None, guidance_func=guidance_func, 
                                           guidance_weight=config.guidance_weight)
                
                # For control, we evaluate the objective I, not MSE of f.
                # This requires re-calculating I for predicted_f.
                
                # Note: `create_guidance_objective` returns a function that takes W_f_hat
                # To calculate the final objective, we need the actual `predicted_f`
                # (in original space) and then apply `pde_solver` etc.
                
                # For simplicity, if we need to get the "value" of I, we need to manually apply
                # the guidance_objective_func *without* autograd.
                # The guidance_func created returns a tensor whose gradient is calculated.
                # Here, we need the actual objective value.
                
                # The `sample` method should return the reconstructed `f`.
                # Then we pass this `f` to the *real* PDE solver to get `u`
                # and compute the objective `I`.
                
                # For now, let's just log a dummy objective value if actual solver is not available.
                # The `create_guidance_objective` returns the objective for W_f_hat.
                # We need to compute it for the final `predicted_f` (reconstructed).
                
                # To calculate the objective value on the sampled output:
                # 1. Apply DWT to predicted_f to get W_predicted_f.
                # 2. Pass W_predicted_f to the guidance_objective_func (without autograd context)
                # This seems overly complicated for evaluation of the final output.
                # Let's assume for evaluation, we want to know the MSE of f, or simply log a dummy for I.
                
                # Let's compute a mock objective value based on the predicted f.
                # The guidance_func is defined to operate on W_f_hat, not the final f.
                # So we would need to DWT `predicted_f` first, then calculate the objective.
                
                # Let's define a separate evaluation objective function or modify guidance_objective_func
                # to accept original-space `f` and return `I`.
                
                # For evaluation, we directly compute the objective on the generated f.
                # The `PDE_Solver_Mock` takes `u0` and `f_recon` (original space).
                
                f_recon_for_eval = predicted_f # This is already in original space

                if config.pde_type == "1d_burgers":
                    u_T_pred = pde_solver(a_data, f_recon_for_eval)
                    term1 = ((u_T_pred - target_u_data)**2).mean(dim=[-1, -2, -3])
                    term2 = (f_recon_for_eval**2).mean(dim=[-1, -2, -3])
                    current_objective_I = (term1 + config.guidance_weight * term2).mean().item()
                elif config.pde_type == "2d_fluid":
                    current_objective_I = pde_solver(a_data, f_recon_for_eval).mean().item() # Mean across batch
                else:
                    current_objective_I = 0.0 # Placeholder
                    
                total_mse += current_objective_I * predicted_f.shape[0] # Using I as error
                num_samples += predicted_f.shape[0]

            else:
                raise ValueError("Evaluation not implemented for this task type.")
                
    if config.task_type == "simulation":
        avg_mse = total_mse / num_samples
        print(f"Average MSE on test set: {avg_mse:.4f}")
    elif config.task_type == "control":
        avg_objective_I = total_mse / num_samples # Here total_mse accumulated I values
        print(f"Average objective I on test set: {avg_objective_I:.4f}")


if __name__ == "__main__":
    cfg = Config(experiment_name="wdno_burgers_simulation")
    cfg.update_for_pde("1d_burgers")
    cfg.task_type = "simulation" # Set task type here

    print(f"Starting training for {cfg.pde_type} {cfg.task_type}...")
    train(cfg)
    print(f"Starting evaluation for {cfg.pde_type} {cfg.task_type}...")
    evaluate(cfg)

    # Example for control task
    # cfg_control = Config(experiment_name="wdno_burgers_control")
    # cfg_control.update_for_pde("1d_burgers")
    # cfg_control.task_type = "control"
    # print(f"Starting training for {cfg_control.pde_type} {cfg_control.task_type}...")
    # train(cfg_control)
    # print(f"Starting evaluation for {cfg_control.pde_type} {cfg_control.task_type}...")
    # evaluate(cfg_control)

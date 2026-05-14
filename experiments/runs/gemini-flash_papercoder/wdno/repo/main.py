import argparse
import os
import sys
import yaml
import torch
import torch.optim
import torch.optim.lr_scheduler
from typing import Any, Dict, List, Optional, Type, Union

# Local imports
from config import Config
from utils import seed_everything, get_device, find_latest_checkpoint
from pde_solvers import BurgersPdeSolver, PdeSolver # Import specific solvers as needed
from wavelet_utils import WaveletTransformManager
from diffusion_model import NoiseScheduler, WaveletDiffusionUNet
from wdno_models import BaseResolutionModel, SuperResolutionModel
from data_module import DataModule
from trainer import DiffusionTrainer
from evaluator import Evaluator


def _get_optimizer_and_scheduler(model: torch.nn.Module, config: Config) -> \
    Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
    """
    Initializes the optimizer and learning rate scheduler based on configuration.

    Args:
        model: The PyTorch model to optimize.
        config: The configuration object.

    Returns:
        A tuple containing the optimizer and optionally the LR scheduler.
    """
    if config.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    else:
        raise ValueError(f"Optimizer {config.optimizer} not supported.")

    scheduler = None
    if config.learning_rate_scheduler == "cosine_annealing":
        # Assumes training_steps is total steps, T_max for cosine annealing
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.training_steps)
    elif config.learning_rate_scheduler != "none":
        raise ValueError(f"Learning rate scheduler {config.learning_rate_scheduler} not supported.")

    return optimizer, scheduler


def main(cli_args: Optional[argparse.Namespace] = None):
    """
    Main function to run the WDNO training and evaluation pipeline.

    Args:
        cli_args: Command-line arguments parsed by argparse. If None,
                  arguments are parsed from sys.argv.
    """
    # 1. Configuration Loading
    config = Config(cli_args=cli_args)
    
    # 2. Global Setup
    seed_everything(config.seed)
    config.device = get_device(config.device) # Update config with actual device used
    os.makedirs(config.save_path, exist_ok=True)
    print(f"Using device: {config.device}")
    print(f"Saving checkpoints to: {config.save_path}")

    # 3. Module Initialization (Order matters due to dependencies)

    # 3.1. PDE Solver
    pde_solver: PdeSolver
    if config.problem_type == "1d_burgers":
        pde_solver = BurgersPdeSolver(config)
    # Add other PDE solvers here as they are implemented
    elif config.problem_type == "1d_advection":
        # Assuming 1d_advection can use a similar solver for control guidance evaluation,
        # or it will raise NotImplementedError if not supported by base class for simulation.
        print("Warning: 1D Advection PDE solver not fully implemented for data generation/control evaluation.")
        pde_solver = PdeSolver(config, config.problem_type) # Placeholder for actual solver
    elif config.problem_type == "1d_navier_stokes":
        print("Warning: 1D Navier-Stokes PDE solver not fully implemented for data generation/control evaluation.")
        pde_solver = PdeSolver(config, config.problem_type) # Placeholder
    elif config.problem_type == "2d_fluid":
        print("Warning: 2D Fluid PDE solver not fully implemented for data generation/control evaluation.")
        pde_solver = PdeSolver(config, config.problem_type) # Placeholder
    elif config.problem_type == "era5":
        print("Warning: ERA5 (real-world dataset) has no active PDE solver for data generation/control evaluation.")
        pde_solver = PdeSolver(config, config.problem_type) # Placeholder
    else:
        raise ValueError(f"Unsupported problem_type: {config.problem_type}")

    # 3.2. Wavelet Transform Manager
    # wavelet_data_dim: 2 for 1D spatial problems (time x space), 3 for 2D spatial problems (time x H x W)
    wavelet_manager = WaveletTransformManager(
        wavelet_type=config.wavelet_type,
        mode=config.wavelet_mode,
        wavelet_data_dim=config.data_dim,
        device=str(config.device)
    )

    # 3.3. Noise Scheduler
    noise_scheduler = NoiseScheduler(config.ddpm_timesteps, device=str(config.device))

    # 3.4. Data Module (Loads data, preprocesses, gets channel info)
    data_module = DataModule(config, wavelet_manager, pde_solver)
    
    # Get channel information for UNet initialization after DataModule setup
    br_input_channels = data_module.get_br_input_channels()
    br_condition_channels = data_module.get_br_condition_channels()
    srm_input_channels = data_module.get_srm_input_channels() if config.super_resolution_task['enabled'] else None
    srm_condition_channels = data_module.get_srm_condition_channels() if config.super_resolution_task['enabled'] else None

    # 3.5. Wavelet Diffusion UNet (BRM)
    unet_br = WaveletDiffusionUNet(
        in_channels=br_input_channels,
        out_channels=br_input_channels, # UNet predicts noise of same shape as input
        cond_channels=br_condition_channels,
        time_embedding_dim=config.unet_time_embedding_dimension,
        dim_mults=config.unet_dimension_multipliers,
        num_down_up_layers=config.unet_num_down_up_layers,
        resnet_block_groups=config.unet_resnet_block_groups,
        attn_heads=config.unet_attention_heads,
        conv_kernel_size=config.unet_conv_kernel_size,
        conv_padding=config.unet_conv_padding,
        conv_stride=config.unet_conv_stride,
        is_3d=(config.data_dim == 3)
    ).to(config.device)

    # 3.6. Base-Resolution Model
    br_model = BaseResolutionModel(
        unet=unet_br,
        noise_scheduler=noise_scheduler,
        config=config,
        wavelet_manager=wavelet_manager,
        problem_mode=('control' if config.control_task['enabled'] else 'simulation'),
        pde_solver=pde_solver
    )

    # 3.7. Super-Resolution Model (if enabled)
    srm_model: Optional[SuperResolutionModel] = None
    unet_srm: Optional[WaveletDiffusionUNet] = None
    if config.super_resolution_task['enabled']:
        if srm_input_channels is None or srm_condition_channels is None:
            raise ValueError("SRM input/condition channels not determined but SR task is enabled.")
        unet_srm = WaveletDiffusionUNet(
            in_channels=srm_input_channels,
            out_channels=srm_input_channels,
            cond_channels=srm_condition_channels,
            time_embedding_dim=config.unet_time_embedding_dimension,
            dim_mults=config.unet_dimension_multipliers,
            num_down_up_layers=config.unet_num_down_up_layers,
            resnet_block_groups=config.unet_resnet_block_groups,
            attn_heads=config.unet_attention_heads,
            conv_kernel_size=config.unet_conv_kernel_size,
            conv_padding=config.unet_conv_padding,
            conv_stride=config.unet_conv_stride,
            is_3d=(config.data_dim == 3)
        ).to(config.device)
        srm_model = SuperResolutionModel(
            unet=unet_srm,
            noise_scheduler=noise_scheduler,
            config=config,
            wavelet_manager=wavelet_manager
        )

    # 3.8. Evaluator
    evaluator = Evaluator(config, wavelet_manager, pde_solver)

    # 4. Workflow Execution Based on Mode
    if cli_args.mode == 'train':
        print("\n--- Training Phase ---")
        # Initialize trainer here, can be reused or re-instantiated
        
        # BRM Training
        print("\nTraining Base-Resolution Model (BRM)...")
        optimizer_br, scheduler_br = _get_optimizer_and_scheduler(br_model, config)
        trainer_br = DiffusionTrainer(config, br_model, optimizer_br, scheduler_br)
        br_train_dataloader = data_module.get_single_resolution_dataloader(
            split='train',
            resolution_idx=0,
            batch_size=config.train_batch_size,
            shuffle=True,
            problem_mode=('control' if config.control_task['enabled'] else 'simulation')
        )
        trainer_br.train(
            dataloader=br_train_dataloader,
            total_steps=config.training_steps,
            model_type_str='BRM'
        )
        
        # SRM Training (if enabled)
        if config.super_resolution_task['enabled'] and srm_model is not None:
            print("\nTraining Super-Resolution Model (SRM)...")
            optimizer_srm, scheduler_srm = _get_optimizer_and_scheduler(srm_model, config)
            trainer_srm = DiffusionTrainer(config, srm_model, optimizer_srm, scheduler_srm)
            srm_train_dataloader = data_module.get_multi_resolution_dataloader(
                batch_size=config.train_batch_size,
                shuffle=True
            )
            trainer_srm.train(
                dataloader=srm_train_dataloader,
                total_steps=config.training_steps,
                model_type_str='SRM'
            )

    elif cli_args.mode == 'evaluate':
        print("\n--- Evaluation Phase ---")
        # Load BRM checkpoint
        br_checkpoint_path = cli_args.br_checkpoint or find_latest_checkpoint(config.save_path, 'BRM')
        if br_checkpoint_path is None:
            raise FileNotFoundError("No BRM checkpoint specified or found for evaluation.")
        print(f"Loading BRM from {br_checkpoint_path}")
        optimizer_br_dummy, scheduler_br_dummy = _get_optimizer_and_scheduler(br_model, config) # Need dummy optimizer for loading state_dict
        load_checkpoint(br_model, optimizer_br_dummy, scheduler_br_dummy, br_checkpoint_path, config.device)

        # Load SRM checkpoint if SR task is enabled
        if config.super_resolution_task['enabled'] and srm_model is not None:
            srm_checkpoint_path = cli_args.sr_checkpoint or find_latest_checkpoint(config.save_path, 'SRM')
            if srm_checkpoint_path is None:
                print("Warning: SRM task enabled but no SRM checkpoint found. SR evaluation will be skipped.")
                config.super_resolution_task['enabled'] = False # Disable SR evaluation
            else:
                print(f"Loading SRM from {srm_checkpoint_path}")
                optimizer_srm_dummy, scheduler_srm_dummy = _get_optimizer_and_scheduler(srm_model, config) # Need dummy optimizer
                load_checkpoint(srm_model, optimizer_srm_dummy, scheduler_srm_dummy, srm_checkpoint_path, config.device)

        # Evaluation tasks
        if config.simulation_task['enabled']:
            print("\nEvaluating Simulation Task...")
            simulation_test_dataloader = data_module.get_single_resolution_dataloader(
                split='test_simulation',
                resolution_idx=0,
                batch_size=1, # Evaluate samples one by one
                shuffle=False,
                problem_mode='simulation'
            )
            data_stats_for_sim = data_module.get_data_stats('simulation_output') # Get stats for 'u'
            simulation_metrics = evaluator.evaluate_simulation(br_model, simulation_test_dataloader, data_stats_for_sim)
            print(f"Simulation Metrics: {simulation_metrics}")

        if config.control_task['enabled']:
            print("\nEvaluating Control Task...")
            control_test_dataloader = data_module.get_single_resolution_dataloader(
                split='test_control',
                resolution_idx=0,
                batch_size=1, # Evaluate samples one by one
                shuffle=False,
                problem_mode='control'
            )
            data_stats_for_control = data_module.get_data_stats('control_output') # Get stats for 'f'
            control_metrics = evaluator.evaluate_control(br_model, control_test_dataloader, data_stats_for_control)
            print(f"Control Metrics: {control_metrics}")

        if config.super_resolution_task['enabled'] and srm_model is not None:
            print("\nEvaluating Super-Resolution Task...")
            sr_test_dataloader = data_module.get_sr_test_dataloader(
                batch_size=1, # Evaluate samples one by one
                shuffle=False
            )
            data_stats_for_sr = data_module.get_data_stats('simulation_output') # Get stats for 'u'
            sr_metrics = evaluator.evaluate_super_resolution(br_model, srm_model, sr_test_dataloader, data_stats_for_sr)
            print(f"Super-Resolution Metrics: {sr_metrics}")
        
        # Ablation studies can be added here, potentially requiring specific CLI flags or config entries
        # if cli_args.ablation == 'fft_comparison':
        #     evaluator.run_fft_comparison(...)
        # ...

    else:
        raise ValueError(f"Unknown mode: {cli_args.mode}. Use 'train' or 'evaluate'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Wavelet Diffusion Neural Operator (WDNO) experiments.")
    parser.add_argument('--problem_name', type=str, default='1d_burgers',
                        help="Name of the PDE problem to run (e.g., '1d_burgers', '2d_fluid').")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'],
                        help="Operation mode: 'train' or 'evaluate'.")
    parser.add_argument('--config_path', type=str, default='config.yaml',
                        help="Path to the main YAML configuration file.")
    parser.add_argument('--br_checkpoint', type=str, default=None,
                        help="Path to a specific Base-Resolution Model checkpoint for evaluation or resuming training.")
    parser.add_argument('--sr_checkpoint', type=str, default=None,
                        help="Path to a specific Super-Resolution Model checkpoint for evaluation or resuming training.")
    parser.add_argument('--ablation', type=str, default=None,
                        help="Optional: specifies a particular ablation study to run (e.g., 'fft_comparison').")
    # Add other common CLI overrides that map directly to config parameters
    parser.add_argument('--device', type=str, default=None, help="Device to use (e.g., 'cuda', 'cpu').")
    parser.add_argument('--save_path', type=str, default=None, help="Path to save checkpoints and logs.")
    parser.add_argument('--seed', type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument('--learning_rate', type=float, default=None, help="Learning rate.")
    parser.add_argument('--train_batch_size', type=int, default=None, help="Training batch size.")
    parser.add_argument('--training_steps', type=int, default=None, help="Total training steps.")
    parser.add_argument('--ddim_steps', type=int, default=None, help="DDIM sampling iterations.")
    parser.add_argument('--ddim_eta', type=float, default=None, help="DDIM eta parameter.")
    parser.add_argument('--guidance_lambda', type=float, default=None, help="Weight for control guidance.")

    args = parser.parse_args()
    main(args)


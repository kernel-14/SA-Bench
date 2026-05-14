## main.py
import os
import torch
import collections
import yaml
from typing import Dict, Any, Callable, Tuple, List

# Import project modules
from config import Config
from utils import set_seed, get_device, count_parameters
from dataset_manager import DatasetManager
from trainer import Trainer
from evaluator import Evaluator

# Import model components
from models.neural_operator import NeuralOperatorModel
from models.adapters import LiftingAdapter, ProjectionAdapter
from models.base_operator import CoreOperator
from models.fno import FNO
from models.mamba_fno import MambaFNO
from models.perceiver_fno import PerceiverFNO
from models.swin_v2 import SwinV2
from models.codano import CoDANo


# Map core operator type names to their classes for dynamic instantiation in Trainer
CORE_OPERATOR_CLASSES = {
    "FNO": FNO,
    "MambaFNO": MambaFNO,
    "PerceiverFNO": PerceiverFNO,
    "SwinV2": SwinV2,
    "CoDANo": CoDANo,
}

def main():
    """
    Main function to run the Universal Neural Operators reproduction experiments.
    It orchestrates configuration loading, data preparation, model initialization,
    training (pre-training, fine-tuning, and scratch), and evaluation across
    different scenarios and core operator architectures.
    """
    print("--- Starting Universal Neural Operators Reproduction ---")

    # 1. Load Configuration
    config_file_path = "config.yaml"
    config = Config.load_config(config_file_path)
    print(f"Configuration loaded from {config_file_path}")

    # 2. Global Setup for Reproducibility and Device
    set_seed(config.seed)
    device = get_device(config.device)
    print(f"Using device: {device}")

    # 3. Initialize Components
    dataset_manager = DatasetManager(config)
    
    # Pass NeuralOperatorModel directly as the model_factory to Trainer.
    # The Trainer will then internally create Lifting, Core, and Projection
    # instances and compose them into a NeuralOperatorModel.
    trainer = Trainer(config, NeuralOperatorModel, dataset_manager)
    evaluator = Evaluator(config, dataset_manager)

    # Dictionary to store all experiment results
    all_results = collections.defaultdict(lambda: collections.defaultdict(dict))

    # Define the list of core operator types to test based on available implementations
    core_operator_types = list(CORE_OPERATOR_CLASSES.keys())
    # For initial testing or specific runs, uncomment and modify:
    # core_operator_types = ["FNO"] 

    # Iterate through each Core Operator type to be evaluated
    for core_type in core_operator_types:
        print(f"\n======== Running experiments for Core Operator: {core_type} ========")
        # Set the current core operator type in the config, which Trainer and Evaluator will use
        config.model_settings['core_operator_type'] = core_type 
        config.model_type = core_type # Also update for potential logging/reporting

        # Placeholder for storing results for the current core_type
        current_model_results = collections.defaultdict(dict)

        # The core operator is pre-trained once and then its state is reused
        # for fine-tuning across all subsequent scenarios for the current `core_type`.
        pretrained_core_state = None 

        # =====================================================================
        # Scenario 1: Out-of-sample parameter values
        # PDEs: Burgers, Gray-Scott, Navier-Stokes for pre-training and fine-tuning.
        # =====================================================================
        print(f"\n--- Scenario 1: Out-of-sample parameter values ({core_type}) ---")
        
        # --- Pre-training Phase (for Scenario 1 PDEs) ---
        # The paper implies a single pre-training on diverse multiphysics tasks.
        # We perform pre-training on Burgers, Gray-Scott, and Navier-Stokes with their 'pretrain' parameter ranges.
        pretrain_pdes_s1 = ['burgers', 'gray_scott', 'navier_stokes']
        pretrain_dataloaders_s1: Dict[str, Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]] = {}
        for pde_name in pretrain_pdes_s1:
            print(f"  Generating data for {pde_name} (pre-training S1)...")
            # Generate raw data (inputs, outputs, min_y, max_y) using pretrain parameter ranges
            x_list, y_list, min_y, max_y = dataset_manager._generate_pde_data(
                config.pde_configs[pde_name], is_pretrain=True
            )
            # Store min/max values for this specific dataset key, used by Evaluator
            dataset_key_p = f"{pde_name}_pretrain_s1_dataset"
            dataset_manager.dataset_min_max_vals[dataset_key_p] = (min_y, max_y)
            # Get DataLoaders for this PDE
            train_dl, val_dl, test_dl = dataset_manager.get_dataloaders(
                x_list, y_list, dataset_key_p,
                config.training_settings['batch_size'], shuffle=True
            )
            pretrain_dataloaders_s1[pde_name] = (train_dl, val_dl, test_dl)

        # Execute the pre-training phase using the Trainer
        print(f"\n  Starting Pre-training for Scenario 1 tasks with {core_type} core...")
        pretrain_results_s1 = trainer.pretrain(pretrain_dataloaders_s1)
        pretrained_core_state = pretrain_results_s1['best_core_state']
        print(f"  Pre-training finished for {core_type}. Best core state saved.")
        
        # --- Fine-tuning & Scratch Training Phase (for Scenario 1 PDEs) ---
        # Evaluate on the same PDEs but with 'finetune' parameter ranges.
        finetune_pdes_s1 = ['burgers', 'gray_scott', 'navier_stokes'] 
        for pde_name in finetune_pdes_s1:
            print(f"\n  Processing {pde_name} (Scenario 1 Fine-tuning & Scratch) with {core_type}...")

            # Generate data for fine-tuning/scratch (using finetune parameter ranges)
            x_list_ft, y_list_ft, min_y_ft, max_y_ft = dataset_manager._generate_pde_data(
                config.pde_configs[pde_name], is_pretrain=False # False for finetune ranges
            )
            ft_dataset_key = f"{pde_name}_finetune_s1_dataset"
            dataset_manager.dataset_min_max_vals[ft_dataset_key] = (min_y_ft, max_y_ft)
            ft_train_dl, ft_val_dl, ft_test_dl = dataset_manager.get_dataloaders(
                x_list_ft, y_list_ft, ft_dataset_key,
                config.training_settings['batch_size'], shuffle=True
            )

            # Fine-tuning: Initialize adapters and train with fixed core
            print(f"    Fine-tuning {pde_name}...")
            finetuned_model = trainer.finetune(pde_name, (ft_train_dl, ft_val_dl, ft_test_dl), pretrained_core_state)
            ft_metrics = evaluator.evaluate(finetuned_model, ft_test_dl, ft_dataset_key)
            current_model_results[pde_name]['finetuned_s1'] = ft_metrics
            current_model_results[pde_name]['finetuned_s1']['param_count'] = count_parameters(finetuned_model) # Full model count
            print(f"    {pde_name} Fine-tuned metrics: {ft_metrics}")

            # Scratch Training: Initialize and train full model from scratch
            print(f"    Training {pde_name} from Scratch...")
            scratch_model = trainer.train_from_scratch(pde_name, (ft_train_dl, ft_val_dl, ft_test_dl))
            scratch_metrics = evaluator.evaluate(scratch_model, ft_test_dl, ft_dataset_key)
            current_model_results[pde_name]['scratch_s1'] = scratch_metrics
            current_model_results[pde_name]['scratch_s1']['param_count'] = count_parameters(scratch_model) # Full model count
            print(f"    {pde_name} Scratch metrics: {scratch_metrics}")


        # =====================================================================
        # Scenario 2: Input function set extension
        # PDEs: Heat -> Heat+Convection; Reaction-Diffusion -> Reaction-Diffusion+Advection
        # The core operator from Scenario 1 pre-training is reused for fine-tuning here.
        # =====================================================================
        print(f"\n--- Scenario 2: Input function set extension ({core_type}) ---")

        # Fine-tuning/Scratch targets for Scenario 2
        # `ft_pdes_s2` maps base PDE name to its extended version.
        ft_pdes_s2 = {'heat': 'heat_convection', 'reaction_diffusion': 'reaction_diffusion_advection'}
        for base_pde, ft_pde_name in ft_pdes_s2.items():
            print(f"\n  Processing {ft_pde_name} (Scenario 2 Fine-tuning & Scratch) with {core_type}...")

            # Generate data for fine-tuning/scratch (uses `is_pretrain=False` to pick specific config)
            x_list_ft, y_list_ft, min_y_ft, max_y_ft = dataset_manager._generate_pde_data(
                config.pde_configs[ft_pde_name], is_pretrain=False
            )
            ft_dataset_key = f"{ft_pde_name}_finetune_s2_dataset"
            dataset_manager.dataset_min_max_vals[ft_dataset_key] = (min_y_ft, max_y_ft)
            ft_train_dl, ft_val_dl, ft_test_dl = dataset_manager.get_dataloaders(
                x_list_ft, y_list_ft, ft_dataset_key,
                config.training_settings['batch_size'], shuffle=True
            )

            # Fine-tuning
            print(f"    Fine-tuning {ft_pde_name}...")
            finetuned_model = trainer.finetune(ft_pde_name, (ft_train_dl, ft_val_dl, ft_test_dl), pretrained_core_state)
            ft_metrics = evaluator.evaluate(finetuned_model, ft_test_dl, ft_dataset_key)
            current_model_results[ft_pde_name]['finetuned_s2'] = ft_metrics
            current_model_results[ft_pde_name]['finetuned_s2']['param_count'] = count_parameters(finetuned_model)
            print(f"    {ft_pde_name} Fine-tuned metrics: {ft_metrics}")

            # Scratch Training
            print(f"    Training {ft_pde_name} from Scratch...")
            scratch_model = trainer.train_from_scratch(ft_pde_name, (ft_train_dl, ft_val_dl, ft_test_dl))
            scratch_metrics = evaluator.evaluate(scratch_model, ft_test_dl, ft_dataset_key)
            current_model_results[ft_pde_name]['scratch_s2'] = scratch_metrics
            current_model_results[ft_pde_name]['scratch_s2']['param_count'] = count_parameters(scratch_model)
            print(f"    {ft_pde_name} Scratch metrics: {scratch_metrics}")


        # =====================================================================
        # Scenario 3: General multi-physics learning (PDEBench based)
        # Pre-train with Advection and Burgers, fine-tune on Reaction-Diffusion.
        # The core operator from Scenario 1 pre-training is reused for fine-tuning here.
        # =====================================================================
        print(f"\n--- Scenario 3: General multi-physics learning ({core_type}) ---")

        # Fine-tuning/Scratch target for Scenario 3 (Reaction-Diffusion from PDEBench-like)
        ft_pde_name_s3 = 'reaction_diffusion_pdebench'
        print(f"\n  Processing {ft_pde_name_s3} (Scenario 3 Fine-tuning & Scratch) with {core_type}...")

        # Generate data for fine-tuning/scratch
        x_list_ft_s3, y_list_ft_s3, min_y_ft_s3, max_y_ft_s3 = dataset_manager._generate_pde_data(
            config.pde_configs[ft_pde_name_s3], is_pretrain=False
        )
        ft_dataset_key_s3 = f"{ft_pde_name_s3}_finetune_s3_dataset"
        dataset_manager.dataset_min_max_vals[ft_dataset_key_s3] = (min_y_ft_s3, max_y_ft_s3)
        ft_train_dl_s3, ft_val_dl_s3, ft_test_dl_s3 = dataset_manager.get_dataloaders(
            x_list_ft_s3, y_list_ft_s3, ft_dataset_key_s3,
            config.training_settings['batch_size'], shuffle=True
        )

        # Fine-tuning
        print(f"    Fine-tuning {ft_pde_name_s3}...")
        finetuned_model_s3 = trainer.finetune(ft_pde_name_s3, (ft_train_dl_s3, ft_val_dl_s3, ft_test_dl_s3), pretrained_core_state)
        ft_metrics_s3 = evaluator.evaluate(finetuned_model_s3, ft_test_dl_s3, ft_dataset_key_s3)
        current_model_results[ft_pde_name_s3]['finetuned_s3'] = ft_metrics_s3
        current_model_results[ft_pde_name_s3]['finetuned_s3']['param_count'] = count_parameters(finetuned_model_s3)
        print(f"    {ft_pde_name_s3} Fine-tuned metrics: {ft_metrics_s3}")

        # Scratch Training
        print(f"    Training {ft_pde_name_s3} from Scratch...")
        scratch_model_s3 = trainer.train_from_scratch(ft_pde_name_s3, (ft_train_dl_s3, ft_val_dl_s3, ft_test_dl_s3))
        scratch_metrics_s3 = evaluator.evaluate(scratch_model_s3, ft_test_dl_s3, ft_dataset_key_s3)
        current_model_results[ft_pde_name_s3]['scratch_s3'] = scratch_metrics_s3
        current_model_results[ft_pde_name_s3]['scratch_s3']['param_count'] = count_parameters(scratch_model_s3)
        print(f"    {ft_pde_name_s3} Scratch metrics: {scratch_metrics_s3}")

        # Store results for the current core operator type
        all_results[core_type] = current_model_results

    # 4. Report All Results
    print("\n\n======== All Experiment Results Summary ========")
    for core_type, results_by_pde in all_results.items():
        print(f"\n--- Core Operator: {core_type} ---")
        for pde_name, pde_results in results_by_pde.items():
            print(f"  PDE: {pde_name}")
            for training_method, metrics in pde_results.items():
                print(f"    {training_method}:")
                for metric_name, value in metrics.items():
                    if metric_name == 'param_count':
                        print(f"      {metric_name}: {value}")
                    else:
                        print(f"      {metric_name}: {value:.6f}")

    # Optional: Save results to a file
    results_output_dir = config.evaluation_settings.get('output_dir', 'results/')
    os.makedirs(results_output_dir, exist_ok=True)
    results_file_name = f"experiment_results_{config.experiment_name}_{config.seed}.yaml"
    results_file_path = os.path.join(results_output_dir, results_file_name)
    with open(results_file_path, 'w', encoding='utf-8') as f:
        yaml.dump(all_results, f, default_flow_style=False)
    print(f"\nAll results saved to {results_file_path}")

    print("\n--- Universal Neural Operators Reproduction Finished Successfully ---")


if __name__ == "__main__":
    main()


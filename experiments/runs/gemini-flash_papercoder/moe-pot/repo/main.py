# main.py
import argparse
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Ensure config and other modules are importable
# Adjust PATH if necessary for relative imports in project structure
# For example, if main.py is in the root, and others in subfolders like model/, data/
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config # Import the Config class
from data.datamodule import PDEDataModule
from evaluator import Evaluator
from model.moepot import MoEPOT
from trainer import Trainer
from utils import cleanup_distributed_training, set_seed, setup_distributed_training


class ExperimentRunner:
    """
    Orchestrates the entire experiment lifecycle: setup, pre-training, fine-tuning,
    downstream tasks, ablation studies, and interpretability analysis.
    """
    def __init__(self, config: Config, rank: int, world_size: int):
        """
        Initializes the ExperimentRunner.

        Args:
            config: The global configuration object.
            rank: The current process rank in distributed training.
            world_size: The total number of processes participating in distributed training.
        """
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{self.rank}' if torch.cuda.is_available() else 'cpu')

        self._set_seed(self.config.seed + self.rank)

        # Create output directories (only on rank 0)
        if self.rank == 0:
            os.makedirs(self.config.output_dir, exist_ok=True)
            self.experiment_dir = os.path.join(self.config.output_dir, self.config.experiment_name)
            os.makedirs(self.experiment_dir, exist_ok=True)
            os.makedirs(os.path.join(self.experiment_dir, self.config.checkpoint_dir), exist_ok=True)
            os.makedirs(os.path.join(self.experiment_dir, self.config.log_dir), exist_ok=True)
            print(f"Experiment outputs will be saved to: {self.experiment_dir}")
        
        # Ensure all ranks have created experiment_dir before proceeding to avoid issues
        if self.world_size > 1:
            dist.barrier()


    def _set_seed(self, seed: int) -> None:
        """Sets random seeds for reproducibility for the current worker."""
        set_seed(seed)
        if self.rank == 0:
            print(f"Random seed set to {seed} for rank {self.rank}")

    def _determine_channels_and_update_config(self, is_pretraining: bool, dataset_name: Optional[str] = None) -> None:
        """
        Determines the dynamic input and output channel counts based on the datasets
        and updates the global configuration. Handles broadcasting in DDP.
        """
        if self.rank == 0:
            # Rank 0 performs the channel determination
            temp_datamodule = PDEDataModule(
                config=self.config,
                is_pretraining=is_pretraining,
                dataset_name=dataset_name
            )
            # Setup will populate max_total_channels and set model.input_channels/output_channels
            temp_datamodule.setup(rollout_eval_length=self.config.model.T_in) # Use T_in for channel determination as a default rollout length
            
            self.config.set_dynamic_channels(
                temp_datamodule.input_channels,
                temp_datamodule.output_channels
            )
            self.config.data.current_dataset_type_map = temp_datamodule.dataset_type_map

            print(f"Determined input/output channels: {self.config.model.input_channels}")
            
            # Prepare data to broadcast
            channels_data = torch.tensor([
                self.config.model.input_channels,
                self.config.model.output_channels
            ], dtype=torch.int32, device=self.device)
            
            # Serialize dataset_type_map
            import json
            map_str = json.dumps(temp_datamodule.dataset_type_map)
            map_bytes = map_str.encode('utf-8')
            map_len = torch.tensor([len(map_bytes)], dtype=torch.int32, device=self.device)

        else:
            # Other ranks initialize dummy tensors
            channels_data = torch.zeros(2, dtype=torch.int32, device=self.device)
            map_len = torch.zeros(1, dtype=torch.int32, device=self.device)

        # Broadcast channel data
        if self.world_size > 1:
            dist.broadcast(channels_data, src=0)
            dist.broadcast(map_len, src=0)
        
        # All ranks update their config
        self.config.set_dynamic_channels(channels_data[0].item(), channels_data[1].item())
        
        # All ranks deserialize dataset_type_map
        if self.rank != 0:
            map_bytes = bytearray(map_len.item())
            if self.world_size > 1:
                dist.broadcast(torch.ByteTensor(map_bytes), src=0)
            map_str = map_bytes.decode('utf-8')
            self.config.data.current_dataset_type_map = json.loads(map_str)

        if self.world_size > 1:
            dist.barrier() # Ensure all ranks have updated their config before proceeding


    def run_pretraining(self) -> str:
        """
        Executes the pre-training phase.

        Returns:
            The path to the best saved model checkpoint.
        """
        if self.rank == 0:
            print("\n--- Starting Pre-training ---")
        
        self.config.update_for_experiment_type("pretrain")
        self._determine_channels_and_update_config(is_pretraining=True)

        datamodule = PDEDataModule(
            config=self.config,
            is_pretraining=True
        )
        datamodule.setup(rollout_eval_length=self.config.model.T_in) # For validation within trainer

        model = MoEPOT(self.config)
        trainer = Trainer(model, datamodule, self.config, self.rank, self.world_size, is_pretraining=True)
        
        best_model_path = trainer.train()
        
        if self.rank == 0:
            print("--- Pre-training Complete ---")
        return best_model_path

    def run_finetuning(self, pretrained_model_path: str, dataset_name: str) -> None:
        """
        Executes the fine-tuning phase on a specific dataset.

        Args:
            pretrained_model_path: Path to the pre-trained model checkpoint.
            dataset_name: The name of the dataset for fine-tuning.
        """
        if self.rank == 0:
            print(f"\n--- Starting Fine-tuning on {dataset_name} ---")

        self.config.update_for_experiment_type("finetune", dataset_name=dataset_name)
        self._determine_channels_and_update_config(is_pretraining=False, dataset_name=dataset_name)

        datamodule = PDEDataModule(
            config=self.config,
            is_pretraining=False,
            dataset_name=dataset_name
        )
        datamodule.setup(rollout_eval_length=self.config.model.T_in) # For validation within trainer

        model = MoEPOT(self.config)
        
        # Load pre-trained weights
        checkpoint = torch.load(pretrained_model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        if self.rank == 0:
            print(f"Loaded pretrained weights from {pretrained_model_path}")

        trainer = Trainer(model, datamodule, self.config, self.rank, self.world_size, is_pretraining=False)
        
        trainer.train()

        # Evaluate the fine-tuned model on the test set after training
        if self.rank == 0:
            print(f"--- Evaluating Fine-tuned Model on {dataset_name} Test Set ---")
            evaluator = Evaluator(model, self.config, self.rank, self.world_size)
            test_dataloader = datamodule.test_dataloader(self.rank, self.world_size, rollout_eval_length=self.config.model.T_in)
            results = evaluator.evaluate_model({'test_set': test_dataloader}, rollout_steps=self.config.model.T_in)
            print(f"Fine-tuned {dataset_name} Test L2RE: {results['test_set']:.4f}")
            print(f"--- Fine-tuning on {dataset_name} Complete ---")

    def run_downstream_task(self, pretrained_model_path: Optional[str], dataset_name: str, train_from_scratch: bool) -> None:
        """
        Executes a downstream task, either training from scratch or fine-tuning from a pre-trained model.

        Args:
            pretrained_model_path: Path to the pre-trained model checkpoint, or None if training from scratch.
            dataset_name: The name of the dataset for the downstream task.
            train_from_scratch: If True, train from scratch; otherwise, fine-tune.
        """
        if self.rank == 0:
            mode = "scratch" if train_from_scratch else "fine-tuning"
            print(f"\n--- Starting Downstream Task on {dataset_name} ({mode}) ---")
        
        self.config.update_for_experiment_type("downstream", dataset_name=dataset_name)
        self._determine_channels_and_update_config(is_pretraining=False, dataset_name=dataset_name)

        datamodule = PDEDataModule(
            config=self.config,
            is_pretraining=False,
            dataset_name=dataset_name
        )
        datamodule.setup(rollout_eval_length=self.config.model.T_in) # For validation within trainer

        model = MoEPOT(self.config)
        
        if not train_from_scratch:
            if not pretrained_model_path:
                raise ValueError("pretrained_model_path must be provided for downstream task fine-tuning.")
            checkpoint = torch.load(pretrained_model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            if self.rank == 0:
                print(f"Loaded pretrained weights from {pretrained_model_path}")
        else:
            if self.rank == 0:
                print("Training downstream model from scratch.")

        trainer = Trainer(model, datamodule, self.config, self.rank, self.world_size, is_pretraining=False)
        
        trainer.train()

        # Evaluate the trained model on the test set
        if self.rank == 0:
            print(f"--- Evaluating Downstream Model on {dataset_name} Test Set ---")
            evaluator = Evaluator(model, self.config, self.rank, self.world_size)
            test_dataloader = datamodule.test_dataloader(self.rank, self.world_size, rollout_eval_length=self.config.model.T_in)
            results = evaluator.evaluate_model({'test_set': test_dataloader}, rollout_steps=self.config.model.T_in)
            print(f"Downstream {dataset_name} Test L2RE: {results['test_set']:.4f}")
            print(f"--- Downstream Task on {dataset_name} Complete ---")

    def run_ablation_study(self, ablation_type: str, ablation_values: List[Any]) -> Dict[Any, Dict[str, float]]:
        """
        Conducts an ablation study by varying a specific hyperparameter.

        Args:
            ablation_type: The name of the hyperparameter to ablate (e.g., 'model.num_routed_experts').
            ablation_values: A list of values to test for the given hyperparameter.

        Returns:
            A dictionary containing evaluation results for each ablation value.
        """
        if self.rank == 0:
            print(f"\n--- Starting Ablation Study for {ablation_type} ---")
            print(f"Ablation values: {ablation_values}")

        ablation_results: Dict[Any, Dict[str, float]] = {}

        for value in ablation_values:
            # Each ablation run needs a clean config and environment
            temp_config = Config(config_path=self.config.config_path, cmd_args=None) # Start with base config
            
            # Dynamically set the ablation parameter in the temporary config
            keys = ablation_type.split('.')
            current_level = temp_config
            for i, key in enumerate(keys):
                if i == len(keys) - 1:
                    current_level[key] = value
                else:
                    current_level = current_level[key]
            
            if self.rank == 0:
                print(f"\n--- Running ablation for {ablation_type}={value} ---")
            
            # Update model specific parameters if ablation_type is model.size
            if ablation_type == 'model.size':
                temp_config.update_for_model_size()
            
            # Execute a full pre-training run with the modified config
            # Create a dedicated runner for this ablation step to isolate effects
            ablation_runner = ExperimentRunner(temp_config, self.rank, self.world_size)
            ablation_runner.config.experiment_name = f"{self.config.experiment_name}_ablation_{ablation_type.replace('.', '_')}_{str(value)}"
            
            best_model_path_ablation = ablation_runner.run_pretraining()

            # Evaluate the best model from this ablation run
            if self.rank == 0:
                print(f"--- Evaluating Zero-shot Performance for {ablation_type}={value} ---")
                
                # Load the best model to evaluate
                eval_model = MoEPOT(temp_config)
                checkpoint = torch.load(best_model_path_ablation, map_location='cpu')
                eval_model.load_state_dict(checkpoint['model_state_dict'])

                # Prepare data for evaluation (all pre-training datasets)
                eval_datamodule = PDEDataModule(
                    config=temp_config,
                    is_pretraining=True # Use pre-training datasets for evaluation context
                )
                eval_datamodule.setup(rollout_eval_length=temp_config.model.T_in)
                eval_dataloaders = {
                    ds_name: eval_datamodule.test_dataloader(self.rank, self.world_size, rollout_eval_length=temp_config.model.T_in)
                    for ds_name in temp_config.data.pretrain_data_info.keys()
                }

                evaluator = Evaluator(eval_model, temp_config, self.rank, self.world_size)
                eval_results = evaluator.evaluate_model(eval_dataloaders, rollout_steps=temp_config.model.T_in)
                
                ablation_results[value] = eval_results
                print(f"Results for {ablation_type}={value}: {eval_results}")
            
            # Ensure all processes sync before next ablation value
            if self.world_size > 1:
                dist.barrier()
        
        if self.rank == 0:
            print("\n--- Ablation Study Complete ---")
            for value, results in ablation_results.items():
                print(f"Summary for {ablation_type}={value}: {results}")

        return ablation_results

    def run_interpretability_analysis(self, pretrained_model_path: str) -> None:
        """
        Performs interpretability analysis on the router-gating network.

        Args:
            pretrained_model_path: Path to the pre-trained model checkpoint.
        """
        if self.rank == 0:
            print("\n--- Starting Interpretability Analysis ---")
        
        # We need the pre-training datasets for interpretability
        self.config.update_for_experiment_type("pretrain") # This populates current_data_info
        self._determine_channels_and_update_config(is_pretraining=True)

        datamodule = PDEDataModule(
            config=self.config,
            is_pretraining=True
        )
        datamodule.setup(rollout_eval_length=self.config.model.T_in) # Only need to pass for consistency, actual eval length not used in this analysis

        model = MoEPOT(self.config)
        
        # Load pre-trained weights
        checkpoint = torch.load(pretrained_model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        if self.rank == 0:
            print(f"Loaded pretrained weights from {pretrained_model_path}")

        evaluator = Evaluator(model, self.config, self.rank, self.world_size)
        interpretability_results = evaluator.run_interpretability(datamodule)
        
        if self.rank == 0:
            print(f"Interpretability Analysis Results: {interpretability_results}")
            print("--- Interpretability Analysis Complete ---")


def parse_main_args() -> argparse.Namespace:
    """
    Parses command-line arguments for main.py execution flow.
    Includes flags for different experiment types and general config overrides.
    """
    parser = argparse.ArgumentParser(description="MoE-POT Reproduction Main Script")

    parser.add_argument("--config_path", type=str, default="config.yaml",
                        help="Path to the main YAML configuration file.")
    
    # Experiment flags
    parser.add_argument("--run_pretraining", action="store_true",
                        help="Run the pre-training phase.")
    parser.add_argument("--run_finetuning", action="store_true",
                        help="Run the fine-tuning phase.")
    parser.add_argument("--run_downstream", action="store_true",
                        help="Run a downstream task.")
    parser.add_argument("--run_ablation", action="store_true",
                        help="Run an ablation study.")
    parser.add_argument("--run_interpretability", action="store_true",
                        help="Run interpretability analysis.")

    # Experiment specific arguments
    parser.add_argument("--finetune_dataset", type=str, default=None,
                        help="Name of the dataset for fine-tuning (e.g., 'FNO_1e-5').")
    parser.add_argument("--downstream_dataset", type=str, default=None,
                        help="Name of the dataset for downstream task (e.g., 'NS_1e-4').")
    parser.add_argument("--downstream_from_scratch", action="store_true",
                        help="For downstream task, train from scratch instead of fine-tuning.")
    parser.add_argument("--ablation_type", type=str, default=None,
                        help="Type of hyperparameter for ablation (e.g., 'model.num_routed_experts', 'model.top_k').")
    parser.add_argument("--ablation_values", type=str, default=None,
                        help="Comma-separated values for ablation study (e.g., '8,16,32' or '1,2,4').")
    parser.add_argument("--pretrained_model_path", type=str, default=None,
                        help="Path to a pre-trained model checkpoint for fine-tuning, downstream, or interpretability.")

    # Allow arbitrary config overrides using the format --section.param_name value
    # This collects all arguments not explicitly parsed above.
    args, unknown_args = parser.parse_known_args()

    # Create a separate Namespace for config overrides
    config_overrides = argparse.Namespace()
    for arg_str in unknown_args:
        if '=' in arg_str:
            key, val = arg_str.split('=', 1)
        else: # Handle boolean flags or single-arg overrides
            key = arg_str
            val = 'true' # Assume it's a boolean flag being set to true if no value

        key = key.lstrip('-') # Remove leading dashes
        # Attempt to convert to appropriate type
        if val.lower() == 'true':
            val = True
        elif val.lower() == 'false':
            val = False
        else:
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    pass # Keep as string

        setattr(config_overrides, key, val)
    
    args.config_overrides = config_overrides
    return args


def main_worker(rank: int, world_size: int, config_path: str, cmd_args_namespace: argparse.Namespace) -> None:
    """
    The main function executed by each distributed process.

    Args:
        rank: The unique identifier for the current process within the distributed group.
        world_size: The total number of processes participating in distributed training.
        config_path: Path to the main YAML configuration file.
        cmd_args_namespace: An argparse.Namespace object containing command-line arguments.
    """
    # Each worker re-loads config and applies overrides to ensure isolated state
    # Need to convert cmd_args_namespace.config_overrides back to a Namespace for Config constructor
    config = Config(config_path=config_path, cmd_args=cmd_args_namespace.config_overrides)
    
    # Initialize distributed training environment
    if world_size > 1:
        setup_distributed_training(rank, world_size, backend=config.distributed.backend)
    
    # Create the experiment runner
    runner = ExperimentRunner(config, rank, world_size)

    # Dispatch tasks based on command-line flags
    if cmd_args_namespace.run_pretraining:
        runner.run_pretraining()
    
    if cmd_args_namespace.run_finetuning:
        if not cmd_args_namespace.finetune_dataset:
            raise ValueError("Finete_dataset must be specified when --run_finetuning is used.")
        if not cmd_args_namespace.pretrained_model_path:
            raise ValueError("pretrained_model_path must be specified for fine-tuning.")
        runner.run_finetuning(cmd_args_namespace.pretrained_model_path, cmd_args_namespace.finetune_dataset)

    if cmd_args_namespace.run_downstream:
        if not cmd_args_namespace.downstream_dataset:
            raise ValueError("Downstream_dataset must be specified when --run_downstream is used.")
        
        pretrained_path = cmd_args_namespace.pretrained_model_path if not cmd_args_namespace.downstream_from_scratch else None
        if not cmd_args_namespace.downstream_from_scratch and not pretrained_path:
             raise ValueError("pretrained_model_path must be specified for downstream task fine-tuning.")

        runner.run_downstream_task(
            pretrained_path,
            cmd_args_namespace.downstream_dataset,
            cmd_args_namespace.downstream_from_scratch
        )

    if cmd_args_namespace.run_ablation:
        if not cmd_args_namespace.ablation_type or not cmd_args_namespace.ablation_values:
            raise ValueError("ablation_type and ablation_values must be specified for ablation study.")
        
        # Parse ablation_values (e.g., '8,16,32' -> [8, 16, 32])
        values_str = cmd_args_namespace.ablation_values.split(',')
        # Attempt to convert to int, then float, otherwise keep as string
        ablation_values_list = []
        for v_str in values_str:
            try:
                ablation_values_list.append(int(v_str))
            except ValueError:
                try:
                    ablation_values_list.append(float(v_str))
                except ValueError:
                    ablation_values_list.append(v_str)

        runner.run_ablation_study(cmd_args_namespace.ablation_type, ablation_values_list)

    if cmd_args_namespace.run_interpretability:
        if not cmd_args_namespace.pretrained_model_path:
            raise ValueError("pretrained_model_path must be specified for interpretability analysis.")
        runner.run_interpretability_analysis(cmd_args_namespace.pretrained_model_path)


    if world_size > 1:
        cleanup_distributed_training()


def main():
    """
    Main entry point for the MoE-POT reproduction project.
    Parses arguments, sets up distributed training, and launches the main worker function.
    """
    cmd_args = parse_main_args()
    
    # Initial config load to get distributed settings
    # We pass the full cmd_args_namespace.config_overrides to Config for robust override.
    base_config = Config(config_path=cmd_args.config_path, cmd_args=cmd_args.config_overrides)
    
    world_size = base_config.distributed.world_size

    # Check for required arguments if not pre-training
    if not cmd_args.run_pretraining:
        if cmd_args.run_finetuning or cmd_args.run_downstream or cmd_args.run_interpretability:
            if not cmd_args.pretrained_model_path:
                raise ValueError("pretrained_model_path must be provided when running fine-tuning, downstream tasks, or interpretability without pre-training.")
            if not os.path.exists(cmd_args.pretrained_model_path):
                raise FileNotFoundError(f"Pre-trained model path not found: {cmd_args.pretrained_model_path}")

    # Launch distributed processes or run in single-process mode
    if world_size > 1:
        print(f"Launching {world_size} distributed processes...")
        # mp.spawn requires passing arguments to main_worker as a tuple
        mp.spawn(main_worker,
                 args=(world_size, cmd_args.config_path, cmd_args),
                 nprocs=world_size,
                 join=True)
    else:
        print("Running in single-process mode...")
        main_worker(0, 1, cmd_args.config_path, cmd_args)

    print("\n--- All experiments finished ---")


if __name__ == "__main__":
    main()


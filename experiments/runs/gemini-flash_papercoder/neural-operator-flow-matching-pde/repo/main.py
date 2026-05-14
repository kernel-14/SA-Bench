import argparse
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, Any, Tuple

# Local imports
from config import Config
from data.dataset_processor import DatasetProcessor
from data.pde_dataset import PDEDataset
from models.p2vae import P2VAEModel
from models.fmt import FMTModel
from training.p2vae_trainer import P2VAETrainer
from training.fmt_trainer import FMTTrainer
from evaluation.evaluator import Evaluator
from utils import logging_utils # Assuming logging_utils contains setup_logging, log_message, close_writers

class Main:
    """
    The main entry point for the Generative PDE Foundation Model project.
    It orchestrates configuration loading, dataset preparation, model instantiation,
    training execution, and evaluation based on command-line arguments.
    """

    def __init__(self):
        """
        Initializes the Main class, parses command-line arguments, loads the configuration,
        and sets up logging.
        """
        parser = argparse.ArgumentParser(description="Generative PDE Foundation Model Training and Evaluation")
        parser.add_argument("--config_path", type=str, default="configs/default.yaml",
                            help="Path to the YAML configuration file.")
        parser.add_argument("--stage", type=str, required=True,
                            choices=["train_p2vae", "train_fmt", "evaluate", "finetune"],
                            help="The stage of the experiment to run (train_p2vae, train_fmt, evaluate, finetune).")
        parser.add_argument("--p2vae_checkpoint", type=str, default=None,
                            help="Path to a pre-trained P2VAE checkpoint for loading. If not provided, looks for a default 'p2vae_best_model.pth' in checkpoint_dir.")
        parser.add_argument("--fmt_checkpoint", type=str, default=None,
                            help="Path to a pre-trained FMT checkpoint for loading. If not provided, looks for a default 'fmt_best_model.pth' in checkpoint_dir.")
        parser.add_argument("--local_rank", type=int, default=-1,
                            help="Local rank for distributed training. Set automatically by torch.distributed.launch.")

        self.args = parser.parse_args()

        # 1. Load Configuration
        self.config = Config(self.args.config_path)
        self.config.load_config()

        # 2. Setup Device and Distributed Training
        self.device, self.is_distributed, self.global_rank = self._setup_device()
        self.dtype: torch.dtype = getattr(torch, self.config.get("global.dtype", "float16"))

        # 3. Setup Logging (per process if distributed, but aggregate if main)
        self.logger, self.tb_writer = logging_utils.setup_logging(self.config, self.args.stage, self.global_rank)
        
        # 4. Initialize Models and Data Loaders
        self.p2vae_model: P2VAEModel = None
        self.fmt_model: FMTModel = None
        self.train_loader: DataLoader = None
        self.val_loader: DataLoader = None
        self.test_loader: DataLoader = None
        
        logging_utils.log_message(self.logger, f"Initialized Main for stage: {self.args.stage} (Rank {self.global_rank})", level=logging_utils.logger.info)
        logging_utils.log_message(self.logger, f"Using device: {self.device}, dtype: {self.dtype}, Distributed: {self.is_distributed}", level=logging_utils.logger.info)

    def _setup_device(self) -> Tuple[torch.device, bool, int]:
        """
        Sets up the device (CPU/GPU) and distributed environment if applicable.

        Returns:
            Tuple[torch.device, bool, int]: (device, is_distributed, global_rank)
        """
        is_distributed: bool = self.args.local_rank != -1
        global_rank: int = 0

        if is_distributed:
            torch.cuda.set_device(self.args.local_rank)
            dist.init_process_group(backend="nccl", init_method="env://")
            global_rank = dist.get_rank()
            device = torch.device(f"cuda:{self.args.local_rank}")
        elif self.config.get("global.device", "cuda") == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        return device, is_distributed, global_rank

    def _load_model_checkpoint(self, model: torch.nn.Module, checkpoint_path: str, model_name: str, map_location: torch.device) -> torch.nn.Module:
        """
        Loads model state_dict from a checkpoint path.

        Args:
            model (torch.nn.Module): The model instance to load state into.
            checkpoint_path (str): Path to the checkpoint file.
            model_name (str): Name of the model for logging purposes.
            map_location (torch.device): Device to map the loaded state_dict to.

        Returns:
            torch.nn.Module: The model with loaded state_dict.
        """
        if checkpoint_path and os.path.exists(checkpoint_path):
            logging_utils.log_message(self.logger, f"Loading {model_name} from checkpoint: {checkpoint_path}", level=logging_utils.logger.info)
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
            
            # Remove 'module.' prefix if the model was saved from a DDP-wrapped model
            state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
            model.load_state_dict(state_dict)

            logging_utils.log_message(self.logger, f"{model_name} loaded successfully.", level=logging_utils.logger.info)
        else:
            logging_utils.log_message(self.logger, f"No valid checkpoint found for {model_name} at: {checkpoint_path}. Starting from scratch.", level=logging_utils.logger.warning)
        return model

    def _prepare_data_loaders(self, include_test: bool = True):
        """
        Prepares and returns train, validation, and optionally test DataLoaders.
        """
        if self.global_rank == 0: # Only rank 0 processes data preparation and logs it
            logging_utils.log_message(self.logger, "Starting dataset preparation...", level=logging_utils.logger.info)
        
        dataset_processor = DatasetProcessor(self.config)
        processed_datasets: Dict[str, PDEDataset] = dataset_processor.prepare_datasets() # Returns Dict[str, PDEDataset]
        
        train_ds = processed_datasets["train"]
        val_ds = processed_datasets["validation"]
        test_ds = processed_datasets["test"] if include_test else None

        # Determine batch size based on current stage
        if self.args.stage == "train_p2vae":
            batch_size = self.config.get("p2vae_training.batch_size", 256)
        elif self.args.stage == "train_fmt":
            batch_size = self.config.get("fmt_training.batch_size", 256)
        else: # Evaluate or finetune
            batch_size = self.config.get("evaluation.batch_size", 16) # Default for eval, adjust if needed

        num_gpus = self.config.get("global.num_gpus", 1) # Assumes config.num_gpus is total GPUs requested
        
        train_sampler = DistributedSampler(train_ds, num_replicas=num_gpus, rank=self.global_rank, shuffle=True) if self.is_distributed else None
        val_sampler = DistributedSampler(val_ds, num_replicas=num_gpus, rank=self.global_rank, shuffle=False) if self.is_distributed else None
        test_sampler = DistributedSampler(test_ds, num_replicas=num_gpus, rank=self.global_rank, shuffle=False) if self.is_distributed and include_test else None

        # Use effective batch size per GPU if distributed
        batch_size_per_gpu = batch_size // num_gpus if self.is_distributed else batch_size

        num_workers = min(os.cpu_count(), 8) # Cap workers to 8 to avoid over-utilization
        if self.is_distributed:
            num_workers = min(os.cpu_count() // num_gpus, 8) if num_gpus > 0 else 0
        
        self.train_loader = DataLoader(
            train_ds,
            batch_size=batch_size_per_gpu,
            shuffle=(train_sampler is None), # Only shuffle if not using DistributedSampler
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=batch_size_per_gpu,
            shuffle=False,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True
        )
        if include_test and test_ds is not None:
            self.test_loader = DataLoader(
                test_ds,
                batch_size=batch_size_per_gpu,
                shuffle=False,
                sampler=test_sampler,
                num_workers=num_workers,
                pin_memory=True
            )
        
        if self.global_rank == 0:
            logging_utils.log_message(self.logger, "Dataset preparation complete.", level=logging_utils.logger.info)

    def run(self):
        """
        Orchestrates the entire execution flow based on the specified stage.
        """
        self._prepare_data_loaders(include_test=(self.args.stage in ["evaluate", "finetune"]))

        checkpoint_dir = self.config.get('logging.checkpoint_dir', './checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        if self.args.stage == "train_p2vae":
            if self.global_rank == 0:
                logging_utils.log_message(self.logger, "Starting P2VAE training stage.", level=logging_utils.logger.info)

            self.p2vae_model = P2VAEModel(self.config).to(self.device, dtype=self.dtype)
            self.p2vae_model = self._load_model_checkpoint(
                self.p2vae_model, 
                self.args.p2vae_checkpoint, 
                "P2VAE", 
                self.device
            )

            if self.is_distributed:
                self.p2vae_model = DDP(self.p2vae_model, device_ids=[self.args.local_rank], find_unused_parameters=False) # find_unused_parameters=False for efficiency unless there are specific reasons

            trainer = P2VAETrainer(
                model=self.p2vae_model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                config=self.config,
                device=self.device,
                logger=self.logger,
                tb_writer=self.tb_writer
            )
            trainer.train() # Trainer saves final and best models
            
            if self.global_rank == 0:
                logging_utils.log_message(self.logger, "P2VAE training complete.", level=logging_utils.logger.info)

        elif self.args.stage == "train_fmt":
            if self.global_rank == 0:
                logging_utils.log_message(self.logger, "Starting FMT training stage.", level=logging_utils.logger.info)

            # 1. Load P2VAE (frozen)
            self.p2vae_model = P2VAEModel(self.config).to(self.device, dtype=self.dtype)
            p2vae_ckpt_path = self.args.p2vae_checkpoint or os.path.join(checkpoint_dir, "p2vae_best_model.pth")
            self.p2vae_model = self._load_model_checkpoint(self.p2vae_model, p2vae_ckpt_path, "P2VAE", self.device)
            self.p2vae_model.eval() # Ensure P2VAE is in eval mode
            for param in self.p2vae_model.parameters():
                param.requires_grad = False # Freeze P2VAE parameters

            # 2. Initialize FMT
            self.fmt_model = FMTModel(self.config).to(self.device, dtype=self.dtype)
            fmt_ckpt_path = self.args.fmt_checkpoint or os.path.join(checkpoint_dir, "fmt_best_model.pth")
            self.fmt_model = self._load_model_checkpoint(self.fmt_model, fmt_ckpt_path, "FMT", self.device)

            if self.is_distributed:
                self.fmt_model = DDP(self.fmt_model, device_ids=[self.args.local_rank], find_unused_parameters=True) # find_unused_parameters=True due to GRU/AdaLN-Zero possibly not being used in all paths

            trainer = FMTTrainer(
                model=self.fmt_model,
                p2vae_model=self.p2vae_model, # Frozen P2VAE
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                config=self.config,
                device=self.device,
                logger=self.logger,
                tb_writer=self.tb_writer
            )
            trainer.train() # Trainer saves final and best models

            if self.global_rank == 0:
                logging_utils.log_message(self.logger, "FMT training complete.", level=logging_utils.logger.info)

        elif self.args.stage == "evaluate":
            if self.global_rank == 0:
                logging_utils.log_message(self.logger, "Starting evaluation stage.", level=logging_utils.logger.info)
            if self.test_loader is None:
                raise ValueError("Test data loader must be prepared for evaluation stage.")

            # Load P2VAE
            self.p2vae_model = P2VAEModel(self.config).to(self.device, dtype=self.dtype)
            p2vae_ckpt_path = self.args.p2vae_checkpoint or os.path.join(checkpoint_dir, "p2vae_best_model.pth")
            self.p2vae_model = self._load_model_checkpoint(self.p2vae_model, p2vae_ckpt_path, "P2VAE", self.device)
            self.p2vae_model.eval()

            # Load FMT
            self.fmt_model = FMTModel(self.config).to(self.device, dtype=self.dtype)
            fmt_ckpt_path = self.args.fmt_checkpoint or os.path.join(checkpoint_dir, "fmt_best_model.pth")
            self.fmt_model = self._load_model_checkpoint(self.fmt_model, fmt_ckpt_path, "FMT", self.device)
            self.fmt_model.eval()

            evaluator = Evaluator(
                p2vae_model=self.p2vae_model,
                fmt_model=self.fmt_model,
                test_loader=self.test_loader,
                config=self.config,
                device=self.device,
                logger=self.logger,
                tb_writer=self.tb_writer # Pass writer for image logging
            )

            if self.global_rank == 0:
                reco_metrics = evaluator.evaluate_reconstruction()
                logging_utils.log_message(self.logger, f"P2VAE Reconstruction Metrics: {reco_metrics}", level=logging_utils.logger.info)

                rollout_metrics = evaluator.evaluate_long_term_rollout()
                logging_utils.log_message(self.logger, f"FMT Long-Term Rollout Metrics: {rollout_metrics}", level=logging_utils.logger.info)

                # Example: generate ensemble for a single sample from test set
                # (This requires careful selection of a sample with sufficient context)
                # This part is highly dependent on how the test_loader yields full trajectories.
                # Assuming the test_loader provides samples suitable for `generate_ensemble`.
                # For simplicity, we'll demonstrate a conceptual call.
                # In a real scenario, this might pick a specific `initial_states_batch` from the test set.
                try:
                    first_batch = next(iter(self.test_loader))
                    # Assuming 'x_0', 'x_1', 'x_2', 'x_3' are separate states
                    # Or a combined 'trajectory' field. Let's assume the latter for simplicity as it's cleaner.
                    # if first_batch has {'x_0':..., 'x_1':..., ... 'x_3':...}
                    initial_states_combined = torch.stack([
                        first_batch['x_0'], first_batch['x_1'], first_batch['x_2'], first_batch['x_3']
                    ], dim=1) # (B, T, C, H, W)
                    
                    num_generations = self.config.get("evaluation.ensemble_generation.num_samples_per_scenario", 32)
                    k_values_to_test = self.config.get("evaluation.ensemble_generation.k_values_to_test", [0.0])

                    for k_val in k_values_to_test:
                        logging_utils.log_message(self.logger, f"Generating ensemble for k_val={k_val}...", level=logging_utils.logger.info)
                        # We use only the last `trajectory_length` states from `initial_states_combined`
                        # for ensemble generation as per the method description.
                        generated_ensembles_batch = evaluator.generate_ensemble(
                            initial_states_batch=initial_states_combined, # Pass the batch
                            num_generations=num_generations,
                            k_val_last_step=k_val
                        )
                        # generated_ensembles_batch is List[Tensor] where each Tensor is (num_generations, C, H, W)
                        logging_utils.log_message(self.logger, f"Generated {len(generated_ensembles_batch)} ensemble batches, each with {num_generations} samples for k={k_val}.", level=logging_utils.logger.info)
                except Exception as e:
                    logging_utils.log_message(self.logger, f"Failed to generate ensemble during evaluation: {e}", level=logging_utils.logger.warning)
                    logging_utils.log_message(self.logger, "Ensure test_loader provides suitable trajectories for ensemble generation example.", level=logging_utils.logger.warning)
            
            if self.is_distributed:
                dist.barrier() # Ensure all processes complete before cleanup
            
            if self.global_rank == 0:
                logging_utils.log_message(self.logger, "Evaluation complete.", level=logging_utils.logger.info)

        elif self.args.stage == "finetune":
            if self.global_rank == 0:
                logging_utils.log_message(self.logger, "Starting fine-tuning stage.", level=logging_utils.logger.info)

            if not self.config.get("finetuning.enabled", False):
                raise ValueError("Fine-tuning stage chosen, but 'finetuning.enabled' is False in config.")

            # Load P2VAE
            self.p2vae_model = P2VAEModel(self.config).to(self.device, dtype=self.dtype)
            p2vae_ckpt_path = self.args.p2vae_checkpoint or os.path.join(checkpoint_dir, "p2vae_best_model.pth")
            self.p2vae_model = self._load_model_checkpoint(self.p2vae_model, p2vae_ckpt_path, "P2VAE", self.device)
            
            # Load FMT
            self.fmt_model = FMTModel(self.config).to(self.device, dtype=self.dtype)
            fmt_ckpt_path = self.args.fmt_checkpoint or os.path.join(checkpoint_dir, "fmt_best_model.pth")
            self.fmt_model = self._load_model_checkpoint(self.fmt_model, fmt_ckpt_path, "FMT", self.device)

            # In DDP mode, the models are wrapped internally by the Evaluator's finetune_and_evaluate.
            # Here, we pass the base models to the evaluator.

            # Prepare specific fine-tuning dataset and target dataset for evaluation after finetuning.
            # This is a placeholder. In a real scenario, this would involve loading specific data
            # for the downstream task (e.g., Kolmogorov turbulence)
            finetune_data_loader = self.train_loader # Using main train loader as placeholder
            target_data_loader = self.test_loader # Using main test loader as placeholder

            if self.global_rank == 0:
                logging_utils.log_message(self.logger, f"Fine-tuning on placeholder data. For real use, configure finetuning.finetune_dataset_name.", level=logging_utils.logger.warning)


            evaluator = Evaluator(
                p2vae_model=self.p2vae_model, # Initial pre-trained P2VAE
                fmt_model=self.fmt_model,     # Initial pre-trained FMT
                test_loader=target_data_loader, # DataLoader for evaluation after finetuning
                config=self.config,
                device=self.device,
                logger=self.logger,
                tb_writer=self.tb_writer
            )
            
            if self.global_rank == 0:
                logging_utils.log_message(self.logger, "Calling finetune_and_evaluate...", level=logging_utils.logger.info)
            finetune_metrics = evaluator.finetune_and_evaluate(finetune_data_loader, target_data_loader)
            
            if self.global_rank == 0:
                logging_utils.log_message(self.logger, f"Fine-tuning Metrics: {finetune_metrics}", level=logging_utils.logger.info)
                logging_utils.log_message(self.logger, "Fine-tuning complete.", level=logging_utils.logger.info)

        else:
            logging_utils.log_message(self.logger, f"Unknown stage specified: {self.args.stage}", level=logging_utils.logger.error)
            raise ValueError(f"Unknown stage: {self.args.stage}")

        if self.is_distributed:
            dist.destroy_process_group()
            if self.global_rank == 0:
                logging_utils.log_message(self.logger, "Distributed process group destroyed.", level=logging_utils.logger.info)
        
        logging_utils.close_writers(self.tb_writer) # Close TensorBoard writer

if __name__ == "__main__":
    main_app = Main()
    main_app.run()


## Code: main.py
import argparse
import os
import random
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from typing import Dict, Any, List, Optional, Tuple

# Custom imports
from config import Config
from data.prompt_simulator import PromptSimulator
from data.sa1b_dataset import SA1BDataset
from data.sav_dataset import SAVDataset
from data.vos_dataset import VOSDataset
from model.sam2_model import SAM2Model
from training.trainer import Trainer
from evaluation.evaluator import Evaluator


class Main(object):
    """
    Main orchestrator for SAM 2 reproduction. Handles argument parsing,
    configuration loading, distributed training setup, component initialization,
    and experiment execution.
    """

    def __init__(self):
        """Initializes the Main class."""
        self.config: Optional[Config] = None
        self.device: Optional[torch.device] = None
        self.is_main_process: bool = True
        self.model: Optional[SAM2Model] = None
        self.trainer: Optional[Trainer] = None
        self.evaluator: Optional[Evaluator] = None
        self.train_loaders: Dict[str, DataLoader] = {}
        self.val_loaders: Dict[str, DataLoader] = {}
        self.test_loaders: Dict[str, DataLoader] = {}
        self.prompt_simulator: Optional[PromptSimulator] = None

    def parse_args(self) -> argparse.Namespace:
        """
        Parses command-line arguments.

        Returns:
            argparse.Namespace: Parsed arguments.
        """
        parser = argparse.ArgumentParser(description="SAM 2 Reproduction")
        parser.add_argument(
            "--config",
            type=str,
            default="config.yaml",
            help="Path to the YAML configuration file.",
        )
        parser.add_argument(
            "--mode",
            type=str,
            default="full_train",
            choices=[
                "pretrain",
                "full_train",
                "finetune",
                "eval_interactive_video",
                "eval_semisupervised_vos",
                "eval_image_segmentation",
            ],
            help="Operation mode: pretrain, full_train, finetune, or various evaluation modes.",
        )
        parser.add_argument(
            "--checkpoint",
            type=str,
            default=None,
            help="Path to a model checkpoint (.pth file) to load.",
        )
        # Arguments for distributed training
        parser.add_argument(
            "--local_rank",
            type=int,
            default=-1,
            help="Local rank for distributed training. Automatically set by torch.distributed.launch.",
        )
        # Generic config override argument
        parser.add_argument(
            "--set",
            nargs="+",
            default=[],
            help="Override configuration parameters. Format: key.path=value. E.g., --set training.full_train.learning_rate=0.001"
        )
        # Direct CLI arguments for common overrides (will be mapped in Config)
        parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
        parser.add_argument("--bs", type=int, default=None, help="Override batch size (primarily for image datasets).")
        parser.add_argument("--device", type=str, default=None, help="Override device ('cuda' or 'cpu').")
        parser.add_argument("--model_type", type=str, default=None, help="Override model.image_encoder.type.")
        parser.add_argument("--seq_len", type=int, default=None, help="Override training.full_train.seq_len.")
        parser.add_argument("--num_gpus", type=int, default=None, help="Override system.num_gpus.")

        args = parser.parse_args()
        return args

    def load_config(self, args: argparse.Namespace) -> None:
        """
        Loads the configuration from a YAML file and applies command-line overrides.

        Args:
            args (argparse.Namespace): Parsed command-line arguments.
        """
        self.config = Config(args.config, cli_args=args)

    def _setup_distributed(self, args: argparse.Namespace) -> None:
        """
        Initializes distributed training environment if multiple GPUs are configured.

        Args:
            args (argparse.Namespace): Parsed command-line arguments.
        """
        if self.config is None:
            raise RuntimeError("Config not loaded. Call load_config() first.")

        num_gpus: int = self.config.system.num_gpus
        
        if num_gpus > 1:
            if args.local_rank == -1:
                # If local_rank is -1, it means not running with torchrun/launch.
                # Attempt a simple single-node, multi-GPU initialization.
                # This is less robust than `torchrun` but can work for debugging/simpler setups.
                if dist.is_initialized():
                    # Already initialized (e.g., from an outer script), use existing rank
                    args.local_rank = dist.get_rank()
                    print(f"Distributed environment already initialized, using existing rank {args.local_rank}.")
                elif torch.cuda.is_available():
                    # Attempt manual initialization if CUDA is available
                    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
                    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', str(random.randint(10000, 20000)))
                    os.environ['WORLD_SIZE'] = str(num_gpus)
                    # When local_rank is not provided, typically it's 0 for the first process.
                    # We will explicitly assign it for a simple, non-robust multi-GPU local setup.
                    # For production, `torchrun` is highly recommended, which sets LOCAL_RANK.
                    # For this implementation, we will use args.local_rank as the device_id directly.
                    # A proper multi-GPU init loop would be in a launcher script.
                    args.local_rank = 0 # Default to first GPU for a single entry point if not set
                    print(f"Warning: num_gpus > 1 but --local_rank not provided. Attempting single-node, multi-GPU init "
                          f"with rank {args.local_rank}. For robust distributed training, use `torchrun`.")
                else:
                    print("Warning: num_gpus > 1 specified but CUDA not available. Running on CPU.")
                    self.config._raw_data['system']['num_gpus'] = 0 # Force to CPU mode
                    return
            
            # Update config with actual distributed ranks
            self.config._raw_data['system']['world_size'] = num_gpus
            self.config._raw_data['system']['rank'] = args.local_rank

            torch.cuda.set_device(args.local_rank)
            dist.init_process_group(
                backend=self.config.system.dist_backend,
                init_method=self.config.system.dist_url,
                world_size=self.config.system.world_size,
                rank=self.config.system.rank,
            )
            self.is_main_process = (self.config.system.rank == 0)
            print(f"Distributed training initialized: Rank {dist.get_rank()}/{dist.get_world_size()} on device {args.local_rank}")
        else:
            # Single GPU or CPU run
            self.is_main_process = True
            print("Running on a single GPU or CPU.")


    def _initialize_components(self, args: argparse.Namespace) -> None:
        """
        Initializes PromptSimulator, Datasets, SAM2Model, Trainer, and Evaluator.

        Args:
            args (argparse.Namespace): Parsed command-line arguments.
        """
        if self.config is None:
            raise RuntimeError("Config not loaded. Call load_config() first.")

        # Determine device
        if self.config.system.device == "cuda" and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{args.local_rank}" if args.local_rank != -1 else "cuda:0")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

        # 1. Initialize PromptSimulator
        self.prompt_simulator = PromptSimulator(self.config)

        # 2. Initialize Datasets and DataLoaders
        def _create_dataloader(
            dataset_cls: Any,
            dataset_name: str,
            split: str,
            is_video: bool,
            batch_size: int,
            shuffle: bool,
            # For finetuning, it sometimes needs specific filters like "most edited"
            is_finetune_data: bool = False
        ) -> DataLoader:
            dataset_instance = dataset_cls(self.config, split, self.prompt_simulator, is_finetune_data=is_finetune_data)
            
            sampler = None
            if self.config.system.num_gpus > 1:
                sampler = DistributedSampler(dataset_instance, num_replicas=self.config.system.world_size, rank=self.config.system.rank, shuffle=shuffle)
                shuffle = False # Sampler handles shuffling

            # Custom collate_fn for video data, ensuring lists are not stacked if batch_size=1
            # For SA1B (images), default collate will handle stacking `(C,H,W)` frames to `(B,C,H,W)`
            # and `(N_masks,1,H,W)` to `(B, N_masks,1,H,W)`.
            # For SAV/VOS (videos), `__getitem__` returns `List[Tensor]` for frames and masks.
            # If batch_size=1 (as configured for video_batch_size), the DataLoader will yield a list
            # containing one such dict. The collate_fn should just unwrap this.
            def custom_collate_fn(batch_list: List[Dict[str, Any]]) -> Dict[str, Any]:
                if is_video and len(batch_list) > 1:
                    raise NotImplementedError(f"Batching video lists (List[Tensor]) for dataset {dataset_name} is not supported. Ensure video_batch_size is 1.")
                
                # If batch_size is 1, just return the single item in the list
                if len(batch_list) == 1:
                    return batch_list[0]
                else: # For image data with batch_size > 1, use default collate
                    return torch.utils.data.dataloader.default_collate(batch_list)
            
            return DataLoader(
                dataset_instance,
                batch_size=batch_size,
                shuffle=shuffle,
                sampler=sampler,
                num_workers=self.config.system.num_workers,
                pin_memory=True, # Improves data transfer to GPU
                collate_fn=custom_collate_fn,
            )

        # Training DataLoaders
        if args.mode in ["pretrain", "full_train", "finetune"]:
            if self.config.training.pretrain.enabled and args.mode == "pretrain":
                sa1b_bs = self.config.training.pretrain.batch_size
                self.train_loaders["SA-1B"] = _create_dataloader(SA1BDataset, "SA-1B", "train", False, sa1b_bs, True)
                self.val_loaders["SA-1B"] = _create_dataloader(SA1BDataset, "SA-1B", "val", False, sa1b_bs, False)
                if self.is_main_process:
                    print(f"Created SA-1B train DataLoader with batch_size={sa1b_bs}")
                    print(f"Created SA-1B val DataLoader with batch_size={sa1b_bs}")

            if self.config.training.full_train.enabled and args.mode in ["full_train", "finetune"]:
                full_train_cfg = self.config.training.full_train
                sa1b_subset_bs = full_train_cfg.sa1b_batch_size
                video_bs = full_train_cfg.video_batch_size
                
                if "SA-1B_subset" in full_train_cfg.datasets:
                    self.train_loaders["SA-1B_subset"] = _create_dataloader(SA1BDataset, "SA-1B", "train", False, sa1b_subset_bs, True)
                    if self.is_main_process:
                        print(f"Created SA-1B_subset train DataLoader with batch_size={sa1b_subset_bs}")
                if "SA-V" in full_train_cfg.datasets:
                    # For full train, we use general SA-V
                    self.train_loaders["SA-V"] = _create_dataloader(SAVDataset, "SA-V", "train", True, video_bs, True)
                    self.val_loaders["SA-V"] = _create_dataloader(SAVDataset, "SA-V", "val", True, video_bs, False)
                    if self.is_main_process:
                        print(f"Created SA-V train DataLoader with batch_size={video_bs}")
                        print(f"Created SA-V val DataLoader with batch_size={video_bs}")
                if "Internal" in full_train_cfg.datasets:
                    # Internal dataset is proprietary. Use SAVDataset as a mock for data structure.
                    # A robust implementation would either have a custom InternalDataset class
                    # or a clear placeholder public dataset configured.
                    if self.is_main_process:
                        print("Warning: 'Internal' dataset is proprietary and cannot be reproduced directly. Using SAVDataset as a placeholder for data structure.")
                    self.train_loaders["Internal"] = _create_dataloader(SAVDataset, "Internal", "train", True, video_bs, True)
                    if self.is_main_process:
                        print(f"Created 'Internal' (mocked as SA-V) train DataLoader with batch_size={video_bs}")

                for vos_ds_name in ["DAVIS", "MOSE", "YouTubeVOS"]:
                    if vos_ds_name in full_train_cfg.datasets:
                        self.train_loaders[vos_ds_name] = _create_dataloader(VOSDataset, vos_ds_name, "train", True, video_bs, True)
                        if self.is_main_process:
                            print(f"Created {vos_ds_name} train DataLoader with batch_size={video_bs}")
            
            if args.mode == "finetune":
                # For finetuning, the paper mentions "top 50% most edited masklets from SA-V and Internal datasets"
                finetune_cfg = self.config.training.finetune
                video_bs = self.config.training.full_train.video_batch_size
                
                if self.is_main_process:
                    print(f"Creating DataLoader for finetune stage (challenging videos).")
                
                # `is_finetune_data=True` can signal the dataset to load/filter specific data.
                # For `Internal`, it's still a placeholder.
                self.train_loaders["Challenging_SA-V"] = _create_dataloader(SAVDataset, "SA-V", "train", True, video_bs, True, is_finetune_data=True)
                if "Internal" in full_train_cfg.datasets: # If internal was also enabled in full_train, mock its finetune data
                    self.train_loaders["Challenging_Internal"] = _create_dataloader(SAVDataset, "Internal", "train", True, video_bs, True, is_finetune_data=True)

        # Evaluation DataLoaders (for specific eval modes)
        if args.mode.startswith("eval_"):
            # Eval uses batch_size=1
            if args.mode == "eval_interactive_video" or args.mode == "eval_semisupervised_vos":
                self.test_loaders["SA-V_val"] = _create_dataloader(SAVDataset, "SA-V", "val", True, 1, False)
                self.test_loaders["SA-V_test"] = _create_dataloader(SAVDataset, "SA-V", "test", True, 1, False)
                
                vos_eval_datasets = self.config.get("evaluation.vos_benchmarks", ["DAVIS", "MOSE", "YouTubeVOS"])
                for ds_name in vos_eval_datasets:
                    self.test_loaders[f"{ds_name}_test"] = _create_dataloader(VOSDataset, ds_name, "test", True, 1, False)
                if self.is_main_process:
                    print(f"Created test DataLoaders for interactive/semi-supervised video evaluation.")

            elif args.mode == "eval_image_segmentation":
                # Paper states 37 zero-shot datasets, including 23 SAM's + 14 new video-derived.
                # For this reproduction, we will use SA-1B val as a simple proxy for image eval.
                # A full implementation would load/configure other image benchmarks.
                self.test_loaders["SA-1B_val_img"] = _create_dataloader(SA1BDataset, "SA-1B", "val", False, 1, False)
                if self.is_main_process:
                    print(f"Created test DataLoader for image segmentation evaluation (SA-1B_val_img).")


        # 3. Initialize SAM2Model
        self.model = SAM2Model(self.config).to(self.device)
        
        # Load checkpoint if provided
        if args.checkpoint:
            if not os.path.exists(args.checkpoint):
                raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
            
            if self.is_main_process:
                print(f"Loading model weights from checkpoint: {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location=self.device)
            
            # Extract model_state_dict from checkpoint, handle 'module.' prefix from DDP saves
            model_state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
            
            # Create a new state_dict without 'module.' prefix if it exists in checkpoint
            # This is for loading DDP-saved model into a non-DDP wrapped `SAM2Model` (during init)
            new_state_dict = {}
            for k, v in model_state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v

            self.model.load_state_dict(new_state_dict, strict=False) # strict=False to allow for minor mismatches
            if self.is_main_process:
                print("Model weights loaded successfully.")


        # 4. Wrap model in DDP if distributed training is enabled
        if self.config.system.num_gpus > 1 and self.config.system.rank != -1:
            self.model = DDP(self.model, device_ids=[self.device.index], find_unused_parameters=True)
            # find_unused_parameters=True might be needed for complex models or if some parts are frozen

        # 5. Initialize Trainer (only if in training mode)
        if args.mode in ["pretrain", "full_train", "finetune"]:
            self.trainer = Trainer(self.model, self.config, self.train_loaders, self.val_loaders, self.device)
            if args.checkpoint:
                # If a checkpoint was provided, attempt to load its optimizer/scheduler state
                # The Trainer's _load_checkpoint will handle if these states exist.
                self.trainer._load_checkpoint(args.checkpoint)
        
        # 6. Initialize Evaluator (always, as it might be used for validation during training)
        self.evaluator = Evaluator(self.model, self.config, self.device)

    def run_experiment(self) -> None:
        """
        Executes the experiment based on the parsed mode.
        """
        if self.config is None:
            raise RuntimeError("Config not loaded. Call load_config() first.")
        if self.model is None or self.prompt_simulator is None or self.evaluator is None:
            raise RuntimeError("Components not initialized. Call _initialize_components() first.")

        args = self.parse_args() # Re-parse args to ensure we have them for mode checks

        if args.mode in ["pretrain", "full_train"]:
            if self.trainer is None:
                raise RuntimeError("Trainer not initialized for training mode.")
            if self.is_main_process:
                print(f"Starting {args.mode} training...")
            self.trainer.train()
            if self.is_main_process:
                print(f"{args.mode} training finished.")

        elif args.mode == "finetune":
            if self.trainer is None:
                raise RuntimeError("Trainer not initialized for finetune mode.")
            if self.is_main_process:
                print("Starting fine-tuning...")
            # The Trainer's internal logic will handle finetuning stage based on config.training.finetune.enabled
            self.trainer.train()
            if self.is_main_process:
                print("Fine-tuning finished.")

        elif args.mode.startswith("eval_"):
            if not self.test_loaders:
                if self.is_main_process:
                    print(f"No test loaders configured for mode {args.mode}. Please check config or data setup.")
                return
            
            if self.is_main_process:
                print(f"Starting evaluation in mode: {args.mode}...")
            all_eval_metrics: Dict[str, Dict[str, float]] = {}

            if args.mode == "eval_interactive_video":
                for name, loader in self.test_loaders.items():
                    if self.is_main_process:
                        print(f"Evaluating interactive video (offline) on {name}...")
                    metrics_offline = self.evaluator.evaluate_interactive_video(loader, mode='offline')
                    all_eval_metrics[f"{name}_offline"] = metrics_offline
                    
                    if self.is_main_process:
                        print(f"Evaluating interactive video (online) on {name}...")
                    metrics_online = self.evaluator.evaluate_interactive_video(loader, mode='online')
                    all_eval_metrics[f"{name}_online"] = metrics_online
            
            elif args.mode == "eval_semisupervised_vos":
                for name, loader in self.test_loaders.items():
                    if self.is_main_process:
                        print(f"Evaluating semi-supervised VOS on {name}...")
                    metrics = self.evaluator.evaluate_semisupervised_vos(loader)
                    all_eval_metrics[name] = metrics
            
            elif args.mode == "eval_image_segmentation":
                for name, loader in self.test_loaders.items():
                    if self.is_main_process:
                        print(f"Evaluating image segmentation on {name}...")
                    metrics = self.evaluator.evaluate_image_segmentation(loader)
                    all_eval_metrics[name] = metrics

            if self.is_main_process:
                print("\n--- Final Evaluation Results ---")
                for dataset_name, metrics in all_eval_metrics.items():
                    print(f"Dataset: {dataset_name}")
                    for metric_name, value in metrics.items():
                        print(f"  {metric_name}: {value:.4f}")
                print("--------------------------------")
        else:
            if self.is_main_process:
                print(f"Unknown or unsupported mode: {args.mode}. No action taken.")


if __name__ == "__main__":
    main_app = Main()
    args = main_app.parse_args()
    
    # Load configuration
    main_app.load_config(args)

    # Set random seeds for reproducibility
    seed = main_app.config.get("system.seed", 42)
    torch.manual_seed(seed)
    random.seed(seed)
    # If using CUDA, set device specific seeds
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False # Might decrease performance but ensures reproducibility

    # Setup distributed training environment
    main_app._setup_distributed(args)

    # Initialize all components
    main_app._initialize_components(args)

    # Run the selected experiment
    main_app.run_experiment()

    # Clean up distributed environment if it was initialized
    if dist.is_initialized():
        dist.destroy_process_group()
        if main_app.is_main_process:
            print("Distributed process group destroyed.")


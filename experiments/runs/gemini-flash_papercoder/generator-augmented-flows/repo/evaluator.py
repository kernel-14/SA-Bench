# evaluator.py

import torch
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List, Any, Optional
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator # Import Accelerator for DDP/device management

# Import necessary torchmetrics
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.inception import InceptionScore

# Project-specific imports
from config import Config
from model import ConsistencyModel, EMA # Assuming ConsistencyModel and EMA are defined in model.py

class Evaluator:
    """
    Handles evaluation of the Consistency Model by generating samples and computing
    FID, KID, and Inception Score.
    """

    def __init__(
        self,
        model: ConsistencyModel, # The primary training model
        ema_model: EMA,
        test_dataloader: DataLoader,
        config: Config,
        accelerator: Optional[Accelerator] = None # Optional Accelerator for distributed setup
    ):
        """
        Initializes the Evaluator.

        Args:
            model (ConsistencyModel): The primary training model instance.
            ema_model (EMA): The EMA wrapper containing the target model for generation.
            test_dataloader (DataLoader): DataLoader for real images (validation/test set)
                                          to compare against.
            config (Config): The global configuration object.
            accelerator (Optional[Accelerator]): The Accelerator instance if using distributed training.
        """
        self.model = model
        self.ema_model = ema_model
        self.test_dataloader = test_dataloader
        self.config = config
        self.accelerator = accelerator # Store accelerator for device/DDP management

        # Set device based on config or accelerator
        self.device = self.accelerator.device if self.accelerator else self.config.DEVICE

        # Initialize metrics
        # Images are expected in [0, 255] uint8 for torchmetrics.
        # Our model outputs are in [-1, 1] float, so we need to convert.
        # FID and KID need the same feature extractor.
        # Using feature=2048 corresponds to the 2048-dimensional output from InceptionV3
        # before the final classification layer.
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True).to(self.device)
        self.kid = KernelInceptionDistance(feature=2048, normalize=True).to(self.device)
        # Renamed to avoid conflict with `is` keyword, which is a built-in Python operator
        self.is_metric = InceptionScore(normalize=True).to(self.device) 

        # Pre-load all real images for efficient metric calculation and consistent sampling
        # Only main process does this to avoid redundant memory usage in DDP
        self.real_images_uint8: Optional[torch.Tensor] = None
        if self.accelerator is None or self.accelerator.is_main_process:
            print(f"Collecting and preprocessing all real images from test dataloader...")
            all_real_images_list = []
            for batch_idx, (images, _) in enumerate(tqdm(self.test_dataloader, desc="Loading real images")):
                # Convert from [-1, 1] float (from dataloader transform) to [0, 255] uint8
                images_uint8 = ((images + 1) / 2 * 255).clamp(0, 255).to(torch.uint8) # Clamp for safety
                all_real_images_list.append(images_uint8)
            self.real_images_uint8 = torch.cat(all_real_images_list, dim=0).to(self.device)
            print(f"Loaded {len(self.real_images_uint8)} real images for evaluation.")
        
        # In distributed setup, ensure all processes wait for the main process to load data
        if self.accelerator:
            self.accelerator.wait_for_everyone()


    def generate_samples(self, num_samples: int, model_to_use: ConsistencyModel) -> torch.Tensor:
        """
        Generates a specified number of images using the provided consistency model.
        For Consistency Models, generation is a single-step process from pure noise.

        Args:
            num_samples (int): The total number of images to generate.
            model_to_use (ConsistencyModel): The model instance to use for generation
                                            (e.g., the EMA model).

        Returns:
            torch.Tensor: A tensor of generated images, shape (num_samples, C, H, W),
                          with pixel values in the range [-1, 1].
        """
        # Unwrap model if DDP is used to get the base model for inference
        if self.accelerator:
            model_to_use = self.accelerator.unwrap_model(model_to_use)

        original_train_state = model_to_use.training
        model_to_use.eval() # Set model to evaluation mode

        generated_images_list: List[torch.Tensor] = []
        
        num_generated = 0
        pbar = tqdm(total=num_samples, desc="Generating samples")

        with torch.no_grad(): # Disable gradient calculations during generation
            while num_generated < num_samples:
                current_batch_size = min(self.config.BATCH_SIZE, num_samples - num_generated)
                
                # 1. Generate latent noise vectors z
                # Shape: (batch_size, img_channels, resolution, resolution)
                z = torch.randn(
                    current_batch_size,
                    self.config.IMG_CHANNELS,
                    self.config.RESOLUTION,
                    self.config.RESOLUTION,
                    device=self.device
                )

                # 2. Create sigma_t tensor for the maximum noise level (sigma_T)
                # For consistency models, we map directly from x_T to x_0.
                sigma_t = torch.full(
                    (current_batch_size,),
                    self.config.SIGMA_T,
                    device=self.device
                )

                # 3. Perform a single forward pass to predict x_0
                # Output will be in [-1, 1] range
                generated_batch = model_to_use(z, sigma_t)
                generated_images_list.append(generated_batch.cpu()) # Move to CPU to accumulate to save GPU memory

                num_generated += current_batch_size
                pbar.update(current_batch_size)
        pbar.close()

        # Concatenate all generated batches
        generated_images = torch.cat(generated_images_list, dim=0)

        # Restore model to its original training state
        model_to_use.train(original_train_state) 
        return generated_images

    def _compute_metrics(self, generated_images: torch.Tensor, real_images_for_run: torch.Tensor) -> Dict[str, float]:
        """
        Computes FID, KID, and IS between generated and a subset of real images.

        Args:
            generated_images (torch.Tensor): Generated images, shape (N, C, H, W), range [-1, 1].
            real_images_for_run (torch.Tensor): Real images sampled for this run, shape (N, C, H, W),
                                                range [0, 255], uint8. This tensor should already be
                                                on the correct device (self.device).

        Returns:
            Dict[str, float]: A dictionary containing computed FID, KID, and IS scores.
        """
        # Ensure metrics are on the correct device (already done in __init__ but good for robustness)
        self.fid.to(self.device)
        self.kid.to(self.device)
        self.is_metric.to(self.device)

        # Reset metric states for a fresh calculation
        self.fid.reset()
        self.kid.reset()
        self.is_metric.reset()

        # Preprocess generated images for metrics: [-1, 1] float -> [0, 255] uint8
        # Clamp to ensure values are within valid range after scaling, before converting to uint8
        # Move to device for metric computation
        generated_images_uint8 = ((generated_images + 1) / 2).clamp(0, 1) * 255
        generated_images_uint8 = generated_images_uint8.to(torch.uint8).to(self.device)

        # Update FID and KID with real images (already in [0, 255] uint8 and on device)
        self.fid.update(real_images_for_run, real=True)
        self.kid.update(real_images_for_run, real=True)
        
        # Update FID, KID, and IS with generated images
        self.fid.update(generated_images_uint8, real=False)
        self.kid.update(generated_images_uint8, real=False)
        self.is_metric.update(generated_images_uint8) # IS is computed only on generated.

        # Compute and return results
        fid_score = self.fid.compute().item()
        kid_score = self.kid.compute().item()
        is_mean, is_std = self.is_metric.compute()
        is_score = is_mean.item() # The paper reports mean IS, not std.

        return {
            'FID': fid_score,
            'KID': kid_score,
            'IS': is_score
        }

    def evaluate(self, current_step: int, use_ema_for_generation: bool = True) -> Dict[str, Tuple[float, float]]:
        """
        Orchestrates the full evaluation process, including multiple runs for confidence intervals.

        Args:
            current_step (int): The current training step, used for logging context.
            use_ema_for_generation (bool): If True, uses the EMA model for image generation;
                                           otherwise, uses the primary training model.

        Returns:
            Dict[str, Tuple[float, float]]: A dictionary where keys are metric names (e.g., 'FID'),
                                            and values are tuples of (mean_score, std_dev_score)
                                            across multiple evaluation runs.
        """
        # Ensure evaluation runs only on the main process in distributed training
        if self.accelerator and not self.accelerator.is_main_process:
            # All processes must wait here before the main process returns,
            # otherwise, subsequent training steps might proceed out of sync.
            self.accelerator.wait_for_everyone()
            return {} # Return empty dict if not main process

        model_to_evaluate = self.ema_model.get_model() if use_ema_for_generation else self.model
        
        # Store initial training state of model_to_evaluate if it's the primary model
        # The EMA model is already set to eval() in its __init__ and update methods.
        # If model_to_evaluate is `self.model`, we should save its state and restore it.
        original_train_state = model_to_evaluate.training
        
        # Set model_to_evaluate to eval mode regardless, as generation is an inference task.
        model_to_evaluate.eval() 

        fid_scores: List[float] = []
        kid_scores: List[float] = []
        is_scores: List[float] = []

        print(f"Starting {self.config.EVAL_RUNS} evaluation runs at step {current_step}...")
        for run_idx in range(self.config.EVAL_RUNS):
            print(f"  Evaluation run {run_idx + 1}/{self.config.EVAL_RUNS}")
            
            # Generate new samples for each run
            generated_images = self.generate_samples(self.config.FID_NUM_SAMPLES, model_to_evaluate)
            
            # Randomly sample 'fid_num_samples' real images from the pre-loaded set
            if self.real_images_uint8 is None:
                raise ValueError("Real images not loaded. Ensure test_dataloader has data or that it runs on main process.")
            
            # Ensure we don't try to sample more real images than available
            num_real_images_available = len(self.real_images_uint8)
            num_samples_to_use = min(self.config.FID_NUM_SAMPLES, num_real_images_available)

            if num_samples_to_use < self.config.FID_NUM_SAMPLES:
                print(f"Warning: Only {num_real_images_available} real images available for metrics, "
                      f"requested {self.config.FID_NUM_SAMPLES}. Adjusting number of real samples.")
                # If not enough real images, we can either raise an error or just use what's available.
                # Using what's available is more robust.
                if num_real_images_available == 0:
                    raise ValueError("No real images available for evaluation metrics.")
                
            # Randomly select indices for real images
            rand_indices = torch.randperm(num_real_images_available)[:num_samples_to_use]
            real_images_for_run = self.real_images_uint8[rand_indices].to(self.device)

            # Compute metrics for the current run
            metrics = self._compute_metrics(generated_images, real_images_for_run)
            fid_scores.append(metrics['FID'])
            kid_scores.append(metrics['KID'])
            is_scores.append(metrics['IS'])
        
        # Calculate mean and standard deviation across runs
        avg_metrics: Dict[str, Tuple[float, float]] = {
            'FID': (float(np.mean(fid_scores)), float(np.std(fid_scores))),
            'KID': (float(np.mean(kid_scores)), float(np.std(kid_scores))),
            'IS': (float(np.mean(is_scores)), float(np.std(is_scores))),
        }

        # Restore model to its original training state if it was the primary model
        model_to_evaluate.train(original_train_state)

        # In distributed setup, ensure all processes wait here after evaluation on main process
        if self.accelerator:
            self.accelerator.wait_for_everyone()

        return avg_metrics


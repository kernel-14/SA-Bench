import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from config import Config
from data.datamodule import PDEDataModule
from data.dataset import PDEDataset
from model.moepot import MoEPOT
from utils import calculate_l2re


class Evaluator:
    """
    Handles model evaluation, including L2 relative error computation via auto-regressive
    rollouts, and interpretability analysis of the router-gating network.
    """

    def __init__(self, model: MoEPOT, config: Config, rank: int, world_size: int):
        """
        Initializes the Evaluator.

        Args:
            model: The MoEPOT model instance.
            config: The global configuration object.
            rank: The current process rank in distributed training.
            world_size: The total number of processes participating in distributed training.
        """
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{self.rank}' if torch.cuda.is_available() else 'cpu')

        # Ensure model is on the correct device and in evaluation mode
        self.model = model.to(self.device)
        self.model.eval()

        # If DDP is initialized and model is not already wrapped, wrap it.
        # However, typically the Trainer passes an already DDP-wrapped model.
        if self.world_size > 1 and not isinstance(self.model, DDP):
            self.model = DDP(self.model, device_ids=[self.rank])

        if self.rank == 0:
            print(f"Evaluator initialized. Model on device: {self.device}")

    @torch.no_grad()
    def evaluate_model(self, dataloaders: Dict[str, DataLoader], rollout_steps: int) -> Dict[str, float]:
        """
        Computes the L2 Relative Error (L2RE) for the model across various datasets
        using an auto-regressive rollout strategy.

        Args:
            dataloaders: A dictionary mapping dataset names (e.g., "FNO_1e-5_test")
                         to their respective torch.utils.data.DataLoader instances.
            rollout_steps: An integer specifying the number of future time steps to predict autoregressively.

        Returns:
            A dictionary where keys are dataset names and values are their average L2RE.
        """
        self.model.eval()  # Ensure model is in evaluation mode
        
        results: Dict[str, float] = {}

        for dataset_name, dataloader in dataloaders.items():
            total_l2re_sum = 0.0
            total_samples_count = 0

            if self.rank == 0:
                print(f"  Evaluating {dataset_name} with {rollout_steps} rollout steps...")
                pbar = tqdm(dataloader, desc=f"  {dataset_name} Evaluation")
            else:
                pbar = dataloader

            for batch_idx, batch in enumerate(pbar):
                # u_seq: (B, T_full, C_total, H_res, W_res) - entire sequence from dataset
                # u_target: (B, C_total, H_res, W_res) - This is just the *first* target frame if T_in was used.
                #           For rollout, we need the GT sequence corresponding to rollout_steps.
                #           The PDEDataset provides 'u_seq' and 'u_target' where u_target is just the NEXT frame.
                #           For multi-step rollout evaluation, we need to access subsequent ground truth frames.
                #           Let's assume the `batch['u_seq']` provided by the `PDEDataset` is actually
                #           a longer sequence `(B, T_available, C, H, W)` from which we can extract both
                #           initial `T_in` and subsequent `rollout_steps` for GT comparison.
                # The PDEDataset actually loads `self.preprocessed_data` as `(N_samples, T_raw, C, H, W)`.
                # `__getitem__` returns `u_seq` as `(T_in, C, H, W)` and `u_target` as `(C, H, W)`.
                # To do rollouts, we need to generate `rollout_steps` predictions.
                # And for L2RE, compare to `rollout_steps` ground truth frames.
                # This means we need the *full ground truth sequence* after the initial T_in frames.
                # This needs to be provided by the dataloader.
                #
                # Re-check PDEDataset `__getitem__` signature in plan: `__getitem__(idx: int) -> Dict[str, torch.Tensor]`
                # returns `u_seq` and `u_target`. This `u_target` is just the *next single frame*.
                #
                # To perform a rollout:
                # 1. We take `u_seq` from batch.
                # 2. We need the `ground_truth_rollout_sequence` from the dataset, starting at `time_step_start + T_in`.
                #
                # This means `PDEDataset.__getitem__` needs to provide more information,
                # specifically the `idx` that allows us to retrieve `u_seq` and a *slice* of GT for rollout.
                #
                # Let's adjust this: The dataloader `batch['u_seq']` is actually the *full* time series `T_raw` frames for evaluation.
                # This is more common in evaluation. `u_seq` becomes `(batch_size, T_raw, C, H, W)`.
                # For `evaluate_model`, `PDEDataset.__getitem__` should ideally provide the full sample,
                # or at least a segment long enough for `T_in + rollout_steps`.
                #
                # For `PDEDataset`, `__getitem__` logic:
                # `data_sample_idx = idx // self.num_time_steps_for_prediction`
                # `time_step_start = idx % self.num_time_steps_for_prediction`
                # `u_seq_input = self.preprocessed_data[data_sample_idx, time_step_start : time_step_start + self.T_in]`
                # `u_target_gt = self.preprocessed_data[data_sample_idx, time_step_start + self.T_in]`
                #
                # So if we want to predict `rollout_steps` frames, we need GT for `time_step_start + T_in` to `time_step_start + T_in + rollout_steps - 1`.
                # The simplest is to ensure the batch provides:
                # - `initial_input_sequence`: `(B, T_in, C, H, W)`
                # - `ground_truth_sequence_for_rollout`: `(B, rollout_steps, C, H, W)`
                # The `PDEDataset` should be modified to supply this structure when `evaluate_model` is called.
                # For now, let's assume `batch['initial_input']` and `batch['gt_rollout']`.
                #
                # Reread `PDEDataset`'s `__getitem__`: "u_seq" (T_in frames), "u_target" (next frame).
                # This means for rollouts, we need to fetch items from the raw `preprocessed_data`.
                #
                # **Resolution of Rollout Data Handling**:
                # The `PDEDataset` is currently set up to provide `T_in` input frames and `1` target frame.
                # For evaluating rollout, we need to iterate on `time_step_start`.
                # A single `__getitem__` call gives `u_seq = u[t:t+T_in]`, `u_target = u[t+T_in]`.
                # To predict `rollout_steps` frames, we start prediction from `u[time_step_start+T_in]`.
                # The actual ground truth for comparison would be `u[time_step_start+T_in : time_step_start+T_in+rollout_steps]`.
                #
                # Let's adjust the retrieval:
                # `u_seq_all_available_frames = batch['u_seq_full_sample']` from the `PDEDataset`.
                # And `PDEDataset` needs to return `time_step_start` as well.
                # This complicates the `PDEDataset` interface for this specific evaluation task.

                # Alternative, simpler approach:
                # When evaluating `evaluate_model`, each `batch` from the `DataLoader` will contain
                # `u_seq` (the initial `T_in` frames) and `u_target` (the single next ground truth frame).
                # We start the rollout with `u_seq`. We generate `predicted_frames_list`.
                # For comparing, we need the subsequent `rollout_steps` ground truth frames.
                # The `PDEDataset` as designed provides `u_seq` and `u_target` (the immediate next frame).
                # To get the *full ground truth sequence* for comparison, we need to get `idx` from the batch or pass
                # enough context for `PDEDataset` to give us the full GT sequence for `rollout_steps`.
                #
                # **Decision for `evaluate_model`**:
                # Modify `PDEDataset.__getitem__` to return:
                # `initial_u_seq`: `(T_in, C, H, W)` (first T_in frames)
                # `full_gt_sequence`: `(T_in + rollout_steps, C, H, W)` (for actual comparison)
                # This way, the batch contains everything needed.
                # This requires `PDEDataset` to know `rollout_steps` during initialization or `__getitem__`.
                #
                # Simpler: The `PDEDataset` returns `u_seq` (T_in frames) and `u_target` (next single frame).
                # For L2RE, we compare `predicted_frame` against `u_target`. The summation in the paper:
                # `sum_{1 <= t <= T} || G_w(u^<t + epsilon) - u^t ||_2^2`. This is single-step.
                # But evaluation in paper Section 5.3 and Appendix C.3 talks about rollout.
                # "predict the solution x_pred for the next 10 steps".
                # Table 12 for rollout at different timesteps.
                # So the `evaluate_model` logic must do a multi-step rollout.

                # Let's assume `PDEDataset` when constructing the DataLoader for `evaluate_model`
                # (e.g., through `val_dataloader` or `test_dataloader` which use `PDEDataset` internally)
                # it provides `u_seq_initial` (T_in frames) and `u_gt_rollout` (rollout_steps frames).
                # This means `PDEDataset.__getitem__` will be context-aware of `rollout_steps`.
                # This implies modifying `PDEDataset` or using a separate dataset for evaluation.
                # For this task, I'll assume `batch` contains `u_initial_input` and `u_gt_rollout_sequence`.

                # Let's assume `PDEDataset.__getitem__` will provide `u_initial_input` (T_in frames)
                # and `u_gt_rollout_sequence` (rollout_steps frames) for `evaluate_model`.
                # This makes the `PDEDataset` slightly different for evaluation path vs training path.
                # For now, I will retrieve `initial_input_u_seq` and then extract `gt_rollout_sequence`
                # directly from the `PDEDataset`'s `self.preprocessed_data` using the `idx`
                # and `time_step_start` which can be returned by `PDEDataset.__getitem__`.
                #
                # **Revised `PDEDataset.__getitem__`**: return `u_seq`, `time_step_start`, `data_sample_idx`.
                # Then in `Evaluator.evaluate_model`, use these to access `self.datamodule.raw_data` directly.
                # No, `PDEDataset` should return `u_seq`, `u_target` and also `u_future_frames` (for rollout).
                #
                # **Final Decision on Rollout Data**:
                # `PDEDataset.__getitem__` will return:
                #   - `u_seq`: `(T_in, C, H, W)` (the initial T_in frames)
                #   - `gt_for_rollout_eval`: `(rollout_steps, C, H, W)` (the subsequent ground truth frames)
                # This requires `PDEDataset` to know the `rollout_steps` for evaluation.
                # I will modify `PDEDataset`'s `__init__` to take `rollout_steps_for_eval` and return `gt_for_rollout_eval` in `__getitem__` accordingly.
                # For simplicity, for this `evaluator.py`, I will assume `PDEDataset` in its evaluation mode
                # provides `initial_input_u_seq` and `ground_truth_rollout_sequence`.

                u_initial_input = batch['initial_input_u_seq'].to(self.device) # (B, T_in, C, H_res, W_res)
                u_gt_rollout_sequence = batch['ground_truth_rollout_sequence'].to(self.device) # (B, rollout_steps, C, H_res, W_res)
                
                batch_size = u_initial_input.shape[0]

                current_input_sequence = u_initial_input # (B, T_in, C, H_res, W_res)
                predicted_frames_list = []

                for r in range(rollout_steps):
                    # Model expects (B, T_in, C, H_res, W_res)
                    predicted_frame, _ = self.model(current_input_sequence) # (B, C_out, H_res, W_res)

                    predicted_frames_list.append(predicted_frame)

                    # Update current_input_sequence for next step
                    # Remove the oldest frame (index 0) and append the new prediction
                    # predicted_frame (B, C_out, H_res, W_res) needs new time dim for concat
                    # Make sure C_out matches C_in (self.config.model.input_channels == self.config.model.output_channels)
                    current_input_sequence = torch.cat(
                        (current_input_sequence[:, 1:, :, :, :], predicted_frame.unsqueeze(1)), # unsqueeze(1) for time dim
                        dim=1
                    ) # (B, T_in, C, H_res, W_res)

                full_prediction_sequence = torch.stack(predicted_frames_list, dim=1) # (B, rollout_steps, C_out, H_res, W_res)

                # Calculate L2RE for the batch
                # Ensure ground truth sequence has enough frames for comparison
                if u_gt_rollout_sequence.shape[1] < rollout_steps:
                    raise ValueError(f"Ground truth sequence in batch has {u_gt_rollout_sequence.shape[1]} frames, "
                                     f"but {rollout_steps} rollout steps are required for evaluation. "
                                     f"Adjust PDEDataset to provide sufficient GT frames.")
                
                # Take the first `rollout_steps` from the GT sequence for comparison
                gt_sequence_for_l2re = u_gt_rollout_sequence[:, :rollout_steps, :, :, :]

                batch_l2re = calculate_l2re(full_prediction_sequence, gt_sequence_for_l2re)
                
                total_l2re_sum += batch_l2re * batch_size
                total_samples_count += batch_size
            
            # Aggregate results across all distributed processes
            if self.world_size > 1:
                # Sums for all_reduce must be on device
                total_l2re_sum_tensor = torch.tensor(total_l2re_sum, device=self.device)
                total_samples_count_tensor = torch.tensor(total_samples_count, device=self.device)
                dist.all_reduce(total_l2re_sum_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_samples_count_tensor, op=dist.ReduceOp.SUM)
                total_l2re_sum = total_l2re_sum_tensor.item()
                total_samples_count = total_samples_count_tensor.item()
            
            if total_samples_count > 0:
                results[dataset_name] = total_l2re_sum / total_samples_count
            else:
                results[dataset_name] = float('inf') # No samples to evaluate

        self.model.train() # Set back to train mode
        return results

    @torch.no_grad()
    def run_interpretability(self, datamodule: PDEDataModule) -> Dict[str, float]:
        """
        Analyzes the router-gating network's ability to classify PDE types based on expert selection patterns.

        Args:
            datamodule: A PDEDataModule instance containing all pre-training datasets.

        Returns:
            A dictionary containing the interpretability accuracy.
        """
        self.model.eval()  # Ensure model is in evaluation mode

        num_routed_experts = self.config.model.num_routed_experts
        num_blocks = self.config.model.num_layers
        interpretability_block_idx = 1  # Block 2 (0-indexed is 1) as per paper (Fig 4c)

        if self.rank == 0:
            print(f"\n--- Running Interpretability Analysis (Block {interpretability_block_idx+1}) ---")

        # Phase 1: Compute Average Expert Distributions for each dataset
        avg_expert_selections_per_dataset: Dict[int, List[torch.Tensor]] = {}
        dataset_type_map_str_to_int = datamodule.get_dataset_info()['dataset_type_map']
        
        # Invert map for easy lookup later: int_id -> str_name
        dataset_type_map_int_to_str = {v: k for k, v in dataset_type_map_str_to_int.items()}

        # Need to iterate through individual datasets from the datamodule
        # datamodule.train_datasets is a List[PDEDataset]
        for dataset_obj in datamodule.train_datasets:
            dataset_name = dataset_obj.dataset_name
            dataset_int_id = dataset_obj.get_dataset_type_idx().item()
            
            # Create a DataLoader for this single dataset for calculating average distributions
            single_dataset_dataloader = DataLoader(
                dataset_obj,
                batch_size=self.config.training.batch_size // self.world_size,
                shuffle=False, # Order doesn't matter for averages
                num_workers=os.cpu_count() // self.world_size if self.world_size > 0 else os.cpu_count(),
                pin_memory=True,
                drop_last=False
            )
            
            if self.rank == 0:
                print(f"  Calculating average expert selections for {dataset_name}...")
            
            avg_dists_for_dataset = self.get_average_expert_selection_dist(
                single_dataset_dataloader, num_routed_experts, num_blocks
            )
            avg_expert_selections_per_dataset[dataset_int_id] = avg_dists_for_dataset

        # Phase 2: Evaluate Classification Accuracy on test sets
        correct_predictions = 0
        total_samples = 0

        for dataset_obj in datamodule.test_datasets:
            dataset_name = dataset_obj.dataset_name
            true_dataset_type_idx = dataset_obj.get_dataset_type_idx().item()
            
            # Create a DataLoader for this single dataset for evaluation
            single_dataset_dataloader = DataLoader(
                dataset_obj,
                batch_size=self.config.training.batch_size // self.world_size,
                shuffle=False,
                num_workers=os.cpu_count() // self.world_size if self.world_size > 0 else os.cpu_count(),
                pin_memory=True,
                drop_last=False
            )

            if self.rank == 0:
                print(f"  Classifying samples from {dataset_name}...")

            for batch_idx, batch in enumerate(single_dataset_dataloader):
                u_seq = batch['u_seq'].to(self.device) # (B, T_in, C, H_res, W_res)
                
                # Perform forward pass to get router weights
                _, router_weights_per_layer = self.model(u_seq) # Noise is not applied in evaluation

                # Extract router outputs for the specific block for classification
                router_output_for_block = router_weights_per_layer[interpretability_block_idx] # (B, num_routed_experts)

                for sample_idx in range(router_output_for_block.shape[0]):
                    input_router_output = router_output_for_block[sample_idx] # (num_routed_experts,)
                    
                    predicted_dataset_idx = self.classify_dataset_type(
                        input_router_output,
                        avg_expert_selections_per_dataset,
                        interpretability_block_idx
                    )

                    if predicted_dataset_idx == true_dataset_type_idx:
                        correct_predictions += 1
                    total_samples += 1
        
        # Aggregate results across all distributed processes
        if self.world_size > 1:
            correct_predictions_tensor = torch.tensor(correct_predictions, device=self.device)
            total_samples_tensor = torch.tensor(total_samples, device=self.device)
            dist.all_reduce(correct_predictions_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
            correct_predictions = correct_predictions_tensor.item()
            total_samples = total_samples_tensor.item()

        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        if self.rank == 0:
            print(f"Interpretability Analysis Accuracy (Block {interpretability_block_idx+1}): {accuracy:.4f}")

        self.model.train() # Set back to train mode
        return {"interpretability_accuracy": accuracy}

    @torch.no_grad()
    def get_average_expert_selection_dist(self, dataloader: DataLoader, num_experts: int, num_blocks: int) -> List[torch.Tensor]:
        """
        Helper method to calculate the average router output (expert selection distribution)
        for each MoE block given a specific dataset's DataLoader.

        Args:
            dataloader: The DataLoader for a specific dataset.
            num_experts: The total number of routed experts.
            num_blocks: The total number of MoE blocks.

        Returns:
            A list of num_blocks tensors, each of shape (num_experts,),
            representing the average distribution for each block.
        """
        self.model.eval() # Ensure model is in evaluation mode

        expert_selection_sums_per_block: List[torch.Tensor] = [
            torch.zeros(num_experts, device=self.device) for _ in range(num_blocks)
        ]
        total_samples_processed = 0

        for batch_idx, batch in enumerate(dataloader):
            u_seq = batch['u_seq'].to(self.device) # (B, T_in, C, H_res, W_res)
            batch_size = u_seq.shape[0]

            # Perform forward pass to get router weights
            # Noise is not applied during evaluation
            _, router_weights_per_layer = self.model(u_seq) # Each element (B, num_routed_experts)

            for l in range(num_blocks):
                expert_selection_sums_per_block[l] += torch.sum(router_weights_per_layer[l], dim=0)
            
            total_samples_processed += batch_size

        # Aggregate sums and counts across all distributed processes
        if self.world_size > 1:
            total_samples_processed_tensor = torch.tensor(total_samples_processed, device=self.device)
            dist.all_reduce(total_samples_processed_tensor, op=dist.ReduceOp.SUM)
            total_samples_processed = total_samples_processed_tensor.item()

            for l in range(num_blocks):
                dist.all_reduce(expert_selection_sums_per_block[l], op=dist.ReduceOp.SUM)

        avg_dists: List[torch.Tensor] = []
        if total_samples_processed > 0:
            for l in range(num_blocks):
                avg_dists.append(expert_selection_sums_per_block[l] / total_samples_processed)
        else:
            avg_dists = [torch.zeros(num_experts, device=self.device) for _ in range(num_blocks)]
        
        return avg_dists

    @torch.no_grad()
    def classify_dataset_type(self,
                              input_router_output: torch.Tensor,
                              avg_dists_per_dataset: Dict[int, List[torch.Tensor]],
                              block_idx: int) -> int:
        """
        Classifies a single input sample's PDE type by comparing its router output
        to pre-computed average expert selection distributions of known datasets
        using cross-entropy distance.

        Args:
            input_router_output: A torch.Tensor of shape (num_routed_experts,) representing
                                 the softmax output of the router for a single block and single input sample.
            avg_dists_per_dataset: A dictionary mapping dataset integer IDs to lists of
                                   average expert selection distributions (one list per block).
            block_idx: An integer specifying which MoE block's router output to use for classification.

        Returns:
            The predicted integer ID of the dataset type.
        """
        min_loss = float('inf')
        predicted_dataset_type_idx = -1
        
        # Add epsilon to input_router_output to avoid log(0) if it contains zeros,
        # but technically it comes from softmax so it shouldn't be exactly zero.
        # However, for numerical stability it's good practice.
        input_router_output_stable = input_router_output + 1e-9

        for dataset_int_id, avg_dist_list_for_dataset in avg_dists_per_dataset.items():
            y_i = avg_dist_list_for_dataset[block_idx] # (num_routed_experts,)
            
            # Add epsilon to y_i to prevent log(0) errors
            y_i_stable = y_i + 1e-9
            
            # Calculate cross-entropy loss (negative sum of p * log(q))
            current_loss = -torch.sum(input_router_output_stable * torch.log(y_i_stable))

            if current_loss < min_loss:
                min_loss = current_loss
                predicted_dataset_type_idx = dataset_int_id
        
        return predicted_dataset_type_idx


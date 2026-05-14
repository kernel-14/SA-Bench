
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from collections import defaultdict

from config import config
from models import MoEPOT
from data import PDEDataset, BalancedBatchSampler
from utils import L2RelativeError

def evaluate_model(model, dataloader, criterion_l2, device):
    model.eval()
    total_l2_error = 0.0
    all_expert_weights_flat = defaultdict(list) # To store expert weights per dataset
    
    with torch.no_grad():
        for u_input, u_target, mask, dataset_names in tqdm(dataloader, desc="Evaluating"):
            u_input = u_input.to(device)
            u_target = u_target.to(device)
            mask = mask.to(device)

            prediction, _, block_expert_weights_list = model(u_input) # block_expert_weights_list contains expert_weights_flat for each block
            
            masked_prediction = prediction * mask
            masked_target = u_target * mask

            l2_error = criterion_l2(masked_prediction, masked_target)
            total_l2_error += l2_error.item()

            # Store expert weights for interpretability analysis
            for i, ds_name in enumerate(dataset_names):
                # For simplicity, let's take weights from the first block for now
                # The paper averages over blocks for interpretability.
                # Here, we will aggregate all and average later per block if needed.
                if len(block_expert_weights_list) > 0:
                    all_expert_weights_flat[ds_name].append(block_expert_weights_list[0][i].cpu().numpy())

    avg_l2_error = total_l2_error / len(dataloader)
    return avg_l2_error, all_expert_weights_flat

def interpretability_analysis(all_expert_weights_flat: dict, num_routed_experts: int):
    """
    Performs interpretability analysis to classify dataset based on router-gating network selection.
    As described in Appendix B.4.
    """
    print("\n--- Interpretability Analysis (Dataset Classification based on Router) ---")
    
    if not all_expert_weights_flat:
        print("No expert weights collected for analysis.")
        return

    # Calculate average expert selection distribution for each dataset in the test set
    avg_dataset_expert_distributions = {}
    for dataset_name, weights_list in all_expert_weights_flat.items():
        # weights_list contains (N_patches, num_routed_experts) for each sample
        # We need to average over samples and patches
        avg_weights = np.mean(np.concatenate(weights_list, axis=0), axis=0) # (num_routed_experts,)
        avg_dataset_expert_distributions[dataset_name] = torch.tensor(avg_weights, dtype=torch.float32)

    # Convert average distributions to log probabilities for cross-entropy
    log_avg_dataset_expert_distributions = {
        name: torch.log(dist + 1e-9) for name, dist in avg_dataset_expert_distributions.items()
    }

    # Evaluate classification accuracy
    correct_classifications = 0
    total_samples = 0
    
    # Re-iterate through the raw expert weights (not the averaged ones) to classify each sample
    for true_dataset_name, weights_list in all_expert_weights_flat.items():
        for sample_weights in weights_list: # sample_weights: (N_patches, num_routed_experts)
            # For each sample, take the average expert distribution over its patches
            sample_avg_weights = torch.tensor(np.mean(sample_weights, axis=0), dtype=torch.float32)
            
            # Convert to log probability
            log_sample_avg_weights = torch.log(sample_avg_weights + 1e-9)

            min_distance = float('inf')
            predicted_dataset_name = None

            for ref_dataset_name, ref_log_dist in log_avg_dataset_expert_distributions.items():
                # Cross-entropy function as defined in paper (B.4)
                distance = -torch.sum(sample_avg_weights * ref_log_dist)
                
                if distance < min_distance:
                    min_distance = distance
                    predicted_dataset_name = ref_dataset_name
            
            if predicted_dataset_name == true_dataset_name:
                correct_classifications += 1
            total_samples += 1
            
    if total_samples > 0:
        accuracy = correct_classifications / total_samples
        print(f"Dataset Classification Accuracy: {accuracy * 100:.2f}%")
    else:
        print("No samples to classify.")


def main():
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dummy dataset paths (must match those used in train.py to find dummy data)
    dummy_dataset_paths = {name: os.path.join(config.data.base_path, f"{name.lower().replace('-', '_').replace('.', '')}.npy") for name in config.data.dataset_names}
    
    # max_channels_in_data must be consistent with training
    max_channels_in_data = 3 # Max channels observed in dummy data

    test_dataset = PDEDataset(
        dataset_paths=dummy_dataset_paths,
        dataset_names=config.data.dataset_names,
        is_train=False, # Load test split
        target_resolution=config.data.h_resolution,
        target_time_steps=config.data.time_steps,
        padding_value=config.data.padding_value,
        noise_epsilon=0.0, # No noise for evaluation
        train_split_ratio=config.data.train_split_ratio,
        max_channels=max_channels_in_data
    )

    test_sampler = BalancedBatchSampler(test_dataset, config.data.balance_weights, config.data.batch_size)

    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        num_workers=config.num_workers,
        pin_memory=True
    )

    # Load model from checkpoint
    checkpoint_path = os.path.join(config.checkpoint.save_dir, f"{config.run_name}_best.pth")
    if not os.path.exists(checkpoint_path):
        print(f"Error: No model checkpoint found at {checkpoint_path}. Please train a model first.")
        return

    # Update model config with actual input/output channels
    config.model.in_channels = max_channels_in_data
    config.model.out_channels = max_channels_in_data

    model = MoEPOT(
        patch_size=config.model.patch_size,
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        embed_dim=config.model.attention_dim,
        mlp_dim=config.model.mlp_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        num_routed_experts=config.model.num_routed_experts,
        num_shared_experts=config.model.num_shared_experts,
        top_k=config.model.top_k_experts,
        H=config.data.h_resolution,
        W=config.data.h_resolution,
        time_steps=config.data.time_steps
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {checkpoint_path}")

    criterion_l2 = L2RelativeError()

    avg_l2_error, all_expert_weights_flat = evaluate_model(model, test_loader, criterion_l2, device)
    print(f"\nAverage L2 Relative Error on test set: {avg_l2_error:.6f}")

    # Perform interpretability analysis
    interpretability_analysis(all_expert_weights_flat, config.model.num_routed_experts)

if __name__ == '__main__':
    main()

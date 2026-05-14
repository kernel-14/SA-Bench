import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from lora_sb_model import inject_lora_sb_layers, LoRASBLayer
from lora_sb_init import compute_delta_w_avg_and_sign, apply_lora_sb_initialization_to_model
from config import LoRASBConfig

# 1. Define a dummy model for demonstration
class DummyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

# 2. Generate dummy data
def generate_dummy_data(input_dim, output_dim, num_samples, batch_size):
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, output_dim, (num_samples,))
    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=batch_size)
    return data_loader

def main():
    # Configuration
    config = LoRASBConfig(
        rank=16,
        scaling_factor=1.0, # As per paper, scaling_factor can be set to 1.0 with LoRA-SB
        target_modules=["linear1", "linear2"], # Target linear layers in our DummyModel
        init_num_samples=50,
        init_learning_rate=1e-4
    )

    # Model parameters
    input_dim = 128
    hidden_dim = 64
    output_dim = 10
    num_data_samples = 1000
    batch_size = 16

    # Instantiate the original model
    original_model = DummyModel(input_dim, hidden_dim, output_dim).to(config.device)
    print("Original Model Architecture:")
    print(original_model)

    # Generate dummy data
    dummy_data_loader = generate_dummy_data(input_dim, output_dim, num_data_samples, batch_size)

    # Loss function for gradient calculation
    loss_fn = nn.CrossEntropyLoss().to(config.device)

    print(f"
Computing average full FT gradients on {config.init_num_samples} samples...")
    # Compute delta_w_avg for original linear layers
    delta_w_avg_map = compute_delta_w_avg_and_sign(
        model=original_model,
        data_loader=dummy_data_loader,
        loss_fn=loss_fn,
        num_samples=config.init_num_samples,
        learning_rate=config.init_learning_rate
    )
    print("Average gradients computed for layers:", delta_w_avg_map.keys())

    # Create a fresh instance of the model for LoRA-SB injection
    # This is important because compute_delta_w_avg_and_sign modified `requires_grad`
    # on the original_model's weights, and we want a clean model state for injection.
    model_for_lora_sb = DummyModel(input_dim, hidden_dim, output_dim).to(config.device)
    # Ensure W_0 in LoRASBLayer correctly captures the original weights of model_for_lora_sb

    print("
Injecting LoRA-SB layers...")
    # Inject LoRA-SB layers into the model
    model_for_lora_sb = inject_lora_sb_layers(
        model=model_for_lora_sb,
        target_modules=config.target_modules,
        rank=config.rank,
        scaling_factor=config.scaling_factor
    )
    print("LoRA-SB Injected Model Architecture:")
    print(model_for_lora_sb)

    print("
Applying LoRA-SB initialization...")
    # Apply the SVD-based initialization to the injected LoRA-SB layers
    apply_lora_sb_initialization_to_model(model_for_lora_sb, delta_w_avg_map)

    print("
Initialization complete. Showing trainable parameters:")
    # Verify only R matrices are trainable
    trainable_params = []
    for name, param in model_for_lora_sb.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
    print(f"Trainable parameters: {trainable_params}")
    if all("R" in name for name in trainable_params):
        print("Successfully set only R matrices as trainable.")
    else:
        print("Warning: Non-R parameters are also trainable or R parameters are missing.")

    # Example of how to setup an optimizer for training
    optimizer = torch.optim.AdamW(model_for_lora_sb.parameters(), lr=config.init_learning_rate)
    print(f"
Optimizer setup with {len(optimizer.param_groups[0]['params'])} trainable parameter groups.")

    print("
LoRA-SB setup demonstration complete.")

if __name__ == "__main__":
    main()

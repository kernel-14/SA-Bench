import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import os

from src.models.universal_neural_operator import UniversalNeuralOperator
from src.utils.metrics import nmae
from config.model_config import MODEL_CONFIGS, TRAINING_CONFIGS

# Placeholder for data loading - replace with actual data loading logic
def load_dataset(dataset_name, data_dim=1):
    print(f"Loading placeholder dataset: {dataset_name}")
    # Simulate loading data: input (a) and output (u)
    if data_dim == 1:
        # Example: (batch_size, spatial_dim, channels)
        input_data = torch.randn(100, 64, 3) # 100 samples, 64 spatial points, 3 input channels
        output_data = torch.randn(100, 64, 1) # 100 samples, 64 spatial points, 1 output channel
    elif data_dim == 2:
        # Example: (batch_size, x_dim, y_dim, channels)
        input_data = torch.randn(100, 32, 32, 3) # 100 samples, 32x32 spatial points, 3 input channels
        output_data = torch.randn(100, 32, 32, 1) # 100 samples, 32x32 spatial points, 1 output channel
    else:
        raise ValueError("Unsupported data_dim for placeholder dataset.")
        
    # Normalize input and output if needed (important for FNOs)
    # For this placeholder, we assume it's already in a reasonable range.
    
    dataset = TensorDataset(input_data, output_data)
    return dataset, input_data.shape[-1], output_data.shape[-1]

def train_epoch(model, dataloader, optimizer, criterion, device, is_pretraining=True):
    model.train()
    total_loss = 0
    total_nmae = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_nmae += nmae(output, y).item()

    avg_loss = total_loss / len(dataloader)
    avg_nmae = total_nmae / len(dataloader)
    return avg_loss, avg_nmae

def evaluate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_nmae = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            total_loss += loss.item()
            total_nmae += nmae(output, y).item()
    avg_loss = total_loss / len(dataloader)
    avg_nmae = total_nmae / len(dataloader)
    return avg_loss, avg_nmae

def pretrain(model_config_name, pretrain_dataset_name, device, save_path="./checkpoints", data_dim=1):
    print(f"
--- Starting Pre-training for {model_config_name} on {pretrain_dataset_name} ---")
    model_config = MODEL_CONFIGS[model_config_name]
    training_config = TRAINING_CONFIGS["pretrain"]

    # Load dataset and determine channel sizes
    dataset, in_channels, out_channels = load_dataset(pretrain_dataset_name, data_dim)
    dataloader = DataLoader(dataset, batch_size=training_config["batch_size"], shuffle=True)

    # Update lifting/projection params with actual channel sizes
    model_config["lifting_params"]["in_channels"] = in_channels
    model_config["projection_params"]["out_channels"] = out_channels

    model = UniversalNeuralOperator(**model_config).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.get_trainable_parameters_for_pretraining(), lr=training_config["learning_rate"])

    os.makedirs(save_path, exist_ok=True)

    for epoch in range(training_config["epochs"]):
        train_loss, train_nmae = train_epoch(model, dataloader, optimizer, criterion, device, is_pretraining=True)
        print(f"Epoch {epoch+1}/{training_config["epochs"]}: Train Loss = {train_loss:.4e}, Train NMAE = {train_nmae:.4f}")

        if (epoch + 1) % training_config["save_interval"] == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f"{model_config_name}_pretrained_epoch_{epoch+1}.pt"))
            print(f"Model saved to {save_path}/{model_config_name}_pretrained_epoch_{epoch+1}.pt")

    print(f"--- Pre-training for {model_config_name} finished ---")
    return model

def finetune(pretrained_model, finetune_dataset_name, device, save_path="./checkpoints", data_dim=1):
    print(f"
--- Starting Fine-tuning for {pretrained_model.core_model_type} on {finetune_dataset_name} ---")
    
    model_config_name = f"{pretrained_model.core_model_type}_{data_dim}D"
    model_config = MODEL_CONFIGS[model_config_name]
    training_config = TRAINING_CONFIGS["finetune"]

    # Freeze core model parameters
    pretrained_model.freeze_core()

    # Load fine-tuning dataset and determine channel sizes
    dataset, in_channels, out_channels = load_dataset(finetune_dataset_name, data_dim)
    dataloader = DataLoader(dataset, batch_size=training_config["batch_size"], shuffle=True)

    # Update lifting/projection params with actual channel sizes for fine-tuning task
    pretrained_model.lifting_adapter = LiftingAdapter(in_channels, model_config["lifting_params"]["out_channels"]).to(device)
    pretrained_model.projection_adapter = ProjectionAdapter(model_config["projection_params"]["in_channels"], out_channels).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(pretrained_model.get_trainable_parameters_for_finetuning(), lr=training_config["learning_rate"])
    
    for epoch in range(training_config["epochs"]):
        train_loss, train_nmae = train_epoch(pretrained_model, dataloader, optimizer, criterion, device, is_pretraining=False)
        print(f"Epoch {epoch+1}/{training_config["epochs"]}: Fine-tune Train Loss = {train_loss:.4e}, Fine-tune Train NMAE = {train_nmae:.4f}")

        if (epoch + 1) % training_config["save_interval"] == 0:
            torch.save(pretrained_model.state_dict(), os.path.join(save_path, f"{pretrained_model.core_model_type}_finetuned_on_{finetune_dataset_name}_epoch_{epoch+1}.pt"))
            print(f"Fine-tuned model saved to {save_path}/{pretrained_model.core_model_type}_finetuned_on_{finetune_dataset_name}_epoch_{epoch+1}.pt")

    print(f"--- Fine-tuning for {pretrained_model.core_model_type} finished ---")
    return pretrained_model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Example Pre-training for a 1D FNO model
    pretrained_fno_1d_model = pretrain(
        model_config_name="FNO_1D", 
        pretrain_dataset_name="BurgersEquation_1D", 
        device=device, 
        data_dim=1
    )

    # Example Fine-tuning for the 1D FNO model on a slightly different 1D problem
    finetuned_fno_1d_model = finetune(
        pretrained_fno_1d_model, 
        finetune_dataset_name="HeatEquation_1D", 
        device=device, 
        data_dim=1
    )

    # Example Pre-training for a 2D MambaFNO model
    pretrained_mambafno_2d_model = pretrain(
        model_config_name="MambaFNO_2D", 
        pretrain_dataset_name="NavierStokes_2D", 
        device=device, 
        data_dim=2
    )

    # Example Fine-tuning for the 2D MambaFNO model on a different 2D problem
    finetuned_mambafno_2d_model = finetune(
        pretrained_mambafno_2d_model, 
        finetune_dataset_name="ReactionDiffusion_2D", 
        device=device, 
        data_dim=2
    )
    
    # Example of pre-training PerceiverFNO (2D)
    pretrained_perceiverfno_2d_model = pretrain(
        model_config_name="PerceiverFNO_2D", 
        pretrain_dataset_name="Advection_2D", 
        device=device, 
        data_dim=2
    )

    # Example of fine-tuning PerceiverFNO (2D)
    finetuned_perceiverfno_2d_model = finetune(
        pretrained_perceiverfno_2d_model,
        finetune_dataset_name="AnotherPhysics_2D",
        device=device,
        data_dim=2
    )

if __name__ == "__main__":
    main()


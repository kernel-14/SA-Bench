import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import WDNO
from data import WaveletDataset
from config import Config

def train():
    # Load configuration
    config = Config()

    # Initialize model
    model = WDNO(
        wavelet_basis=config.wavelet_basis,
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim
    )
    model = model.to(config.device)

    # Load dataset
    train_dataset = WaveletDataset(config.train_data_path)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.MSELoss()

    # Training loop
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(config.device), targets.to(config.device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{config.epochs}], Loss: {epoch_loss/len(train_loader):.4f}")

    # Save the trained model
    torch.save(model.state_dict(), config.model_save_path)

if __name__ == "__main__":
    train()
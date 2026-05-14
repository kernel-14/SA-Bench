import torch
from torch.optim import AdamW
import torch.nn as nn
from torch.utils.data import DataLoader
from src.autoencoder import P2VAE
from src.dataset import load_data

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_p2vae(data_dir, epochs=100, batch_size=64, latent_dim=16):
    train_loader = load_data(data_dir, batch_size, split='train')
    valid_loader = load_data(data_dir, batch_size, split='valid')
    
    model = P2VAE(input_dim=3, latent_dim=latent_dim).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            reconstructed, mu, log_var = model(batch)
            loss = model.loss_function(reconstructed, batch, mu, log_var, beta=1e-3)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss}')
    
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            for batch in valid_loader:
                batch = batch.to(DEVICE)
                reconstructed, mu, log_var = model(batch)
                loss = model.loss_function(reconstructed, batch, mu, log_var, beta=1e-3)
                total_val_loss += loss.item()
            print(f'Validation Loss: {total_val_loss}')

if __name__ == '__main__':
    train_p2vae(data_dir='/home/data')

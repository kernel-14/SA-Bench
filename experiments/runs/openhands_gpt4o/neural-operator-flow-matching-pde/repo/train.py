import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import P2VAE, FlowMarchingTransformer
from data import PDEDataLoader
from config import Config

def train():
    config = Config()

    # Initialize models
    p2vae = P2VAE(input_dim=config.input_dim, latent_dim=config.latent_dim, hidden_dim=config.hidden_dim).to(config.device)
    fmt = FlowMarchingTransformer(latent_dim=config.latent_dim, hidden_dim=config.hidden_dim, num_layers=config.num_layers).to(config.device)

    # Optimizers
    optimizer_p2vae = optim.AdamW(p2vae.parameters(), lr=config.lr_p2vae, weight_decay=config.weight_decay)
    optimizer_fmt = optim.AdamW(fmt.parameters(), lr=config.lr_fmt, weight_decay=config.weight_decay)

    # Loss functions
    reconstruction_loss = nn.MSELoss()

    # DataLoader
    train_loader = PDEDataLoader(config.train_data_path, batch_size=config.batch_size, shuffle=True)

    for epoch in range(config.epochs):
        p2vae.train()
        fmt.train()

        for batch in train_loader:
            x = batch.to(config.device)

            # P2VAE forward pass
            x_recon, mu, logvar = p2vae(x)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            recon_loss = reconstruction_loss(x_recon, x)
            loss_p2vae = recon_loss + config.beta * kl_loss

            optimizer_p2vae.zero_grad()
            loss_p2vae.backward()
            optimizer_p2vae.step()

            # FMT forward pass
            latent_states = p2vae.encoder(x)
            src, tgt = latent_states[:-1], latent_states[1:]
            fmt_output = fmt(src, tgt)
            fmt_loss = reconstruction_loss(fmt_output, tgt)

            optimizer_fmt.zero_grad()
            fmt_loss.backward()
            optimizer_fmt.step()

        print(f"Epoch {epoch + 1}/{config.epochs}, P2VAE Loss: {loss_p2vae.item()}, FMT Loss: {fmt_loss.item()}")

if __name__ == "__main__":
    train()
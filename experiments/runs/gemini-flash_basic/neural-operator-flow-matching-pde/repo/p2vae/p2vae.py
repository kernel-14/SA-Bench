import torch
import torch.nn as nn

class P2VAE(nn.Module):
    def __init__(self, in_channels=3, latent_channels=16, base_channels=64, img_size=128, latent_size=16):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.base_channels = base_channels
        self.img_size = img_size
        self.latent_size = latent_size

        # Encoder: c3p128 -> c16p16
        # The paper mentions reusing SD-VAE architecture. This is a simplified representation.
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1), # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1), # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1), # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(base_channels * 4, latent_channels, kernel_size=3, stride=1, padding=1) # No change in resolution, adjust channels
        )

        # Decoder: c16p16 -> c3p128
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, base_channels * 4, kernel_size=3, stride=1, padding=1), # No change in resolution
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=3, stride=2, padding=1, output_padding=1), # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=3, stride=2, padding=1, output_padding=1), # 32 -> 64
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1), # 64 -> 128
            nn.Sigmoid() # Assuming output is normalized to [0, 1] for image-like data
        )

        # For KL divergence, typically mean and log-variance are predicted
        self.fc_mu = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)
        self.fc_logvar = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, beta=1e-3):
        # Reconstruction loss (MSE)
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')

        # KL divergence loss
        # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + beta * kl_loss, recon_loss, kl_loss


# Example usage (for verification, not part of the submission)
if __name__ == "__main__":
    # P2VAE-16M uses base_channels=64
    p2vae_16m = P2VAE(base_channels=64)
    print(f"P2VAE-16M Parameters: {sum(p.numel() for p in p2vae_16m.parameters())}")

    # P2VAE-87M uses base_channels=128
    p2vae_87m = P2VAE(base_channels=128)
    print(f"P2VAE-87M Parameters: {sum(p.numel() for p in p2vae_87m.parameters())}")

    # Test forward pass
    dummy_input = torch.randn(1, 3, 128, 128) # Batch size 1, 3 channels, 128x128
    recon_x, mu, logvar = p2vae_16m(dummy_input)
    print("Reconstructed shape:", recon_x.shape)
    print("Mu shape:", mu.shape)
    print("Logvar shape:", logvar.shape)

    loss, recon_loss, kl_loss = p2vae_16m.loss_function(recon_x, dummy_input, mu, logvar)
    print("Total Loss:", loss.item())
    print("Reconstruction Loss:", recon_loss.item())
    print("KL Loss:", kl_loss.item())

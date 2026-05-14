import torch
import torch.nn as nn

class P2VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(P2VAE, self).__init__()
        # Encoder Layers
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(128 * 32 * 32, latent_dim)  # Mapping to mean
        self.fc_var = nn.Linear(128 * 32 * 32, latent_dim) # Mapping to variance
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, input_dim, kernel_size=3, stride=2, padding=1)
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x).view(x.size(0), -1)
        mu = self.fc_mu(encoded)
        log_var = self.fc_var(encoded)
        z = self.reparameterize(mu, log_var)
        decoded = self.decoder(z.view(z.size(0), -1, 1, 1))
        return decoded, mu, log_var

    def loss_function(self, reconstructed, original, mu, log_var, beta=0.001):
        reconstruction_loss = nn.MSELoss()(reconstructed, original)
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return reconstruction_loss + beta * kl_div

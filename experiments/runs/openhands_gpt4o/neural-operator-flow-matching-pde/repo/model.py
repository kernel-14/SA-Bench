import torch
import torch.nn as nn
import torch.nn.functional as F

class P2VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(P2VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(hidden_dim * 2 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2 * 16 * 16, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2 * 16 * 16),
            nn.ReLU(),
            nn.Unflatten(1, (hidden_dim * 2, 16, 16)),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, input_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

class FlowMarchingTransformer(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_layers):
        super(FlowMarchingTransformer, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.transformer = nn.Transformer(d_model=latent_dim, nhead=8, num_encoder_layers=num_layers, num_decoder_layers=num_layers)
        self.fc_out = nn.Linear(latent_dim, latent_dim)

    def forward(self, src, tgt):
        output = self.transformer(src, tgt)
        output = self.fc_out(output)
        return output
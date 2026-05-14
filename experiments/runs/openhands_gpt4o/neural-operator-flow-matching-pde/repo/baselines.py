import torch
import torch.nn as nn

class UNetBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(UNetBaseline, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, input_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output

class FNOBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FNOBaseline, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.conv3 = nn.Conv2d(hidden_dim, input_dim, kernel_size=1)

    def forward(self, x):
        x = torch.fft.fft2(x)
        x = self.conv1(x.real) + 1j * self.conv1(x.imag)
        x = self.conv2(x.real) + 1j * self.conv2(x.imag)
        x = torch.fft.ifft2(x)
        return self.conv3(x.real)
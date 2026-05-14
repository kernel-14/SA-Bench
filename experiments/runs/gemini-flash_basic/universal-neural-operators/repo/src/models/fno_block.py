import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to retain

        self.scale = (1 / (in_channels * out_channels)) ** 0.5
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Transform input to Fourier space
        x_ft = torch.fft.rfft(x)

        #Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        out_ft[:, :, -self.modes1:] = self.compl_mul1d(x_ft[:, :, -self.modes1:], self.weights2)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNOBlock1d(nn.Module):
    def __init__(self, width, modes, num_layers):
        super(FNOBlock1d, self).__init__()
        self.width = width
        self.modes = modes
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.ws = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(SpectralConv1d(self.width, self.width, self.modes))
            self.ws.append(nn.Conv1d(self.width, self.width, 1))

    def forward(self, x):
        # x: (batchsize, width, x_dim)
        for i in range(self.num_layers):
            x1 = self.convs[i](x)
            x2 = self.ws[i](x)
            x = x1 + x2
            if i < self.num_layers - 1:
                x = F.gelu(x)
        return x

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 
        self.modes2 = modes2 

        self.scale = (1 / (in_channels * out_channels)) ** 0.5
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-2, -1])

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:] = self.compl_mul2d(x_ft[:, :, :self.modes1, -self.modes2:], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:] = self.compl_mul2d(x_ft[:, :, -self.modes1:, -self.modes2:], self.weights4)

        x = torch.fft.irfftn(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNOBlock2d(nn.Module):
    def __init__(self, width, modes1, modes2, num_layers):
        super(FNOBlock2d, self).__init__()
        self.width = width
        self.modes1 = modes1
        self.modes2 = modes2
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.ws = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(SpectralConv2d(self.width, self.width, self.modes1, self.modes2))
            self.ws.append(nn.Conv2d(self.width, self.width, 1))

    def forward(self, x):
        # x: (batchsize, width, x_dim, y_dim)
        for i in range(self.num_layers):
            x1 = self.convs[i](x)
            x2 = self.ws[i](x)
            x = x1 + x2
            if i < self.num_layers - 1:
                x = F.gelu(x)
        return x

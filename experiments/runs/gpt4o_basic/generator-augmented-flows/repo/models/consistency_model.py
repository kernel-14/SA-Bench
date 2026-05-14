import torch
import torch.nn as nn
import torch.nn.functional as F

class ConsistencyModel(nn.Module):
    def __init__(self, input_channels, resolution):
        super(ConsistencyModel, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * resolution * resolution // 4, 256)
        self.fc2 = nn.Linear(256, input_channels * resolution * resolution)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(x.size(0), 3, 32, 32)  # Output reshaped


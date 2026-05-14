import torch
import torch.nn as nn

class RoboticWorldModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, context_horizon, forecast_horizon):
        super(RoboticWorldModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.context_horizon = context_horizon
        self.forecast_horizon = forecast_horizon

        # GRU for sequential processing
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        
        # Fully connected layers for output (mean & std dev)
        self.output_mean = nn.Linear(hidden_dim, output_dim)
        self.output_std = nn.Linear(hidden_dim, output_dim)

    def forward(self, observations, actions):
        # Combine sequences of observations and actions
        x = torch.cat([observations, actions], dim=-1)

        # Process through GRU
        output, hidden = self.gru(x)

        # Predict mean and std dev of distributions (Gaussian parameters)
        mean = self.output_mean(output)
        std_dev = self.output_std(output)

        # Ensure standard deviation is positive (softplus activation)
        std_dev = torch.nn.functional.softplus(std_dev)

        return mean, std_dev



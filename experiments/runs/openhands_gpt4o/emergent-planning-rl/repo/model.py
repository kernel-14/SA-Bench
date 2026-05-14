import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, spatial_size):
        height, width = spatial_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class DRC(nn.Module):
    def __init__(self, input_channels, hidden_dim, num_layers, num_internal_ticks, action_dim):
        super(DRC, self).__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_internal_ticks = num_internal_ticks
        self.action_dim = action_dim

        self.encoder = nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1)
        self.convlstm_layers = nn.ModuleList([
            ConvLSTMCell(hidden_dim, hidden_dim, kernel_size=3) for _ in range(num_layers)
        ])
        self.policy_head = nn.Linear(hidden_dim * 8 * 8, action_dim)
        self.value_head = nn.Linear(hidden_dim * 8 * 8, 1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        x = self.encoder(x)
        hidden_states = [layer.init_hidden(batch_size, (height, width)) for layer in self.convlstm_layers]

        for _ in range(self.num_internal_ticks):
            for i, layer in enumerate(self.convlstm_layers):
                h, c = hidden_states[i]
                h, c = layer(x, (h, c))
                hidden_states[i] = (h, c)
                x = h

        x = x.view(batch_size, -1)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value
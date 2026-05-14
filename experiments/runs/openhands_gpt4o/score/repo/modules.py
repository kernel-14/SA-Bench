import torch
import torch.nn as nn

class RewardShapingModule(nn.Module):
    def __init__(self, alpha: float):
        super(RewardShapingModule, self).__init__()
        self.alpha = alpha

    def forward(self, first_attempt_reward, second_attempt_reward):
        progress_reward = self.alpha * (second_attempt_reward - first_attempt_reward)
        return progress_reward
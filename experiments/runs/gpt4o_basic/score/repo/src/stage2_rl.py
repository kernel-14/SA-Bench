import torch
import torch.nn.functional as F
from torch.optim import Adam

class Stage2RL:
    def __init__(self, model, dataset, shaped_reward_multiplier):
        self.model = model
        self.dataset = dataset
        self.shaped_reward_multiplier = shaped_reward_multiplier
        self.optimizer = Adam(self.model.parameters(), lr=1e-5)

    def compute_reward(self, y_pred, y_true):
        # Reward computation stub
        return torch.sum(y_pred == y_true).item()

    def compute_shaped_reward(self, y_pred_first, y_pred_second, y_true):
        progress = torch.sum(y_pred_second != y_pred_first).item()
        final_reward = self.compute_reward(y_pred_second, y_true)
        return final_reward + self.shaped_reward_multiplier * progress

    def step(self):
        self.model.train()
        for batch in self.dataset:
            x, y_true = batch
            y_pred_first = self.model(x)
            y_pred_second = self.model(torch.cat([x, y_pred_first], dim=1))

            # Compute rewards
            shaped_reward = self.compute_shaped_reward(y_pred_first, y_pred_second, y_true)

            loss = -shaped_reward

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, epochs):
        for epoch in range(epochs):
            self.step()
            print(fStage II Epoch {epoch + 1}/{epochs} completed.)



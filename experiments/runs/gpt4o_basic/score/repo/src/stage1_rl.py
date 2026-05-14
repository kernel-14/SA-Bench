import torch
import torch.nn.functional as F
from torch.optim import Adam

class Stage1RL:
    def __init__(self, model, reference_model, dataset, kl_multiplier):
        self.model = model
        self.reference_model = reference_model
        self.dataset = dataset
        self.kl_multiplier = kl_multiplier
        self.optimizer = Adam(self.model.parameters(), lr=1e-5)

    def compute_reward(self, y_pred, y_true):
        # Reward computation stub
        return torch.sum(y_pred == y_true).item()

    def step(self):
        self.model.train()
        for batch in self.dataset:
            x, y_true = batch
            y_pred_first = self.model(x)

            # KL-Divergence penalty
            kl_penalty = F.kl_div(F.log_softmax(y_pred_first, dim=-1), 
                                  F.softmax(self.reference_model(x), dim=-1), reduction='batchmean')

            y_pred_second = self.model(torch.cat([x, y_pred_first], dim=1))
            reward = self.compute_reward(y_pred_second, y_true)

            loss = (-reward + self.kl_multiplier * kl_penalty)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, epochs):
        for epoch in range(epochs):
            self.step()
            print(fEpoch {epoch + 1}/{epochs} completed.)



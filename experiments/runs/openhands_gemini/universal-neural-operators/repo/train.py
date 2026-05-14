
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import os
import numpy as np

from config import get_config
from model import get_model
from data import get_dataloader

def NMAE(pred, target, grid_min=None, grid_max=None, epsilon=1e-8):
    """
    Range-Normalized Mean Absolute Error (NMAE)
    NMAE(theta) = 1/|D_test| sum_{(a,u) in D_test} ||G_theta(a) - u||_1,G / (max_G u - min_G u + epsilon)
    """
    if grid_min is None:
        grid_min = target.min()
    if grid_max is None:
        grid_max = target.max()

    range_u = grid_max - grid_min + epsilon
    nmae = torch.mean(torch.abs(pred - target)) / range_u
    return nmae

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)

        # Load data
        self.train_loader, input_channels, output_channels = get_dataloader(config, split='train')
        self.val_loader, _, _ = get_dataloader(config, split='val')
        self.test_loader, _, _ = get_dataloader(config, split='test')

        # Initialize model with updated config (input/output channels are now set)
        model_config = self.config
        model_config.input_channels = input_channels
        model_config.output_channels = output_channels
        self.model = get_model(model_config).to(self.device)

        # Optimizer and Scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)

        # Loss function
        self.criterion = nn.MSELoss()

        # Output directory
        os.makedirs(config.output_dir, exist_ok=True)

    def _train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_nmae = 0
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_nmae += NMAE(outputs, targets).item()

            if batch_idx % self.config.log_interval == 0:
                print(f"Train Epoch: {epoch} [{batch_idx * len(inputs)}/{len(self.train_loader.dataset)} "
                      f"({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

        avg_loss = total_loss / len(self.train_loader)
        avg_nmae = total_nmae / len(self.train_loader)
        self.scheduler.step()
        return avg_loss, avg_nmae

    def _evaluate(self, loader, desc="Validation"):
        self.model.eval()
        total_loss = 0
        total_nmae = 0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                total_nmae += NMAE(outputs, targets).item()

        avg_loss = total_loss / len(loader)
        avg_nmae = total_nmae / len(loader)
        print(f'{desc} set: Average loss: {avg_loss:.4f}, Average NMAE: {avg_nmae:.4f}')
        return avg_loss, avg_nmae

    def pretrain(self):
        print("Starting Pre-training...")
        best_val_loss = float('inf')
        for epoch in range(1, self.config.pretrain_epochs + 1):
            train_loss, train_nmae = self._train_one_epoch(epoch)
            val_loss, val_nmae = self._evaluate(self.val_loader)
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train NMAE={train_nmae:.4f}, "
                  f"Val Loss={val_loss:.4f}, Val NMAE={val_nmae:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.config.output_dir, 'pretrain_best_model.pth'))
                print(f"Saved best model at epoch {epoch} with validation loss: {best_val_loss:.4f}")

        # Evaluate on test set after pre-training
        print("Evaluating pre-trained model on test set...")
        test_loss, test_nmae = self._evaluate(self.test_loader, desc="Test")
        print(f"Pre-train Test Loss: {test_loss:.4f}, Pre-train Test NMAE: {test_nmae:.4f}")

    def finetune(self, pretrained_model_path=None):
        print("Starting Fine-tuning...")
        if pretrained_model_path:
            self.model.load_state_dict(torch.load(pretrained_model_path))
            print(f"Loaded pretrained model from {pretrained_model_path}")

        # Freeze operator blocks if fine-tuning adapters only
        if self.config.fine_tune_adapters_only:
            for name, param in self.model.named_parameters():
                if "lifting_adapters" not in name and "projection_adapters" not in name:
                    param.requires_grad = False
            # Re-initialize optimizer with only trainable parameters
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                        lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
            self.scheduler = StepLR(self.optimizer, step_size=self.config.scheduler_step_size, gamma=self.config.scheduler_gamma)


        best_val_loss = float('inf')
        for epoch in range(1, self.config.finetune_epochs + 1):
            train_loss, train_nmae = self._train_one_epoch(epoch)
            val_loss, val_nmae = self._evaluate(self.val_loader)
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train NMAE={train_nmae:.4f}, "
                  f"Val Loss={val_loss:.4f}, Val NMAE={val_nmae:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.config.output_dir, 'finetune_best_model.pth'))
                print(f"Saved best model at epoch {epoch} with validation loss: {best_val_loss:.4f}")
        
        # Evaluate on test set after fine-tuning
        print("Evaluating fine-tuned model on test set...")
        test_loss, test_nmae = self._evaluate(self.test_loader, desc="Test")
        print(f"Fine-tune Test Loss: {test_loss:.4f}, Fine-tune Test NMAE: {test_nmae:.4f}")


def main():
    config = get_config()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    trainer = Trainer(config)

    # Example workflow: pre-train, then fine-tune
    trainer.pretrain()
    trainer.finetune(pretrained_model_path=os.path.join(config.output_dir, 'pretrain_best_model.pth'))

if __name__ == '__main__':
    main()

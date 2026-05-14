
import torch

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random

class PDEDataset(Dataset):
    def __init__(self, data_path, pde_type, split='train', subsample_rate=1, output_res=64, num_samples=1000):
        self.data_path = data_path
        self.pde_type = pde_type
        self.split = split
        self.subsample_rate = subsample_rate
        self.output_res = output_res
        self.num_samples = num_samples # For simulated data

        self.input_data = [] # List of input functions (a)
        self.output_data = [] # List of output solutions (u)

        # In a real scenario, this would load actual data from files.
        # For reproduction, we simulate data based on PDE type.
        self._generate_simulated_data()
        self._split_data()

        # Determine input and output channels based on simulated data
        self.input_channels = self.input_data[0].shape[-1]
        self.output_channels = self.output_data[0].shape[-1]

    def _generate_simulated_data(self):
        # Simulate data for different PDE types
        # In a real scenario, you'd load pre-computed solutions from PDEBench or similar.
        print(f"Simulating data for {self.pde_type}...")
        initial_spatial_dim = 128 # Assume original resolution
        
        for i in range(self.num_samples):
            # Input 'a' could be initial conditions, coefficients, mesh info
            # Output 'u' is the solution field
            
            # Simulate data at initial_spatial_dim, then subsample
            if self.pde_type == 'Burgers':
                input_a_full = np.random.rand(initial_spatial_dim, 2) # e.g., initial condition + spatial coord
                output_u_full = np.random.rand(initial_spatial_dim, 1) # Solution at time T
            elif self.pde_type == 'Gray-Scott':
                input_a_full = np.random.rand(initial_spatial_dim, 3) # e.g., initial u, v, and spatial coord
                output_u_full = np.random.rand(initial_spatial_dim, 2) # Solution for u, v
            elif self.pde_type == 'Navier-Stokes':
                input_a_full = np.random.rand(initial_spatial_dim, 3) # e.g., initial velocity, pressure, spatial coord
                output_u_full = np.random.rand(initial_spatial_dim, 2) # Final velocity field
            elif self.pde_type == 'Heat':
                input_a_full = np.random.rand(initial_spatial_dim, 2) # Initial temp + spatial coord
                output_u_full = np.random.rand(initial_spatial_dim, 1) # Final temp distribution
            elif self.pde_type == 'Reaction-Diffusion':
                input_a_full = np.random.rand(initial_spatial_dim, 3) # Initial concentrations + spatial coord
                output_u_full = np.random.rand(initial_spatial_dim, 2) # Final concentrations
            else:
                raise ValueError(f"Unknown PDE type: {self.pde_type}")
            
            # Apply subsampling
            input_a_sub = input_a_full[::self.subsample_rate]
            output_u_sub = output_u_full[::self.subsample_rate]

            # Resize to output_res if different
            if output_u_sub.shape[0] != self.output_res:
                input_a_sub = torch.tensor(input_a_sub, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1) # (1, C, S)
                output_u_sub = torch.tensor(output_u_sub, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1) # (1, C, S)

                input_a_resized = F.interpolate(input_a_sub, size=self.output_res, mode='linear', align_corners=False).squeeze(0).permute(1, 0) # (S', C)
                output_u_resized = F.interpolate(output_u_sub, size=self.output_res, mode='linear', align_corners=False).squeeze(0).permute(1, 0) # (S', C)
                
                self.input_data.append(input_a_resized)
                self.output_data.append(output_u_resized)
            else:
                self.input_data.append(torch.tensor(input_a_sub, dtype=torch.float32))
                self.output_data.append(torch.tensor(output_u_sub, dtype=torch.float32))

    def _split_data(self):
        # In a real scenario, this would load data based on pre-defined splits
        # Here, we randomly split the simulated data
        total_samples = len(self.input_data)
        indices = list(range(total_samples))
        random.shuffle(indices)

        train_idx_end = int(total_samples * 0.8)
        val_idx_end = int(total_samples * (0.8 + 0.1))

        if self.split == 'train':
            self.indices = indices[:train_idx_end]
        elif self.split == 'val':
            self.indices = indices[train_idx_end:val_idx_end]
        elif self.split == 'test':
            self.indices = indices[val_idx_end:]
        else:
            raise ValueError(f"Unknown split: {self.split}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        return self.input_data[actual_idx], self.output_data[actual_idx]

def get_dataloader(config, split='train'):
    dataset = PDEDataset(
        data_path=config.data_path,
        pde_type=config.dataset_name,
        split=split,
        subsample_rate=config.subsample_rate,
        output_res=config.output_res,
        num_samples=1000 # Using a fixed number of samples for simulation
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(split == 'train'),
        num_workers=config.num_workers,
        pin_memory=True
    )
    return dataloader, dataset.input_channels, dataset.output_channels


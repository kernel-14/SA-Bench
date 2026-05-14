# model.py

import torch
import torch.nn as nn
from modules import SpatialPyramid, TemporalPyramid, DiffusionTransformer

class PyramidalFlowMatchingModel(nn.Module):
    def __init__(self, num_stages: int, base_model: nn.Module):
        super(PyramidalFlowMatchingModel, self).__init__()
        self.num_stages = num_stages
        self.base_model = base_model
        self.spatial_pyramid = SpatialPyramid(num_stages)
        self.temporal_pyramid = TemporalPyramid(num_stages)

    def forward(self, x):
        # Apply spatial pyramid
        x = self.spatial_pyramid(x)
        # Apply temporal pyramid
        x = self.temporal_pyramid(x)
        # Pass through the base model
        x = self.base_model(x)
        return x

# Example instantiation
if __name__ == "__main__":
    base_model = DiffusionTransformer()
    model = PyramidalFlowMatchingModel(num_stages=3, base_model=base_model)
    print(model)
import torch
import torch.nn as nn
from spatial_pyramid import spatial_pyramidal_flow
from temporal_pyramid import temporal_pyramidal_flow

class PyramidalFlowMatching(nn.Module):
    def __init__(self, resolutions, num_temporal_resolutions):
        super(PyramidalFlowMatching, self).__init__()
        self.resolutions = resolutions
        self.num_temporal_resolutions = num_temporal_resolutions

    def forward(self, current_frame, history_frames):
        # Spatial pyramid flow
        spatial_results = spatial_pyramidal_flow(current_frame, self.resolutions)

        # Temporal pyramid flow
        temporal_results = temporal_pyramidal_flow(current_frame, history_frames, self.num_temporal_resolutions)

        return spatial_results, temporal_results

def unified_training_step(model, batch_input):
    """Simulated training step."""
    current_frame, history_frames = batch_input
    spatial_results, temporal_results = model(current_frame, history_frames)
    # Compute loss (placeholders for actual loss computation)
    loss = sum([torch.mean(res) for res in spatial_results]) + torch.mean(temporal_results)
    return loss

if __name__ == "__main__":
    # Example implementation
    model = PyramidalFlowMatching(resolutions=[2, 4, 8], num_temporal_resolutions=[2, 4])
    current_frame = torch.rand(1, 3, 256, 256)
    history_frames = [torch.rand(1, 3, 256, 256) for _ in range(4)]
    loss = unified_training_step(model, (current_frame, history_frames))
    print(f"Training step completed with loss: {loss.item()}")


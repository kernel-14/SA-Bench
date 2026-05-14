import torch
import torch.nn.functional as F

def downsample(tensor, scale):
    """Downsample a tensor by a given scale factor."""
    return F.interpolate(tensor, scale_factor=1/scale, mode="bilinear", align_corners=False)

def temporal_pyramidal_flow(current_frame, history_frames, resolutions):
    """Conditionally generate the current frame from autoregressive histories at temporal pyramid resolutions."""

    conditioned_frames = []

    for i, resolution in enumerate(resolutions):
        history_downsampled = [downsample(frame, resolution) for frame in history_frames]
        frame_condition = torch.stack(history_downsampled, dim=0)
        conditioned_frames.append(frame_condition)

    # Combine conditions with current frame
    final_condition = sum(conditioned_frames) / len(conditioned_frames)
    result_frame = current_frame * 0.5 + final_condition * 0.5

    return result_frame

if __name__ == "__main__":
    # Example usage
    current_frame = torch.rand(1, 3, 256, 256)
    history_frames = [torch.rand(1, 3, 256, 256) for _ in range(4)]
    resolutions = [2, 4]
    output_frame = temporal_pyramidal_flow(current_frame, history_frames, resolutions)
    print(f"Generated conditioned frame with shape: {output_frame.shape}")


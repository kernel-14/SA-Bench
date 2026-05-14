import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import os
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional, Union

# Assuming Config class is available from config.py
# To avoid circular imports, we use a forward reference or assume it's imported
# For the purpose of this file, we define a minimal stub or rely on explicit import in main.py
# For standalone testing, a local stub might be necessary.
try:
    from config import Config, TrainingStageConfig # This will be imported by main.py
except ImportError:
    # Minimal stub for linting/IDE if config.py is not yet fully available
    class TrainingStageConfig:
        learning_rate: float = 1e-4
        warmup_steps: int = 1000
        optimizer_beta1: float = 0.9
        optimizer_beta2: float = 0.999
        optimizer_epsilon: float = 1e-6
        weight_decay: float = 1e-4

    class ComputeConfig:
        device: str = "cuda"

    class DataPathsConfig:
        model_weights: str = "./weights/model.pth"

    class Config:
        training: Dict[int, TrainingStageConfig] = {
            1: TrainingStageConfig(), 2: TrainingStageConfig(), 3: TrainingStageConfig()
        }
        compute: ComputeConfig = ComputeConfig()
        data_paths: DataPathsConfig = DataPathsConfig()


# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def downsample(tensor: torch.Tensor, factor: int, mode: str = "bilinear") -> torch.Tensor:
    """
    Spatially downsamples an input tensor.

    Args:
        tensor (torch.Tensor): The input tensor of shape (B, C, H, W) or (B, C, D, H, W).
        factor (int): The downsampling ratio.
        mode (str): The interpolation algorithm. For 2D: "nearest", "bilinear", "bicubic".
                    For 3D: "nearest", "trilinear". Defaults to "bilinear".

    Returns:
        torch.Tensor: The downsampled tensor.
    """
    if factor <= 0:
        raise ValueError("Downsampling factor must be a positive integer.")
    
    current_shape = tensor.shape
    new_dims = [dim // factor for dim in current_shape[-tensor.ndim+2:]] # Spatial dimensions
    
    if tensor.ndim == 4:  # (B, C, H, W)
        if len(new_dims) != 2:
            raise ValueError(f"Expected 2 spatial dimensions for 2D tensor, got {len(new_dims)}")
        new_size = (new_dims[0], new_dims[1])
        if mode not in ["nearest", "bilinear", "bicubic"]:
            logger.warning(f"Unsupported 2D interpolation mode '{mode}'. Falling back to 'bilinear'.")
            mode = "bilinear"
    elif tensor.ndim == 5:  # (B, C, D, H, W)
        if len(new_dims) != 3:
            raise ValueError(f"Expected 3 spatial dimensions for 3D tensor, got {len(new_dims)}")
        new_size = (new_dims[0], new_dims[1], new_dims[2])
        if mode not in ["nearest", "trilinear"]:
            logger.warning(f"Unsupported 3D interpolation mode '{mode}'. Falling back to 'trilinear'.")
            mode = "trilinear"
    else:
        raise ValueError(f"Unsupported tensor dimension: {tensor.ndim}. Expected 4 (2D) or 5 (3D).")

    return F.interpolate(tensor, size=new_size, mode=mode, align_corners=False if "linear" in mode else None)


def upsample(tensor: torch.Tensor, factor: int, mode: str = "bilinear") -> torch.Tensor:
    """
    Spatially upsamples an input tensor.

    Args:
        tensor (torch.Tensor): The input tensor of shape (B, C, H, W) or (B, C, D, H, W).
        factor (int): The upsampling ratio.
        mode (str): The interpolation algorithm. For 2D: "nearest", "bilinear", "bicubic".
                    For 3D: "nearest", "trilinear". Defaults to "bilinear".

    Returns:
        torch.Tensor: The upsampled tensor.
    """
    if factor <= 0:
        raise ValueError("Upsampling factor must be a positive integer.")

    current_shape = tensor.shape
    new_dims = [dim * factor for dim in current_shape[-tensor.ndim+2:]] # Spatial dimensions
    
    if tensor.ndim == 4:  # (B, C, H, W)
        if len(new_dims) != 2:
            raise ValueError(f"Expected 2 spatial dimensions for 2D tensor, got {len(new_dims)}")
        new_size = (new_dims[0], new_dims[1])
        if mode not in ["nearest", "bilinear", "bicubic"]:
            logger.warning(f"Unsupported 2D interpolation mode '{mode}'. Falling back to 'bilinear'.")
            mode = "bilinear"
    elif tensor.ndim == 5:  # (B, C, D, H, W)
        if len(new_dims) != 3:
            raise ValueError(f"Expected 3 spatial dimensions for 3D tensor, got {len(new_dims)}")
        new_size = (new_dims[0], new_dims[1], new_dims[2])
        if mode not in ["nearest", "trilinear"]:
            logger.warning(f"Unsupported 3D interpolation mode '{mode}'. Falling back to 'trilinear'.")
            mode = "trilinear"
    else:
        raise ValueError(f"Unsupported tensor dimension: {tensor.ndim}. Expected 4 (2D) or 5 (3D).")

    return F.interpolate(tensor, size=new_size, mode=mode, align_corners=False if "linear" in mode else None)


def get_spatial_pos_embed(H: int, W: int, D: int, embed_dim: int, device: torch.device) -> torch.Tensor:
    """
    Generates sinusoidal position embeddings for spatial dimensions (D, H, W).
    Combines 1D sinusoidal embeddings for each dimension.
    The 'extrapolation' aspect is implicitly handled by generating embeddings
    dynamically for the given D, H, W.

    Args:
        H (int): Height of the feature map.
        W (int): Width of the feature map.
        D (int): Depth/number of frames of the feature map.
        embed_dim (int): The dimension of the embeddings. This should match the DiT's hidden_size.
        device (torch.device): The computation device.

    Returns:
        torch.Tensor: A tensor of shape (1, D*H*W, embed_dim) representing the spatial
                      position embeddings.

    Note: The original design did not include `embed_dim` as an argument.
          It has been added as it is essential for generating meaningful positional embeddings.
    """
    if embed_dim % 2 != 0:
        raise ValueError(f"Embedding dimension {embed_dim} must be even for sinusoidal encoding.")

    # Divide embed_dim by the number of spatial dimensions (3 for D,H,W; 2 for H,W)
    # to distribute the embedding features.
    num_spatial_dims = 3 if D > 1 else 2
    dim_per_coord = embed_dim // num_spatial_dims

    # Ensure dim_per_coord is even for sine/cosine pairs
    if dim_per_coord % 2 != 0:
        dim_per_coord += 1 # Make it even, will truncate/pad later if needed
    
    # Generate 1D positional encodings
    def _get_1d_sincos_embed(length: int, embed_dim_1d: int) -> torch.Tensor:
        position = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim_1d, 2, dtype=torch.float32, device=device) * -(torch.log(torch.tensor(10000.0, device=device)) / embed_dim_1d))
        sin_pe = torch.sin(position * div_term)
        cos_pe = torch.cos(position * div_term)
        pe = torch.cat([sin_pe, cos_pe], dim=-1)
        return pe # Shape: (length, embed_dim_1d)

    # For 3D (D, H, W)
    if D > 1:
        pe_d = _get_1d_sincos_embed(D, dim_per_coord) # (D, dim_per_coord)
        pe_h = _get_1d_sincos_embed(H, dim_per_coord) # (H, dim_per_coord)
        pe_w = _get_1d_sincos_embed(W, dim_per_coord) # (W, dim_per_coord)

        # Create a grid and combine
        # Expand for broadcasting
        pe_d_expanded = pe_d.view(D, 1, 1, dim_per_coord)
        pe_h_expanded = pe_h.view(1, H, 1, dim_per_coord)
        pe_w_expanded = pe_w.view(1, 1, W, dim_per_coord)

        # Sum the embeddings (common practice for combining 1D PEs to higher dimensions)
        # Resulting shape: (D, H, W, dim_per_coord) for each.
        # Then concatenate along the feature dimension (last dim)
        spatial_pos_embed = torch.cat([
            pe_d_expanded.repeat(1, H, W, 1), # D_PE broadcast across H, W
            pe_h_expanded.repeat(D, 1, W, 1), # H_PE broadcast across D, W
            pe_w_expanded.repeat(D, H, 1, 1)  # W_PE broadcast across D, H
        ], dim=-1)
        
        # Flatten D, H, W and then ensure feature dim matches embed_dim
        spatial_pos_embed = spatial_pos_embed.view(D * H * W, -1)
    else: # For 2D (H, W) - e.g., first frame or image training
        pe_h = _get_1d_sincos_embed(H, dim_per_coord) # (H, dim_per_coord)
        pe_w = _get_1d_sincos_embed(W, dim_per_coord) # (W, dim_per_coord)
        
        pe_h_expanded = pe_h.view(H, 1, dim_per_coord)
        pe_w_expanded = pe_w.view(1, W, dim_per_coord)

        spatial_pos_embed = torch.cat([
            pe_h_expanded.repeat(1, W, 1),
            pe_w_expanded.repeat(H, 1, 1)
        ], dim=-1)

        spatial_pos_embed = spatial_pos_embed.view(H * W, -1)

    # Pad or truncate to ensure final embed_dim match
    if spatial_pos_embed.shape[-1] < embed_dim:
        padding = torch.zeros(spatial_pos_embed.shape[0], embed_dim - spatial_pos_embed.shape[-1], device=device)
        spatial_pos_embed = torch.cat([spatial_pos_embed, padding], dim=-1)
    elif spatial_pos_embed.shape[-1] > embed_dim:
        spatial_pos_embed = spatial_pos_embed[..., :embed_dim]

    return spatial_pos_embed.unsqueeze(0) # Add batch dimension (1, D*H*W, embed_dim)


def get_temporal_rope_embed(T: int, head_dim: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates Rotary Position Embeddings (RoPE) parameters for the temporal dimension.
    This function precomputes the cosine and sine frequencies.

    Args:
        T (int): The number of time steps (frames).
        head_dim (int): The dimension of a single attention head.
        device (torch.device): The computation device.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing (cos_freqs, sin_freqs)
        both of shape (T, head_dim // 2). These need to be expanded to (T, head_dim)
        and applied as complex exponentials in the attention mechanism.

    Note: The original design did not include `head_dim` as an argument.
          It has been added as it is essential for RoPE calculation.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"Head dimension {head_dim} must be even for RoPE.")

    # Create inverse frequencies (theta_i)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    
    # Create positions (m)
    t = torch.arange(T, dtype=torch.float32, device=device)
    
    # Compute frequencies (m * theta_i)
    freqs = torch.einsum("i,j->ij", t, inv_freq) # (T, head_dim // 2)

    # Compute cos and sin parts
    cos_freqs = freqs.cos() # (T, head_dim // 2)
    sin_freqs = freqs.sin() # (T, head_dim // 2)

    return cos_freqs, sin_freqs


def create_optimizer_scheduler(
    model: torch.nn.Module,
    config: Config,
    stage_idx: int
) -> Tuple[optim.Optimizer, LambdaLR]:
    """
    Creates the AdamW optimizer and a constant learning rate scheduler with warmup.

    Args:
        model (torch.nn.Module): The model whose parameters are to be optimized.
        config (Config): The global configuration object.
        stage_idx (int): The index of the current training stage (1, 2, or 3).

    Returns:
        Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        A tuple containing the optimizer and the learning rate scheduler.
    """
    stage_config: TrainingStageConfig = config.training[stage_idx]

    optimizer = optim.AdamW(
        model.parameters(),
        lr=stage_config.learning_rate,
        betas=(stage_config.optimizer_beta1, stage_config.optimizer_beta2),
        eps=stage_config.optimizer_epsilon,
        weight_decay=stage_config.weight_decay,
    )

    def lr_lambda(current_step: int):
        if current_step < stage_config.warmup_steps:
            return float(current_step) / float(max(1, stage_config.warmup_steps))
        return 1.0

    scheduler = LambdaLR(optimizer, lr_lambda)

    logger.info(f"Optimizer and scheduler created for stage {stage_idx} with LR: {stage_config.learning_rate}, Warmup Steps: {stage_config.warmup_steps}")
    return optimizer, scheduler


def configure_distributed(rank: Optional[int] = None, world_size: Optional[int] = None):
    """
    Sets up the distributed training environment using PyTorch Distributed Data Parallel (DDP).
    This function relies on environment variables (e.g., set by torchrun or accelerate).

    Args:
        rank (Optional[int]): The rank of the current process. If None, it tries to get from env.
        world_size (Optional[int]): The total number of processes. If None, it tries to get from env.
    """
    if dist.is_initialized():
        logger.info("Distributed environment already initialized.")
        return

    master_addr = os.environ.get("MASTER_ADDR")
    master_port = os.environ.get("MASTER_PORT")
    
    if master_addr and master_port:
        if rank is None:
            rank = int(os.environ.get("RANK", "0"))
        if world_size is None:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        if world_size > 1:
            dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
            torch.cuda.set_device(local_rank)
            dist.barrier()
            logger.info(f"Distributed training initialized: rank {rank}/{world_size}, local rank {local_rank}")
            if rank != 0:
                # Suppress output from non-main processes
                import builtins as __builtin__
                def print_pass(*args, **kwargs):
                    pass
                __builtin__.print = print_pass
        else:
            logger.info("Single process detected, DDP not initialized.")
    else:
        logger.info("MASTER_ADDR and/or MASTER_PORT not set. Skipping DDP initialization.")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: LambdaLR,
    step: int,
    stage_idx: int,
    config: Config,
    save_path: Optional[Union[str, Path]] = None
):
    """
    Saves the training state (model, optimizer, scheduler, current step, stage)
    to allow for resuming training or for later inference.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        scheduler (torch.optim.lr_scheduler.LambdaLR): The scheduler to save.
        step (int): The current training step.
        stage_idx (int): The current training stage index.
        config (Config): The global configuration object.
        save_path (Optional[Union[str, Path]]): Explicit path to save the checkpoint.
                                                  If None, it's generated from config.
    """
    # Only save from the main process in a distributed setup
    if dist.is_initialized() and dist.get_rank() != 0:
        return

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    checkpoint_dict = {
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "global_step": step,
        "current_stage_idx": stage_idx,
        "config": config, # Save full config for reproducibility
    }

    if save_path is None:
        checkpoint_dir = Path(config.data_paths.model_weights).parent
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        filename = f"checkpoint_stage_{stage_idx}_step_{step}.pth"
        save_path = checkpoint_dir / filename
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint_dict, save_path)
    logger.info(f"Checkpoint saved to {save_path}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LambdaLR] = None,
    config: Config,
    checkpoint_path: Optional[Union[str, Path]] = None
) -> Tuple[int, int]:
    """
    Loads a previously saved training state into the model, optimizer, and scheduler.

    Args:
        model (torch.nn.Module): The model to load weights into.
        optimizer (Optional[torch.optim.Optimizer]): The optimizer to load state into (optional).
        scheduler (Optional[torch.optim.lr_scheduler.LambdaLR]): The scheduler to load state into (optional).
        config (Config): The global configuration object.
        checkpoint_path (Optional[Union[str, Path]]): Explicit path to the checkpoint file.
                                                       If None, uses config.data_paths.model_weights.

    Returns:
        Tuple[int, int]: A tuple containing (global_step, current_stage_idx) loaded from the checkpoint.
                         Returns (0, 0) if no checkpoint is found or loaded.
    """
    if checkpoint_path is None:
        checkpoint_path = Path(config.data_paths.model_weights)
    else:
        checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        logger.warning(f"Checkpoint file not found at {checkpoint_path}. Starting from scratch.")
        return 0, 0

    map_location = torch.device(config.compute.device)
    try:
        checkpoint_dict = torch.load(checkpoint_path, map_location=map_location)
    except Exception as e:
        logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
        return 0, 0

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model.module.load_state_dict(checkpoint_dict["model_state_dict"])
    else:
        model.load_state_dict(checkpoint_dict["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint_dict:
        optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint_dict:
        scheduler.load_state_dict(checkpoint_dict["scheduler_state_dict"])

    global_step = checkpoint_dict.get("global_step", 0)
    current_stage_idx = checkpoint_dict.get("current_stage_idx", 0)
    
    logger.info(f"Checkpoint loaded from {checkpoint_path}. Resuming from stage {current_stage_idx}, step {global_step}.")
    return global_step, current_stage_idx


def get_default_device() -> torch.device:
    """
    Returns the appropriate default device (CUDA if available, otherwise CPU).
    This function checks system availability, not the config.compute.device preference.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# Example Usage (for testing and demonstration)
if __name__ == "__main__":
    print("--- Testing Utility Functions ---")
    
    # 1. Test downsample/upsample
    print("\nTesting downsample/upsample...")
    test_tensor_2d = torch.randn(1, 3, 64, 64)
    downsampled_2d = downsample(test_tensor_2d, 2, mode="bilinear")
    upsampled_2d = upsample(downsampled_2d, 2, mode="bilinear")
    print(f"Original 2D shape: {test_tensor_2d.shape}, Downsampled 2D shape: {downsampled_2d.shape}, Upsampled 2D shape: {upsampled_2d.shape}")
    assert downsampled_2d.shape == (1, 3, 32, 32)
    assert upsampled_2d.shape == (1, 3, 64, 64)

    test_tensor_3d = torch.randn(1, 3, 8, 64, 64)
    downsampled_3d = downsample(test_tensor_3d, 2, mode="trilinear")
    upsampled_3d = upsample(downsampled_3d, 2, mode="trilinear")
    print(f"Original 3D shape: {test_tensor_3d.shape}, Downsampled 3D shape: {downsampled_3d.shape}, Upsampled 3D shape: {upsampled_3d.shape}")
    assert downsampled_3d.shape == (1, 3, 4, 32, 32)
    assert upsampled_3d.shape == (1, 3, 8, 64, 64)

    # 2. Test get_spatial_pos_embed
    print("\nTesting get_spatial_pos_embed...")
    device_test = get_default_device()
    spatial_pe_2d = get_spatial_pos_embed(H=32, W=32, D=1, embed_dim=1024, device=device_test)
    print(f"Spatial PE (2D) shape: {spatial_pe_2d.shape}")
    assert spatial_pe_2d.shape == (1, 32*32, 1024)

    spatial_pe_3d = get_spatial_pos_embed(H=32, W=32, D=4, embed_dim=1024, device=device_test)
    print(f"Spatial PE (3D) shape: {spatial_pe_3d.shape}")
    assert spatial_pe_3d.shape == (1, 4*32*32, 1024)

    # 3. Test get_temporal_rope_embed
    print("\nTesting get_temporal_rope_embed...")
    cos_freqs, sin_freqs = get_temporal_rope_embed(T=8, head_dim=64, device=device_test)
    print(f"Temporal RoPE cos_freqs shape: {cos_freqs.shape}, sin_freqs shape: {sin_freqs.shape}")
    assert cos_freqs.shape == (8, 32)
    assert sin_freqs.shape == (8, 32)

    # 4. Test create_optimizer_scheduler (requires a dummy model and config)
    print("\nTesting create_optimizer_scheduler...")
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 1)
        def forward(self, x):
            return self.linear(x)

    dummy_model = DummyModel()
    dummy_config = Config() # Using the stub Config here
    optimizer, scheduler = create_optimizer_scheduler(dummy_model, dummy_config, 1)
    print(f"Optimizer: {type(optimizer).__name__}, LR Scheduler: {type(scheduler).__name__}")
    assert isinstance(optimizer, optim.AdamW)
    assert isinstance(scheduler, LambdaLR)
    
    initial_lr = optimizer.param_groups[0]['lr']
    for step in range(dummy_config.training[1].warmup_steps + 5):
        optimizer.step()
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        if step < dummy_config.training[1].warmup_steps:
            assert current_lr == initial_lr * (step + 1) / dummy_config.training[1].warmup_steps
        else:
            assert current_lr == initial_lr
    print(f"LR scheduling seems correct. Final LR: {current_lr}")

    # 5. Test get_default_device
    print("\nTesting get_default_device...")
    default_dev = get_default_device()
    print(f"Default device: {default_dev}")
    assert isinstance(default_dev, torch.device)

    # 6. Test save/load_checkpoint (mock DDP, no actual distributed setup)
    print("\nTesting save/load_checkpoint...")
    # Create a dummy model for saving
    dummy_model_save = DummyModel().to(device_test)
    dummy_optimizer_save, dummy_scheduler_save = create_optimizer_scheduler(dummy_model_save, dummy_config, 1)
    
    save_dir = Path("./temp_checkpoints")
    save_dir.mkdir(exist_ok=True)
    temp_checkpoint_path = save_dir / "test_checkpoint.pth"

    save_checkpoint(dummy_model_save, dummy_optimizer_save, dummy_scheduler_save, 100, 1, dummy_config, temp_checkpoint_path)
    
    # Create new dummy model/optimizer/scheduler for loading
    dummy_model_load = DummyModel().to(device_test)
    dummy_optimizer_load, dummy_scheduler_load = create_optimizer_scheduler(dummy_model_load, dummy_config, 1)

    # Check initial weights are different
    assert not torch.equal(dummy_model_save.linear.weight.data, dummy_model_load.linear.weight.data)

    global_step_loaded, stage_idx_loaded = load_checkpoint(
        dummy_model_load, dummy_optimizer_load, dummy_scheduler_load, dummy_config, temp_checkpoint_path
    )
    
    print(f"Loaded global_step: {global_step_loaded}, current_stage_idx: {stage_idx_loaded}")
    assert global_step_loaded == 100
    assert stage_idx_loaded == 1
    
    # Check if weights are now the same
    assert torch.equal(dummy_model_save.linear.weight.data, dummy_model_load.linear.weight.data)
    
    # Cleanup
    import shutil
    shutil.rmtree(save_dir)
    print("Checkpoint save/load test passed and cleaned up.")

    print("\nAll utility tests completed.")


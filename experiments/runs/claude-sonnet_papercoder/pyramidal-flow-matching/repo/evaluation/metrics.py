```python
## evaluation/metrics.py
"""Evaluation metrics and orchestration for Pyramidal Flow Matching.

Provides two layers of evaluation functionality:

1. **Standalone utility functions** (module-level):
   - compute_fid: FID between real and generated image directories
   - compute_fvd: Fréchet Video Distance using I3D features
   - compute_psnr: Peak Signal-to-Noise Ratio
   - compute_ssim: Structural Similarity Index
   - save_video_grid: Save video tensors as MP4

2. **MetricEvaluator class**:
   Orchestrates VBench (16 dimensions), EvalCrafter (~17 metrics),
   and ablation studies (FID on MS-COCO, FVD on MSR-VTT) as described
   in the paper (Sections 4.1, 4.3, 4.4, Appendix C.2).

Paper alignment:
    - VBench: Table 1, Table 5 — Total=81.72, Quality=84.74, Semantic=69.62
    - EvalCrafter: Table 2, Table 6 — VQAA=86.09, VQAT=88.31, IS=18.49
    - FID ablation: Section 4.4, Fig. 7 — 3K MS-COCO prompts
    - FVD ablation: Appendix C.2, Fig. 12b — MSR-VTT benchmark

Config references (configs/default.yaml):
    eval.vbench.enabled: true
    eval.vbench.prompts_path: "data/eval/vbench_prompts.txt"
    eval.vbench.output_dir: "outputs/vbench"
    eval.vbench.num_frames: 121
    eval.vbench.resolution: [768, 768]
    eval.vbench.fps: 24
    eval.evalcrafter.enabled: true
    eval.evalcrafter.prompts_path: "data/eval/evalcrafter_prompts.txt"
    eval.evalcrafter.output_dir: "outputs/evalcrafter"
    eval.fid_coco.enabled: false
    eval.fid_coco.num_prompts: 3000
    eval.fid_coco.coco_ref_dir: "data/coco/val2017"
    eval.fid_coco.output_dir: "outputs/fid_coco"
    eval.fvd_msrvtt.enabled: false
    eval.fvd_msrvtt.msrvtt_dir: "data/msrvtt"
    eval.fvd_msrvtt.output_dir: "outputs/fvd_msrvtt"

Usage:
    from evaluation.metrics import MetricEvaluator, compute_fid, compute_fvd

    evaluator = MetricEvaluator(config)
    vbench_results = evaluator.evaluate_vbench(sampler)
    evalcrafter_results = evaluator.evaluate_evalcrafter(sampler)
    fid = evaluator.evaluate_ablation_fid(sampler, coco_prompts)
    fvd = evaluator.evaluate_ablation_fvd(sampler, msr_vtt_dir)
"""

import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from utils.distributed import is_main_process
from utils.logging import get_logger

## ---------------------------------------------------------------------------
## Module-level logger
## ---------------------------------------------------------------------------
logger = get_logger(__name__)

## ---------------------------------------------------------------------------
## Optional dependency availability flags
## ---------------------------------------------------------------------------
_PIL_AVAILABLE: bool = False
_IMAGEIO_AVAILABLE: bool = False
_SKIMAGE_AVAILABLE: bool = False
_PYTORCH_FID_AVAILABLE: bool = False
_SCIPY_AVAILABLE: bool = False
_TQDM_AVAILABLE: bool = False
_DECORD_AVAILABLE: bool = False

try:
    from PIL import Image  # type: ignore[import]
    _PIL_AVAILABLE = True
except ImportError:
    logger.warning(
        "Pillow not available. Image saving for FID will be disabled. "
        "Install with: pip install Pillow==10.3.0"
    )

try:
    import imageio  # type: ignore[import]
    import imageio_ffmpeg  # type: ignore[import]
    _IMAGEIO_AVAILABLE = True
except ImportError:
    logger.warning(
        "imageio or imageio-ffmpeg not available. Video saving will be disabled. "
        "Install with: pip install imageio==2.34.0 imageio-ffmpeg==0.4.9"
    )

try:
    from skimage.metrics import structural_similarity  # type: ignore[import]
    _SKIMAGE_AVAILABLE = True
except ImportError:
    logger.warning(
        "scikit-image not available. SSIM computation will be disabled. "
        "Install with: pip install scikit-image==0.23.2"
    )

try:
    from pytorch_fid import fid_score as pytorch_fid_score  # type: ignore[import]
    _PYTORCH_FID_AVAILABLE = True
except ImportError:
    logger.warning(
        "pytorch-fid not available. FID computation will be disabled. "
        "Install with: pip install pytorch-fid==0.3.0"
    )

try:
    from scipy.linalg import sqrtm  # type: ignore[import]
    _SCIPY_AVAILABLE = True
except ImportError:
    logger.warning(
        "scipy not available. FVD matrix square root will be disabled. "
        "Install with: pip install scipy==1.13.0"
    )

try:
    from tqdm import tqdm  # type: ignore[import]
    _TQDM_AVAILABLE = True
except ImportError:
    # Fallback: identity wrapper
    def tqdm(iterable: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        """No-op tqdm fallback when tqdm is not installed."""
        return iterable

try:
    import decord  # type: ignore[import]
    decord.bridge.set_bridge("torch")
    _DECORD_AVAILABLE = True
except ImportError:
    logger.warning(
        "decord not available. MSR-VTT video loading will be disabled. "
        "Install with: pip install decord==0.6.0"
    )

## ---------------------------------------------------------------------------
## Constants
## ---------------------------------------------------------------------------
_I3D_FEATURE_DIM: int = 2048          # I3D pool feature dimension
_I3D_INPUT_SIZE: int = 224            # I3D expected spatial size
_I3D_MIN_FRAMES: int = 8              # I3D minimum temporal frames
_I3D_CLIP_FRAMES: int = 16            # Standard I3D clip length
_FID_INCEPTION_DIMS: int = 2048       # Inception-v3 pool layer dims
_FID_MIN_SAMPLES_WARNING: int = 2048  # Warn if fewer samples for FID
_FVD_MIN_SAMPLES_WARNING: int = 10    # Warn if fewer samples for FVD
_VIDEO_EXTENSIONS: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".webm")
_IMAGE_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
_DEFAULT_FID_BATCH_SIZE: int = 64
_DEFAULT_FVD_BATCH_SIZE: int = 8
_DEFAULT_VIDEO_GRID_FPS: int = 24
_DEFAULT_COCO_NUM_PROMPTS: int = 3000
_DEFAULT_MSRVTT_NUM_VIDEOS: int = 1000


## ---------------------------------------------------------------------------
## Standalone utility functions
## ---------------------------------------------------------------------------


def compute_fid(
    real_dir: str,
    fake_dir: str,
    device: str = "cuda",
    batch_size: int = _DEFAULT_FID_BATCH_SIZE,
) -> float:
    """Computes Fréchet Inception Distance between real and generated images.

    Uses the pytorch-fid library which extracts Inception-v3 pool features
    (2048-dim) from both directories and computes the Fréchet distance between
    the resulting Gaussian distributions.

    Used in the spatial pyramid ablation (Section 4.4, Fig. 7) to compare
    convergence speed of pyramidal flow vs. standard flow matching on MS-COCO.

    Args:
        real_dir: Path to directory containing real reference images.
            For the paper's ablation: MS-COCO val2017 images.
            Must contain at least one .png or .jpg file.
        fake_dir: Path to directory containing generated images.
            For the paper's ablation: 3K images generated from COCO prompts.
            Must contain at least one .png or .jpg file.
        device: PyTorch device string for Inception-v3 inference.
            Defaults to "cuda". Falls back to "cpu" if CUDA unavailable.
        batch_size: Batch size for Inception-v3 feature extraction.
            Defaults to 64. Reduce if GPU OOM occurs.

    Returns:
        Scalar FID value. Lower is better. Returns float('inf') if
        computation fails (e.g., pytorch-fid not installed, empty dirs).

    Raises:
        ValueError: If either directory does not exist or contains no images.

    Example:
        >>> fid = compute_fid(
        ...     real_dir="data/coco/val2017",
        ...     fake_dir="outputs/fid_generated",
        ...     device="cuda",
        ... )
        >>> print(f"FID: {fid:.4f}")
    """
    if not _PYTORCH_FID_AVAILABLE:
        logger.error(
            "pytorch-fid not available. Cannot compute FID. "
            "Install with: pip install pytorch-fid==0.3.0"
        )
        return float("inf")

    # ----------------------------------------------------------------
    # Validate directories
    # ----------------------------------------------------------------
    for dir_path, dir_name in [(real_dir, "real_dir"), (fake_dir, "fake_dir")]:
        if not os.path.isdir(dir_path):
            raise ValueError(
                f"{dir_name}='{dir_path}' does not exist or is not a directory. "
                f"Ensure the path points to a valid image directory."
            )

        # Count image files
        image_files: List[str] = [
            f for f in os.listdir(dir_path)
            if Path(f).suffix.lower() in _IMAGE_EXTENSIONS
        ]
        if len(image_files) == 0:
            raise ValueError(
                f"{dir_name}='{dir_path}' contains no image files "
                f"(expected extensions: {_IMAGE_EXTENSIONS}). "
                f"Ensure images are saved before calling compute_fid."
            )

        if len(image_files) < _FID_MIN_SAMPLES_WARNING:
            logger.warning(
                "%s='%s' contains only %d images. "
                "FID statistics may be unreliable with fewer than %d samples. "
                "The paper uses 3K prompts for the MS-COCO ablation.",
                dir_name,
                dir_path,
                len(image_files),
                _FID_MIN_SAMPLES_WARNING,
            )

    # ----------------------------------------------------------------
    # Validate device
    # ----------------------------------------------------------------
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning(
            "CUDA not available. Falling back to CPU for FID computation. "
            "This will be significantly slower."
        )
        device = "cpu"

    # ----------------------------------------------------------------
    # Compute FID using pytorch-fid
    # ----------------------------------------------------------------
    logger.info(
        "Computing FID: real_dir='%s', fake_dir='%s', "
        "device=%s, batch_size=%d",
        real_dir,
        fake_dir,
        device,
        batch_size,
    )

    try:
        fid_value: float = pytorch_fid_score.calculate_fid_given_paths(
            paths=[real_dir, fake_dir],
            batch_size=batch_size,
            device=device,
            dims=_FID_INCEPTION_DIMS,
        )
        logger.info("FID computed: %.4f", fid_value)
        return float(fid_value)

    except Exception as exc:
        logger.error(
            "FID computation failed: %s. "
            "Returning float('inf') as fallback.",
            exc,
        )
        return float("inf")


def compute_fvd(
    real_videos: Union[Tensor, List[str]],
    fake_videos: Union[Tensor, List[str]],
    device: str = "cuda",
    batch_size: int = _DEFAULT_FVD_BATCH_SIZE,
) -> float:
    """Computes Fréchet Video Distance using I3D features.

    Extracts I3D (Inflated 3D ConvNet) features from real and generated
    videos, then computes the Fréchet distance between the resulting
    Gaussian distributions in the 2048-dimensional feature space.

    Used in the temporal pyramid ablation (Appendix C.2, Fig. 12b) to
    compare pyramidal flow vs. full-sequence diffusion on MSR-VTT.

    Args:
        real_videos: Either a Tensor of shape [N, C, T, H, W] with values
            in [0, 1], or a list of video file paths. Videos are resized
            to 224×224 and clipped to 16 frames for I3D compatibility.
        fake_videos: Either a Tensor of shape [N, C, T, H, W] with values
            in [0, 1], or a list of video file paths. Must have the same
            number of samples N as real_videos.
        device: PyTorch device string for I3D inference. Defaults to "cuda".
        batch_size: Batch size for I3D feature extraction. Defaults to 8.
            Reduce if GPU OOM occurs (I3D is memory-intensive).

    Returns:
        Scalar FVD value. Lower is better. Returns float('inf') if
        computation fails (e.g., scipy not installed, too few samples).

    Example:
        >>> real = torch.rand(100, 3, 16, 256, 256)  # [N, C, T, H, W]
        >>> fake = torch.rand(100, 3, 16, 256, 256)
        >>> fvd = compute_fvd(real, fake, device="cuda")
        >>> print(f"FVD: {fvd:.4f}")
    """
    if not _SCIPY_AVAILABLE:
        logger.error(
            "scipy not available. Cannot compute FVD matrix square root. "
            "Install with: pip install scipy==1.13.0"
        )
        return float("inf")

    # ----------------------------------------------------------------
    # Validate device
    # ----------------------------------------------------------------
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning(
            "CUDA not available. Falling back to CPU for FVD computation."
        )
        device = "cpu"

    torch_device: torch.device = torch.device(device)

    # ----------------------------------------------------------------
    # Load videos if file paths provided
    # ----------------------------------------------------------------
    real_tensor: Tensor = _load_videos_for_fvd(real_videos, torch_device)
    fake_tensor: Tensor = _load_videos_for_fvd(fake_videos, torch_device)

    n_real: int = real_tensor.shape[0]
    n_fake: int = fake_tensor.shape[0]

    if n_real < _FVD_MIN_SAMPLES_WARNING or n_fake < _FVD_MIN_SAMPLES_WARNING:
        logger.warning(
            "FVD may be unreliable with fewer than %d samples. "
            "Got n_real=%d, n_fake=%d.",
            _FVD_MIN_SAMPLES_WARNING,
            n_real,
            n_fake,
        )

    logger.info(
        "Computing FVD: n_real=%d, n_fake=%d, device=%s, batch_size=%d",
        n_real,
        n_fake,
        device,
        batch_size,
    )

    # ----------------------------------------------------------------
    # Load I3D model for feature extraction
    # ----------------------------------------------------------------
    i3d_model: Optional[torch.nn.Module] = _load_i3d_model(torch_device)
    if i3d_model is None:
        logger.error(
            "Failed to load I3D model. Cannot compute FVD. "
            "Returning float('inf') as fallback."
        )
        return float("inf")

    # ----------------------------------------------------------------
    # Extract I3D features
    # ----------------------------------------------------------------
    try:
        real_features: np.ndarray = _extract_i3d_features(
            real_tensor, i3d_model, torch_device, batch_size
        )
        fake_features: np.ndarray = _extract_i3d_features(
            fake_tensor, i3d_model, torch_device, batch_size
        )
    except Exception as exc:
        logger.error(
            "I3D feature extraction failed: %s. "
            "Returning float('inf') as fallback.",
            exc,
        )
        return float("inf")

    # ----------------------------------------------------------------
    # Compute Fréchet distance
    # ----------------------------------------------------------------
    try:
        fvd_value: float = _frechet_distance(real_features, fake_features)
        logger.info("FVD computed: %.4f", fvd_value)
        return fvd_value
    except Exception as exc:
        logger.error(
            "Fréchet distance computation failed: %s. "
            "Returning float('inf') as fallback.",
            exc,
        )
        return float("inf")


def compute_psnr(
    pred: Tensor,
    target: Tensor,
) -> float:
    """Computes Peak Signal-to-Noise Ratio between predicted and target tensors.

    Standard PSNR formula: PSNR = 10 * log10(MAX^2 / MSE)
    where MAX = 1.0 (assuming inputs normalized to [0, 1]).

    Args:
        pred: Predicted tensor of shape [B, C, H, W] or [B, C, T, H, W].
            Values should be in [0, 1].
        target: Ground truth tensor of the same shape as pred.
            Values should be in [0, 1].

    Returns:
        Scalar PSNR value in dB. Returns float('inf') if MSE is zero
        (perfect reconstruction). Higher is better.

    Example:
        >>> pred = torch.rand(4, 3, 256, 256)
        >>> target = torch.rand(4, 3, 256, 256)
        >>> psnr = compute_psnr(pred, target)
        >>> print(f"PSNR: {psnr:.2f} dB")
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have the same shape. "
            f"Got pred.shape={tuple(pred.shape)}, "
            f"target.shape={tuple(target.shape)}."
        )

    # Clamp to [0, 1] to handle minor floating point violations
    pred_clamped: Tensor = pred.clamp(0.0, 1.0)
    target_clamped: Tensor = target.clamp(0.0, 1.0)

    mse: Tensor = F.mse_loss(pred_clamped, target_clamped, reduction="mean")
    mse_val: float = mse.item()

    if mse_val == 0.0:
        return float("inf")

    # PSNR = 10 * log10(1.0 / MSE) for MAX=1.0
    psnr: float = 10.0 * math.log10(1.0 / mse_val)
    return psnr


def compute_ssim(
    pred: Tensor,
    target: Tensor,
) -> float:
    """Computes Structural Similarity Index between predicted and target tensors.

    Computes SSIM per image/frame and averages over the batch and temporal
    dimensions. Uses scikit-image's implementation with default parameters
    (window size 7, data range 1.0).

    Args:
        pred: Predicted tensor of shape [B, C, H, W] or [B, C, T, H, W].
            Values should be in [0, 1].
        target: Ground truth tensor of the same shape as pred.
            Values should be in [0, 1].

    Returns:
        Scalar SSIM value in [0, 1]. Higher is better. Returns 0.0 if
        scikit-image is not available.

    Example:
        >>> pred = torch.rand(4, 3, 256, 256)
        >>> target = torch.rand(4, 3, 256, 256)
        >>> ssim = compute_ssim(pred, target)
        >>> print(f"SSIM: {ssim:.4f}")
    """
    if not _SKIMAGE_AVAILABLE:
        logger.warning(
            "scikit-image not available. SSIM computation disabled. "
            "Install with: pip install scikit-image==0.23.2"
        )
        return 0.0

    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have the same shape. "
            f"Got pred.shape={tuple(pred.shape)}, "
            f"target.shape={tuple(target.shape)}."
        )

    # Convert to numpy for scikit-image
    pred_np: np.ndarray = pred.detach().cpu().float().numpy()
    target_np: np.ndarray = target.detach().cpu().float().numpy()

    # Clamp to [0, 1]
    pred_np = np.clip(pred_np, 0.0, 1.0)
    target_np = np.clip(target_np, 0.0, 1.0)

    # Handle 5D video tensors: flatten B and T into batch dimension
    if pred_np.ndim == 5:
        # [B, C, T, H, W] → [B*T, C, H, W]
        B, C, T, H, W = pred_np.shape
        pred_np = pred_np.transpose(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        target_np = target_np.transpose(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

    # pred_np: [N, C, H, W]
    N: int = pred_np.shape[0]
    ssim_values: List[float] = []

    for i in range(N):
        # scikit-image expects [H, W, C] for multichannel
        pred_hwc: np.ndarray = pred_np[i].transpose(1, 2, 0)   # [H, W, C]
        target_hwc: np.ndarray = target_np[i].transpose(1, 2, 0)

        try:
            ssim_val: float = float(
                structural_similarity(
                    pred_hwc,
                    target_hwc,
                    data_range=1.0,
                    channel_axis=-1,
                    win_size=7,
                )
            )
            ssim_values.append(ssim_val)
        except Exception as exc:
            logger.warning(
                "SSIM computation failed for sample %d: %s. Skipping.",
                i,
                exc,
            )

    if not ssim_values:
        return 0.0

    return float(np.mean(ssim_values))


def save_video_grid(
    videos: List[Tensor],
    path: str,
    fps: int = _DEFAULT_VIDEO_GRID_FPS,
) -> None:
    """Saves a list of video tensors as a tiled MP4 grid.

    Tiles multiple videos into a grid layout (ceil(sqrt(N)) columns) and
    saves as an MP4 file using imageio-ffmpeg. Only executes on the main
    process (rank 0) to avoid duplicate writes in distributed settings.

    Args:
        videos: List of video tensors, each of shape [C, T, H, W] with
            values in [0, 1] float. All videos must have the same shape.
        path: Output file path for the MP4 video. Parent directory is
            created if it does not exist.
        fps: Frames per second for the output video. Defaults to 24
            (paper's generation fps, config.inference.default_fps).

    Example:
        >>> videos = [torch.rand(3, 121, 768, 768) for _ in range(4)]
        >>> save_video_grid(videos, "outputs/sample_grid.mp4", fps=24)
    """
    if not is_main_process():
        return

    if not _IMAGEIO_AVAILABLE:
        logger.warning(
            "imageio or imageio-ffmpeg not available. "
            "Cannot save video grid. "
            "Install with: pip install imageio==2.34.0 imageio-ffmpeg==0.4.9"
        )
        return

    if not videos:
        logger.warning("save_video_grid called with empty video list. Skipping.")
        return

    # ----------------------------------------------------------------
    # Validate and normalize video tensors
    # ----------------------------------------------------------------
    # All videos must have the same shape [C, T, H, W]
    ref_shape: Tuple[int, ...] = tuple(videos[0].shape)
    for i, v in enumerate(videos):
        if tuple(v.shape) != ref_shape:
            logger.warning(
                "Video %d has shape %s, expected %s. "
                "Skipping mismatched video.",
                i,
                tuple(v.shape),
                ref_shape,
            )
            videos = [v for j, v in enumerate(videos) if tuple(v.shape) == ref_shape]
            break

    if not videos:
        logger.warning("No valid videos to save after shape validation.")
        return

    C: int = videos[0].shape[0]
    T: int = videos[0].shape[1]
    H: int = videos[0].shape[2]
    W: int = videos[0].shape[3]
    N: int = len(videos)

    # ----------------------------------------------------------------
    # Compute grid layout
    # ----------------------------------------------------------------
    num_cols: int = max(1, math.ceil(math.sqrt(N)))
    num_rows: int = max(1, math.ceil(N / num_cols))

    # Pad video list to fill the grid
    while len(videos) < num_rows * num_cols:
        videos.append(torch.zeros(C, T, H, W))

    # ----------------------------------------------------------------
    # Create output directory
    # ----------------------------------------------------------------
    output_path: Path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # Convert videos to uint8 numpy arrays and tile into grid
    # ----------------------------------------------------------------
    # Each video: [C, T, H, W] float [0,1] → [T, H, W, C] uint8 [0,255]
    video_arrays: List[np.ndarray] = []
    for v in videos:
        v_np: np.ndarray = (
            v.detach().cpu().float().clamp(0.0, 1.0).numpy()
            * 255.0
        ).astype(np.uint8)
        # [C, T, H, W] → [T, H, W, C]
        v_np = v_np.transpose(1, 2, 3, 0)
        video_arrays.append(v_np)

    # Build grid frames: for each timestep, tile all videos
    grid_frames: List[np.ndarray] = []
    for t in range(T):
        # Collect frames at timestep t from all videos
        row_frames: List[np.ndarray] = []
        for row_idx in range(num_rows):
            col_frames: List[np.ndarray] = []
            for col_idx in range(num_cols):
                vid_idx: int = row_idx * num_cols + col_idx
                frame: np.ndarray = video_arrays[vid_idx][t]  # [H, W, C]
                col_frames.append(frame)
            # Concatenate columns horizontally: [H, W*num_cols, C]
            row_frame: np.ndarray = np.concatenate(col_frames, axis=1)
            row_frames.append(row_frame)
        # Concatenate rows vertically: [H*num_rows, W*num_cols, C]
        grid_frame: np.ndarray = np.concatenate(row_frames, axis=0)
        grid_frames.append(grid_frame)

    # ----------------------------------------------------------------
    # Write MP4 using imageio
    # ----------------------------------------------------------------
    try:
        writer = imageio.get_writer(
            str(output_path),
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=1,  # Allow non-multiple-of-16 dimensions
        )
        for frame in grid_frames:
            writer.append_data(frame)
        writer.close()

        logger.info(
            "Video grid saved: path='%s', N=%d, T=%d, H=%d, W=%d, fps=%d, "
            "grid=%dx%d",
            str(output_path),
            N,
            T,
            H,
            W,
            fps,
            num_rows,
            num_cols,
        )
    except Exception as exc:
        logger.error(
            "Failed to save video grid to '%s': %s",
            str(output_path),
            exc,
        )


## ---------------------------------------------------------------------------
## Private helpers for FVD computation
## ---------------------------------------------------------------------------


def _load_videos_for_fvd(
    videos: Union[Tensor, List[str]],
    device: torch.device,
) -> Tensor:
    """Loads and preprocesses videos for I3D feature extraction.

    Handles both Tensor inputs and file path lists. Resizes to 224×224,
    clips to I3D_CLIP_FRAMES frames, and normalizes to [0, 1].

    Args:
        videos: Either a Tensor [N, C, T, H, W] in [0, 1], or a list of
            video file paths.
        device: Target device for the output tensor.

    Returns:
        Tensor of shape [N, C, I3D_CLIP_FRAMES, 224, 224] in [0, 1].
    """
    if isinstance(videos, Tensor):
        # Already a tensor: preprocess in-place
        return _preprocess_videos_for_i3d(videos, device)

    # Load from file paths
    if not _DECORD_AVAILABLE:
        logger.error(
            "decord not available. Cannot load videos from file paths for FVD. "
            "Pass video tensors directly or install decord."
        )
        # Return empty tensor
        return torch.zeros(0, 3, _I3D_CLIP_FRAMES, _I3D_INPUT_SIZE, _I3D_INPUT_SIZE)

    loaded_videos: List[Tensor] = []
    for video_path in videos:
        try:
            vr = decord.VideoReader(video_path)
            total_frames: int = len(vr)
            # Sample I3D_CLIP_FRAMES uniformly
            frame_indices: List[int] = _sample_frame_indices(
                total_frames, _I3D_CLIP_FRAMES
            )
            frames: Tensor = vr.get_batch(frame_indices)  # [T, H, W, C]
            # [T, H, W, C] → [C, T, H, W], normalize to [0, 1]
            frames = frames.perm
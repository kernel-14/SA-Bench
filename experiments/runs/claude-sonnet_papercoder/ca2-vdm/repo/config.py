## config.py
"""Configuration module for Ca2-VDM.

This module defines the Config dataclass that centralizes all hyperparameters
and paths used across the entire Ca2-VDM codebase. It loads from a YAML file
and resolves task/stage/model-type-specific settings into a flat interface.

Paper: Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal
Generation and Cache Sharing.
"""

import dataclasses
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from omegaconf import OmegaConf


@dataclass(frozen=True)
class Config:
    """Flat configuration dataclass for Ca2-VDM.

    All hyperparameters are sourced from config.yaml and the paper.
    Use Config.from_config(path) to construct from a YAML file.

    Attributes:
        task: Task type, either 't2v' or 'video_prediction'.
        stage: Training stage (1 or 2), only relevant for task='t2v'.
        model_type: Model variant: 'ca2vdm', 'os_fix', or 'os_ext'.
        resolution: Spatial resolution of video frames (e.g., 256).
        chunk_len: Number of frames per autoregressive chunk (l).
        p_max: Maximum number of conditional (prefix) frames (P_max).
        l_train: Maximum training sequence length (L_train).
        n_max: Maximum prefix multiplier; P in {1, 1+l, ..., 1+n_max*l}.
        prefix_spatial_len: Sub-prefix length P' for spatial attention.
        batch_size: Training batch size.
        lr: Learning rate for AdamW optimizer.
        weight_decay: Weight decay for AdamW optimizer.
        num_steps: Total number of training steps.
        use_prefix: Whether to use clean prefix frames during training.
        ddpm_T: Total diffusion timesteps T.
        beta_start: Starting beta value for noise schedule.
        beta_end: Ending beta value for noise schedule.
        learn_sigma: Whether to learn the noise variance (improved DDPM).
        inference_steps: Number of denoising steps at inference.
        cfg_scale: Classifier-free guidance scale for T2V.
        clean_prefix_timestep: Timestep index for clean prefix (always 0).
        cyclic_tpe_enabled: Whether to use cyclic temporal positional embeddings.
        spatial_cache_chunks: Number of spatial KV-cache chunks to store (1).
        num_ar_steps: Number of autoregressive steps for evaluation.
        device: Compute device ('cuda' or 'cpu').
        output_dir: Directory for saving outputs.
        checkpoint_dir: Directory for saving/loading checkpoints.
        data_root: Root directory for datasets.
        dataset_name: Name of the dataset to use.
        model_dim: Hidden dimension of the Transformer model.
        num_heads: Number of attention heads.
        num_layers: Number of Transformer blocks.
        mlp_ratio: MLP hidden dimension multiplier in Transformer blocks.
        vae_channels: Number of VAE latent channels (4 for SD VAE).
        vae_downsample: Spatial downsampling factor of VAE (8x).
        context_dim: Dimension of text encoder output (T5).
        dropout: Dropout rate for attention layers.
        use_cross_attn: Whether to use visual-text cross attention.
        vae_path: Path to pretrained Stable Diffusion VAE.
        t5_path: Path to pretrained T5 text encoder.
        opensora_ckpt: Path to Open-Sora v1.0 checkpoint for initialization.
        i3d_ckpt: Path to pretrained I3D model for FVD evaluation.
        ucf101_prompts: Path to UCF-101 PYoCo descriptive prompts JSON.
        log_every: Log training loss every N steps.
        save_every: Save checkpoint every N steps.
        use_wandb: Whether to use Weights & Biases logging.
        use_tensorboard: Whether to use TensorBoard logging.
        mixed_precision: Mixed precision type ('bf16', 'fp16', or 'no').
        dataloader_num_workers: Number of DataLoader worker processes.
        pin_memory: Whether to pin memory in DataLoader.
        num_gpus: Number of GPUs to use for training.
    """

    # ── Task and model variant ────────────────────────────────────────────────
    task: str = "video_prediction"
    stage: int = 2
    model_type: str = "ca2vdm"

    # ── Video and sequence dimensions ─────────────────────────────────────────
    resolution: int = 256
    chunk_len: int = 8
    p_max: int = 25
    l_train: int = 33
    n_max: int = 3
    prefix_spatial_len: int = 3

    # ── Training hyperparameters ──────────────────────────────────────────────
    batch_size: int = 8
    lr: float = 2e-5
    weight_decay: float = 0.01
    num_steps: int = 11000
    use_prefix: bool = True

    # ── Diffusion schedule ────────────────────────────────────────────────────
    ddpm_T: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    learn_sigma: bool = True

    # ── Inference settings ────────────────────────────────────────────────────
    inference_steps: int = 100
    cfg_scale: float = 7.5
    clean_prefix_timestep: int = 0
    cyclic_tpe_enabled: bool = True
    spatial_cache_chunks: int = 1
    num_ar_steps: int = 6

    # ── Hardware and runtime ──────────────────────────────────────────────────
    device: str = "cuda"

    # ── Paths ─────────────────────────────────────────────────────────────────
    output_dir: str = "outputs/"
    checkpoint_dir: str = "checkpoints/"
    data_root: str = "data/"
    dataset_name: str = "skytimelapse"
    vae_path: str = "pretrained/sd-vae"
    t5_path: str = "pretrained/t5-large"
    opensora_ckpt: str = "pretrained/opensora-v1"
    i3d_ckpt: str = "pretrained/i3d_kinetics.pt"
    ucf101_prompts: str = "data/ucf101/pyoco_prompts.json"

    # ── Model architecture ────────────────────────────────────────────────────
    model_dim: int = 1152
    num_heads: int = 16
    num_layers: int = 28
    mlp_ratio: float = 4.0
    vae_channels: int = 4
    vae_downsample: int = 8
    context_dim: int = 1024
    dropout: float = 0.0
    use_cross_attn: bool = False

    # ── Logging and checkpointing ─────────────────────────────────────────────
    log_every: int = 100
    save_every: int = 1000
    use_wandb: bool = False
    use_tensorboard: bool = True

    # ── Hardware configuration ────────────────────────────────────────────────
    mixed_precision: str = "bf16"
    dataloader_num_workers: int = 4
    pin_memory: bool = True
    num_gpus: int = 1

    @classmethod
    def from_config(
        cls,
        path: str,
        task: Optional[str] = None,
        stage: int = 2,
        model_type: str = "ca2vdm",
    ) -> "Config":
        """Load and resolve configuration from a YAML file.

        Reads the YAML config, selects the correct task/stage/model_type
        section, and returns a fully populated flat Config instance.

        Args:
            path: Path to the YAML configuration file.
            task: Override the task from YAML ('t2v' or 'video_prediction').
                  If None, reads from the YAML top-level 'task' key.
            stage: Training stage (1 or 2). Only relevant for task='t2v'.
                   Defaults to 2.
            model_type: Model variant ('ca2vdm', 'os_fix', or 'os_ext').
                        Defaults to 'ca2vdm'.

        Returns:
            A fully populated and validated Config instance.

        Raises:
            FileNotFoundError: If the config YAML file does not exist.
            ValueError: If any configuration invariant is violated.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")

        raw_cfg = OmegaConf.load(path)
        cfg_dict: Dict[str, Any] = OmegaConf.to_container(raw_cfg, resolve=True)

        # Resolve task: CLI override takes precedence over YAML
        resolved_task: str = task if task is not None else cfg_dict.get("task", "video_prediction")

        # Validate task, stage, model_type before resolution
        valid_tasks = {"t2v", "video_prediction"}
        valid_stages = {1, 2}
        valid_model_types = {"ca2vdm", "os_fix", "os_ext"}

        if resolved_task not in valid_tasks:
            raise ValueError(
                f"Invalid task '{resolved_task}'. Must be one of {valid_tasks}."
            )
        if stage not in valid_stages:
            raise ValueError(
                f"Invalid stage '{stage}'. Must be one of {valid_stages}."
            )
        if model_type not in valid_model_types:
            raise ValueError(
                f"Invalid model_type '{model_type}'. Must be one of {valid_model_types}."
            )

        # Resolve all fields from the nested YAML structure
        resolved = _resolve_task_config(cfg_dict, resolved_task, stage, model_type)

        # Determine device
        device: str = "cuda" if torch.cuda.is_available() else "cpu"

        # Build the Config instance
        config = cls(
            # Task and model variant
            task=resolved_task,
            stage=stage,
            model_type=model_type,
            # Video and sequence dimensions
            resolution=resolved["resolution"],
            chunk_len=resolved["chunk_len"],
            p_max=resolved["p_max"],
            l_train=resolved["l_train"],
            n_max=resolved["n_max"],
            prefix_spatial_len=cfg_dict["ca2vdm"]["prefix_spatial_len"],
            # Training hyperparameters
            batch_size=resolved["batch_size"],
            lr=resolved["lr"],
            weight_decay=resolved["weight_decay"],
            num_steps=resolved["num_steps"],
            use_prefix=resolved["use_prefix"],
            # Diffusion schedule
            ddpm_T=cfg_dict["diffusion"]["T"],
            beta_start=cfg_dict["diffusion"]["beta_start"],
            beta_end=cfg_dict["diffusion"]["beta_end"],
            learn_sigma=cfg_dict["diffusion"]["learn_sigma"],
            # Inference settings
            inference_steps=cfg_dict["inference"]["num_steps"],
            cfg_scale=cfg_dict["inference"]["cfg_scale"],
            clean_prefix_timestep=cfg_dict["autoregressive"]["clean_prefix_timestep"],
            cyclic_tpe_enabled=cfg_dict["autoregressive"]["cyclic_tpe"]["enabled"],
            spatial_cache_chunks=cfg_dict["autoregressive"]["spatial_cache_chunks"],
            num_ar_steps=cfg_dict["evaluation"]["temporal_consistency"]["num_ar_steps"],
            # Hardware and runtime
            device=device,
            # Paths
            output_dir=cfg_dict["paths"]["output_dir"],
            checkpoint_dir=cfg_dict["paths"]["checkpoint_dir"],
            data_root=cfg_dict["paths"]["data_root"],
            dataset_name=resolved["dataset_name"],
            vae_path=cfg_dict["paths"]["vae_path"],
            t5_path=cfg_dict["paths"]["t5_path"],
            opensora_ckpt=cfg_dict["paths"]["opensora_ckpt"],
            i3d_ckpt=cfg_dict["paths"]["i3d_ckpt"],
            ucf101_prompts=cfg_dict["paths"]["ucf101_prompts"],
            # Model architecture
            model_dim=cfg_dict["model"]["model_dim"],
            num_heads=cfg_dict["model"]["num_heads"],
            num_layers=cfg_dict["model"]["num_layers"],
            mlp_ratio=cfg_dict["model"]["mlp_ratio"],
            vae_channels=cfg_dict["model"]["vae_channels"],
            vae_downsample=cfg_dict["model"]["vae_downsample"],
            context_dim=cfg_dict["model"]["context_dim"],
            dropout=cfg_dict["model"]["dropout"],
            use_cross_attn=resolved["use_cross_attn"],
            # Logging and checkpointing
            log_every=cfg_dict["logging"]["log_every"],
            save_every=cfg_dict["logging"]["save_every"],
            use_wandb=cfg_dict["logging"]["use_wandb"],
            use_tensorboard=cfg_dict["logging"]["use_tensorboard"],
            # Hardware configuration
            mixed_precision=cfg_dict["hardware"]["mixed_precision"],
            dataloader_num_workers=cfg_dict["hardware"]["dataloader_num_workers"],
            pin_memory=cfg_dict["hardware"]["pin_memory"],
            num_gpus=cfg_dict["hardware"]["num_gpus"],
        )

        # Validate invariants after construction
        _validate(config)

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the Config to a plain Python dictionary.

        Returns:
            A flat dictionary mapping field names to their values.
            Suitable for logging to wandb/tensorboard or saving to JSON.
        """
        return dataclasses.asdict(self)


def _resolve_task_config(
    cfg: Dict[str, Any],
    task: str,
    stage: int,
    model_type: str,
) -> Dict[str, Any]:
    """Resolve task/stage/model_type-specific fields from the nested YAML dict.

    Selects the correct nested section of the config based on the task,
    training stage, and model type, and returns a flat dict of resolved values.

    Args:
        cfg: The full parsed YAML config as a plain Python dict.
        task: Task type ('t2v' or 'video_prediction').
        stage: Training stage (1 or 2), relevant only for task='t2v'.
        model_type: Model variant ('ca2vdm', 'os_fix', or 'os_ext').

    Returns:
        A flat dict containing resolved values for all task-specific fields.
    """
    resolved: Dict[str, Any] = {}

    if task == "video_prediction":
        vp_cfg = cfg["video_prediction"]
        resolved["resolution"] = vp_cfg["resolution"]
        resolved["chunk_len"] = vp_cfg["chunk_len"]
        resolved["dataset_name"] = vp_cfg["dataset"]
        # use_cross_attn is always False for video prediction (no text input)
        resolved["use_cross_attn"] = False

        if model_type in ("ca2vdm", "os_ext"):
            section = vp_cfg["ca2vdm_and_os_ext"]
            resolved["p_max"] = section["p_max"]
            resolved["l_train"] = section["l_train"]
            resolved["n_max"] = section["n_max"]
            resolved["batch_size"] = section["batch_size"]
            resolved["lr"] = float(section["learning_rate"])
            resolved["weight_decay"] = section["weight_decay"]
            resolved["num_steps"] = section["num_steps"]
            resolved["use_prefix"] = True  # Always use prefix for ca2vdm/os_ext
        else:
            # os_fix baseline for video prediction
            section = vp_cfg["os_fix_baseline"]
            resolved["p_max"] = section["prefix_len"]
            resolved["l_train"] = section["l_train"]
            resolved["n_max"] = 0  # Fixed prefix, no variable n
            resolved["batch_size"] = section["batch_size"]
            resolved["lr"] = float(section["learning_rate"])
            resolved["weight_decay"] = section["weight_decay"]
            resolved["num_steps"] = section["num_steps"]
            resolved["use_prefix"] = True  # OS-Fix always uses fixed prefix

    else:
        # task == 't2v'
        t2v_cfg = cfg["t2v"]
        resolved["resolution"] = t2v_cfg["resolution"]
        resolved["chunk_len"] = t2v_cfg["chunk_len"]
        resolved["dataset_name"] = t2v_cfg["dataset"]
        # use_cross_attn follows model config for T2V (text is always used)
        resolved["use_cross_attn"] = cfg["model"]["use_cross_attn"]

        if model_type == "os_fix":
            section = t2v_cfg["os_fix_baseline"]
            resolved["p_max"] = section["prefix_len"]
            resolved["l_train"] = section["l_train"]
            resolved["n_max"] = 0  # Fixed prefix for OS-Fix
            resolved["batch_size"] = section["batch_size"]
            resolved["lr"] = float(section["learning_rate"])
            resolved["weight_decay"] = section["weight_decay"]
            resolved["num_steps"] = section["num_steps"]
            resolved["use_prefix"] = True
        elif stage == 1:
            # Stage 1: causal modeling without clean prefix on 32-frame videos
            stage1_cfg = t2v_cfg["stage1"]
            resolved["p_max"] = t2v_cfg["p_max"]  # P_max defined globally for T2V
            resolved["l_train"] = stage1_cfg["num_frames"]  # 32 frames
            resolved["n_max"] = t2v_cfg["n_max"]
            resolved["batch_size"] = stage1_cfg["batch_size"]
            resolved["lr"] = float(stage1_cfg["learning_rate"])
            resolved["weight_decay"] = stage1_cfg["weight_decay"]
            resolved["num_steps"] = stage1_cfg["num_steps"]
            resolved["use_prefix"] = stage1_cfg["use_prefix"]  # False for stage 1
        else:
            # Stage 2: training with clean prefix on up to 65-frame videos
            stage2_cfg = t2v_cfg["stage2"]
            resolved["p_max"] = t2v_cfg["p_max"]  # 49
            resolved["l_train"] = t2v_cfg["l_train_max"]  # 65
            resolved["n_max"] = t2v_cfg["n_max"]
            resolved["batch_size"] = stage2_cfg["batch_size"]
            resolved["lr"] = float(stage2_cfg["learning_rate"])
            resolved["weight_decay"] = stage2_cfg["weight_decay"]
            resolved["num_steps"] = stage2_cfg["num_steps"]
            resolved["use_prefix"] = stage2_cfg["use_prefix"]  # True for stage 2

    return resolved


def _validate(config: Config) -> None:
    """Validate configuration invariants from the paper.

    Checks that all derived relationships between hyperparameters hold,
    as specified in the paper's methodology.

    Args:
        config: The Config instance to validate.

    Raises:
        ValueError: If any invariant is violated.
    """
    # 1. Prefix spatial length must be less than chunk length (paper: P' < l)
    if config.prefix_spatial_len >= config.chunk_len:
        raise ValueError(
            f"prefix_spatial_len ({config.prefix_spatial_len}) must be less than "
            f"chunk_len ({config.chunk_len}). Paper states P' < l."
        )

    # 2. Model dimension must be divisible by number of heads
    if config.model_dim % config.num_heads != 0:
        raise ValueError(
            f"model_dim ({config.model_dim}) must be divisible by "
            f"num_heads ({config.num_heads})."
        )

    # 3. For ca2vdm/os_ext: validate p_max = 1 + n_max * chunk_len
    if config.model_type in ("ca2vdm", "os_ext"):
        expected_p_max = 1 + config.n_max * config.chunk_len
        if config.p_max != expected_p_max:
            raise ValueError(
                f"For model_type='{config.model_type}', p_max must equal "
                f"1 + n_max * chunk_len = 1 + {config.n_max} * {config.chunk_len} "
                f"= {expected_p_max}, but got p_max={config.p_max}. "
                f"Paper invariant: P_max = 1 + n*l."
            )

    # 4. For ca2vdm/os_ext in stage 2 or video_prediction:
    #    l_train should equal p_max + chunk_len
    if config.model_type in ("ca2vdm", "os_ext"):
        is_t2v_stage2 = config.task == "t2v" and config.stage == 2
        is_video_pred = config.task == "video_prediction"
        if is_t2v_stage2 or is_video_pred:
            expected_l_train = config.p_max + config.chunk_len
            if config.l_train != expected_l_train:
                raise ValueError(
                    f"For model_type='{config.model_type}' in stage 2 / video_prediction, "
                    f"l_train must equal p_max + chunk_len = "
                    f"{config.p_max} + {config.chunk_len} = {expected_l_train}, "
                    f"but got l_train={config.l_train}. "
                    f"Paper invariant: L_train = P_max + l."
                )

    # 5. For t2v stage 1: use_prefix must be False
    if config.task == "t2v" and config.stage == 1 and config.use_prefix:
        raise ValueError(
            "For task='t2v' stage=1, use_prefix must be False. "
            "Stage 1 trains causal modeling without clean prefix frames."
        )

    # 6. For video_prediction: use_cross_attn must be False
    if config.task == "video_prediction" and config.use_cross_attn:
        raise ValueError(
            "For task='video_prediction', use_cross_attn must be False. "
            "Video prediction does not use text conditioning."
        )

    # 7. Diffusion schedule sanity checks
    if config.beta_start <= 0 or config.beta_end <= 0:
        raise ValueError(
            f"beta_start ({config.beta_start}) and beta_end ({config.beta_end}) "
            f"must be positive."
        )
    if config.beta_start >= config.beta_end:
        raise ValueError(
            f"beta_start ({config.beta_start}) must be less than "
            f"beta_end ({config.beta_end})."
        )
    if config.ddpm_T <= 0:
        raise ValueError(f"ddpm_T ({config.ddpm_T}) must be positive.")

    # 8. Inference steps must be positive and <= ddpm_T
    if config.inference_steps <= 0 or config.inference_steps > config.ddpm_T:
        raise ValueError(
            f"inference_steps ({config.inference_steps}) must be in "
            f"(0, ddpm_T={config.ddpm_T}]."
        )

    # 9. clean_prefix_timestep must be 0 (paper invariant: tEmb(0) for clean prefix)
    if config.clean_prefix_timestep != 0:
        raise ValueError(
            f"clean_prefix_timestep must be 0 (paper: clean prefix always uses "
            f"tEmb(0)), but got {config.clean_prefix_timestep}."
        )

    # 10. spatial_cache_chunks must be 1 (paper: only store one chunk of spatial KV)
    if config.spatial_cache_chunks != 1:
        raise ValueError(
            f"spatial_cache_chunks must be 1 (paper: spatial KV-cache stores "
            f"only one chunk), but got {config.spatial_cache_chunks}."
        )

    # 11. VAE channels must be 4 (SD VAE)
    if config.vae_channels != 4:
        raise ValueError(
            f"vae_channels must be 4 for Stable Diffusion VAE, "
            f"but got {config.vae_channels}."
        )

    # 12. VAE downsample must be 8 (SD VAE 8x spatial downsampling)
    if config.vae_downsample != 8:
        raise ValueError(
            f"vae_downsample must be 8 for Stable Diffusion VAE, "
            f"but got {config.vae_downsample}."
        )

    # 13. Resolution must be positive and divisible by vae_downsample
    if config.resolution <= 0:
        raise ValueError(f"resolution ({config.resolution}) must be positive.")
    if config.resolution % config.vae_downsample != 0:
        raise ValueError(
            f"resolution ({config.resolution}) must be divisible by "
            f"vae_downsample ({config.vae_downsample})."
        )

    # 14. num_ar_steps must be positive
    if config.num_ar_steps <= 0:
        raise ValueError(f"num_ar_steps ({config.num_ar_steps}) must be positive.")

    # 15. Learning rate must be positive
    if config.lr <= 0:
        raise ValueError(f"lr ({config.lr}) must be positive.")

    # 16. batch_size must be positive
    if config.batch_size <= 0:
        raise ValueError(f"batch_size ({config.batch_size}) must be positive.")

    # 17. num_steps must be positive
    if config.num_steps <= 0:
        raise ValueError(f"num_steps ({config.num_steps}) must be positive.")

    # 18. cfg_scale must be positive for T2V
    if config.task == "t2v" and config.cfg_scale <= 0:
        raise ValueError(
            f"cfg_scale ({config.cfg_scale}) must be positive for task='t2v'."
        )

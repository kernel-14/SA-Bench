from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import Ca2VDMConfig, EvalConfig, get_t2v_ca2vdm_config, get_vidpred_ca2vdm_config
from data import MSRVTTDataset, SkyTimelapseDataset, UCF101Dataset, collate_fn
from inference import AutoregressiveInference, OSExtInference, OSFixInference
from model import build_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# I3D Feature Extractor for FVD
# ---------------------------------------------------------------------------

class I3DFeatureExtractor(nn.Module):
    """
    I3D model for extracting video features for FVD computation.
    Uses pretrained I3D on Kinetics (following StyleGAN-V codebase).

    FVD is computed on the feature space of I3D following:
    Unterthiner et al., "FVD: A New Metric for Video Generation", ICLR 2019 Workshop.
    """

    def __init__(self, checkpoint_path: str, device: torch.device):
        super().__init__()
        self.device = device
        self.model = self._load_i3d(checkpoint_path)
        self.model.eval()

    def _load_i3d(self, checkpoint_path: str) -> nn.Module:
        """Load pretrained I3D model."""
        try:
            # Try loading from checkpoint
            model = torch.hub.load("facebookresearch/pytorchvideo", "i3d_r50", pretrained=False)
            if os.path.exists(checkpoint_path):
                state_dict = torch.load(checkpoint_path, map_location="cpu")
                model.load_state_dict(state_dict, strict=False)
        except Exception:
            # Fallback: use torchvision's video model
            from torchvision.models.video import r3d_18
            model = r3d_18(pretrained=True)
            # Remove final classification layer
            model.fc = nn.Identity()
        return model.to(self.device)

    @torch.no_grad()
    def extract_features(self, videos: torch.Tensor) -> torch.Tensor:
        """
        Extract I3D features from video clips.

        Args:
            videos: (B, T, C, H, W) float tensor in [0, 1]
                    T must be >= 16 (I3D minimum)
        Returns:
            features: (B, feature_dim)
        """
        # I3D expects (B, C, T, H, W)
        videos = videos.permute(0, 2, 1, 3, 4).to(self.device)

        # Normalize to [-1, 1] or [0, 1] depending on I3D variant
        videos = videos * 2.0 - 1.0

        # Resize to 224x224 if needed
        B, C, T, H, W = videos.shape
        if H != 224 or W != 224:
            videos = F.interpolate(
                videos.reshape(B * C, T, H, W).unsqueeze(1),
                size=(T, 224, 224),
                mode="trilinear",
                align_corners=False,
            ).squeeze(1).reshape(B, C, T, 224, 224)

        features = self.model(videos)
        if isinstance(features, dict):
            features = features.get("pool", list(features.values())[0])
        return features.reshape(B, -1)


# ---------------------------------------------------------------------------
# FVD Computation
# ---------------------------------------------------------------------------

def compute_fvd(
    real_features: np.ndarray,
    fake_features: np.ndarray,
) -> float:
    """
    Compute Frechet Video Distance (FVD) between real and generated video features.

    FVD = ||mu_r - mu_f||^2 + Tr(Sigma_r + Sigma_f - 2 * sqrt(Sigma_r @ Sigma_f))

    Args:
        real_features: (N, D) numpy array
        fake_features: (N, D) numpy array
    Returns:
        fvd: scalar FVD score
    """
    from scipy import linalg

    mu_r = np.mean(real_features, axis=0)
    mu_f = np.mean(fake_features, axis=0)
    sigma_r = np.cov(real_features, rowvar=False)
    sigma_f = np.cov(fake_features, rowvar=False)

    diff = mu_r - mu_f
    mean_term = np.dot(diff, diff)

    # Matrix square root
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_f, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    trace_term = np.trace(sigma_r) + np.trace(sigma_f) - 2 * np.trace(covmean)
    fvd = mean_term + trace_term
    return float(fvd)


# ---------------------------------------------------------------------------
# Video Generation for Evaluation
# ---------------------------------------------------------------------------

class VideoGenerationDataset(Dataset):
    """Dataset wrapper for generating videos from prompts/first frames."""

    def __init__(
        self,
        source_dataset: Dataset,
        num_samples: int,
    ):
        self.source = source_dataset
        self.num_samples = min(num_samples, len(source_dataset))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict:
        return self.source[idx]


# ---------------------------------------------------------------------------
# FVD Evaluator
# ---------------------------------------------------------------------------

class FVDEvaluator:
    """
    FVD evaluator following the paper's evaluation protocol.

    For in-chunk FVD (Table 1, 2): evaluate single 16-frame chunk
    For temporal consistency FVD (Table 3): evaluate chunk i vs chunk 1
    For ablation FVD (Table 4): evaluate each AR step chunk vs ground truth
    """

    def __init__(
        self,
        i3d_extractor: I3DFeatureExtractor,
        eval_config: EvalConfig,
        device: torch.device,
    ):
        self.i3d = i3d_extractor
        self.config = eval_config
        self.device = device

    def extract_video_features(
        self,
        dataloader: DataLoader,
        max_samples: int = 2048,
    ) -> np.ndarray:
        """Extract I3D features from a dataset."""
        all_features = []
        count = 0

        for batch in tqdm(dataloader, desc="Extracting features"):
            frames = batch["frames"].to(self.device)  # (B, L, C, H, W)
            B, L, C, H, W = frames.shape

            # Use first 16 frames for FVD
            frames_16 = frames[:, :16]  # (B, 16, C, H, W)
            frames_01 = (frames_16 + 1) / 2  # [-1,1] -> [0,1]

            features = self.i3d.extract_features(frames_01)
            all_features.append(features.cpu().numpy())
            count += B
            if count >= max_samples:
                break

        return np.concatenate(all_features, axis=0)[:max_samples]

    def evaluate_in_chunk_fvd(
        self,
        model,
        inference_engine,
        real_dataloader: DataLoader,
        num_samples: int = 2048,
        chunk_len: int = 16,
    ) -> float:
        """
        Evaluate in-chunk FVD (Tables 1 & 2 in paper).
        Generates single chunks and computes FVD against real videos.
        """
        # Extract real features
        logger.info("Extracting real video features...")
        real_features = self.extract_video_features(real_dataloader, num_samples)

        # Generate fake videos
        logger.info("Generating fake videos...")
        fake_features = []
        count = 0

        for batch in tqdm(real_dataloader, desc="Generating"):
            frames = batch["frames"].to(self.device)
            B = frames.shape[0]
            first_frame = frames[:, 0]  # (B, C, H, W) — use first frame as condition

            # Generate one chunk
            generated = inference_engine.generate(
                first_frame=first_frame,
                num_ar_steps=1,
                verbose=False,
            )
            # generated: (B, 1+l, C, H, W)
            gen_chunk = generated[:, 1:1+chunk_len]  # (B, l, C, H, W)
            gen_01 = (gen_chunk + 1) / 2

            features = self.i3d.extract_features(gen_01)
            fake_features.append(features.cpu().numpy())
            count += B
            if count >= num_samples:
                break

        fake_features = np.concatenate(fake_features, axis=0)[:num_samples]
        fvd = compute_fvd(real_features, fake_features)
        logger.info(f"In-chunk FVD: {fvd:.1f}")
        return fvd

    def evaluate_temporal_consistency_fvd(
        self,
        inference_engine,
        real_dataloader: DataLoader,
        num_ar_steps: int = 6,
        chunk_len: int = 8,
        num_samples: int = 512,
    ) -> Dict[int, float]:
        """
        Evaluate temporal consistency FVD (Table 3 in paper).
        Computes FVD between AR step i and AR step 1 for i=2,...,6.

        Each model generates 48 frames with 6 AR steps (l=8).
        FVD evaluated on three 16-frame chunks (2 AR steps each).
        """
        # Extract real features for reference (first chunk)
        logger.info("Extracting reference features (chunk 1)...")
        ref_features_list = []
        gen_chunk_features: Dict[int, List[np.ndarray]] = {i: [] for i in range(1, num_ar_steps + 1)}
        count = 0

        for batch in tqdm(real_dataloader, desc="Generating for temporal FVD"):
            frames = batch["frames"].to(self.device)
            B = frames.shape[0]
            first_frame = frames[:, 0]

            generated = inference_engine.generate(
                first_frame=first_frame,
                num_ar_steps=num_ar_steps,
                verbose=False,
            )
            # generated: (B, 1 + num_ar_steps*chunk_len, C, H, W)

            # Extract features for each AR step chunk
            for ar_step in range(1, num_ar_steps + 1):
                start = 1 + (ar_step - 1) * chunk_len
                end = start + chunk_len
                chunk = generated[:, start:end]  # (B, l, C, H, W)
                chunk_01 = (chunk + 1) / 2
                features = self.i3d.extract_features(chunk_01)
                gen_chunk_features[ar_step].append(features.cpu().numpy())

            count += B
            if count >= num_samples:
                break

        # Compute FVD between chunk i and chunk 1
        chunk1_features = np.concatenate(gen_chunk_features[1], axis=0)[:num_samples]
        fvd_results = {}

        for ar_step in range(2, num_ar_steps + 1):
            chunk_i_features = np.concatenate(gen_chunk_features[ar_step], axis=0)[:num_samples]
            fvd = compute_fvd(chunk1_features, chunk_i_features)
            fvd_results[ar_step] = fvd
            logger.info(f"FVD (chunk 1 vs chunk {ar_step}): {fvd:.1f}")

        return fvd_results

    def evaluate_ablation_fvd(
        self,
        inference_engine,
        real_dataloader: DataLoader,
        num_ar_steps: int = 6,
        chunk_len: int = 16,
        num_samples: int = 512,
    ) -> Dict[int, float]:
        """
        Evaluate ablation FVD (Table 4 in paper).
        Each model generates 96 frames with 6 AR steps (l=16).
        FVD evaluated for each AR step chunk vs ground truth 16-frame videos.
        """
        logger.info("Extracting real features for ablation evaluation...")
        real_features = self.extract_video_features(real_dataloader, num_samples)

        chunk_fvd_results = {}
        count = 0
        chunk_features: Dict[int, List[np.ndarray]] = {i: [] for i in range(1, num_ar_steps + 1)}

        for batch in tqdm(real_dataloader, desc="Generating for ablation FVD"):
            frames = batch["frames"].to(self.device)
            B = frames.shape[0]
            first_frame = frames[:, 0]

            generated = inference_engine.generate(
                first_frame=first_frame,
                num_ar_steps=num_ar_steps,
                verbose=False,
            )

            for ar_step in range(1, num_ar_steps + 1):
                start = 1 + (ar_step - 1) * chunk_len
                end = start + chunk_len
                chunk = generated[:, start:end]
                chunk_01 = (chunk + 1) / 2
                features = self.i3d.extract_features(chunk_01)
                chunk_features[ar_step].append(features.cpu().numpy())

            count += B
            if count >= num_samples:
                break

        for ar_step in range(1, num_ar_steps + 1):
            chunk_i_features = np.concatenate(chunk_features[ar_step], axis=0)[:num_samples]
            fvd = compute_fvd(real_features, chunk_i_features)
            chunk_fvd_results[ar_step] = fvd
            logger.info(f"Ablation FVD (chunk {ar_step}): {fvd:.1f}")

        return chunk_fvd_results


# ---------------------------------------------------------------------------
# Efficiency Evaluation (FLOPs and Time)
# ---------------------------------------------------------------------------

def measure_inference_time(
    inference_engine,
    first_frame: torch.Tensor,
    num_ar_steps: int,
    num_frames_total: int = 80,
    chunk_len: int = 8,
    num_warmup: int = 2,
    num_runs: int = 3,
) -> float:
    """
    Measure wall-clock time for autoregressive generation.
    Reproduces Table 5 in the paper (80 frames at 256x256).

    Args:
        inference_engine: inference engine instance
        first_frame: (1, C, H, W) first frame latent
        num_ar_steps: number of AR steps
        num_frames_total: total frames to generate (80 in paper)
        chunk_len: l
        num_warmup: warmup runs
        num_runs: timing runs
    Returns:
        avg_time: average generation time in seconds
    """
    import time

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            inference_engine.generate(first_frame, num_ar_steps=2, verbose=False)

    # Timing
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        with torch.no_grad():
            inference_engine.generate(first_frame, num_ar_steps=num_ar_steps, verbose=False)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        times.append(time.time() - start)

    return float(np.mean(times))


def count_flops_attention(
    seq_len: int,
    head_dim: int,
    num_heads: int,
) -> int:
    """
    Count FLOPs for one attention layer.
    FLOPs = 2 * seq_len * seq_len * head_dim * num_heads (for QK^T and AV)
    """
    return 2 * seq_len * seq_len * head_dim * num_heads


def analyze_computational_cost(
    num_ar_steps: int,
    chunk_len: int,
    p_max: int,
    hidden_dim: int,
    num_heads: int,
    hw: int,
    model_type: str = "ca2vdm",
) -> Dict[str, List[int]]:
    """
    Analyze FLOPs for temporal, spatial, and cross-attention layers
    across AR steps (reproduces Figure 8 in paper).

    Args:
        num_ar_steps: number of AR steps
        chunk_len: l
        p_max: P_max
        hidden_dim: model dimension
        num_heads: number of attention heads
        hw: H*W (spatial tokens per frame)
        model_type: "ca2vdm", "osfix", or "osext"
    Returns:
        flops_per_step: dict with "temporal", "spatial", "cross" FLOPs per step
    """
    head_dim = hidden_dim // num_heads
    flops = {"temporal": [], "spatial": [], "cross": []}

    for step in range(1, num_ar_steps + 1):
        p_k = min(1 + (step - 1) * chunk_len, p_max)

        if model_type == "ca2vdm":
            # Temporal: only denoising target (l frames) attends to (P_k + l) frames
            # But P_k is in KV-cache, so computation is l x (P_k + l) per spatial grid
            temp_seq = chunk_len  # query length
            temp_kv = p_k + chunk_len  # key/value length
            temporal_flops = 2 * temp_seq * temp_kv * head_dim * num_heads * hw

            # Spatial: each frame attends to (P'+1)*HW tokens
            # P' = 3, so (P'+1)*HW = 4*HW
            spatial_flops = 2 * hw * (4 * hw) * head_dim * num_heads * chunk_len

            # Cross: l frames x text_seq (constant)
            text_seq = 120
            cross_flops = 2 * hw * text_seq * head_dim * num_heads * chunk_len

        elif model_type == "osext":
            # Temporal: all (P_k + l) frames attend to all (P_k + l) frames
            total_seq = p_k + chunk_len
            temporal_flops = 2 * total_seq * total_seq * head_dim * num_heads * hw

            # Spatial: each frame attends to HW tokens
            spatial_flops = 2 * hw * hw * head_dim * num_heads * total_seq

            # Cross: all frames x text_seq
            text_seq = 120
            cross_flops = 2 * hw * text_seq * head_dim * num_heads * total_seq

        else:  # osfix
            # Fixed prefix P
            fixed_p = p_max // 3  # approximate fixed prefix
            total_seq = fixed_p + chunk_len
            temporal_flops = 2 * total_seq * total_seq * head_dim * num_heads * hw
            spatial_flops = 2 * hw * hw * head_dim * num_heads * total_seq
            text_seq = 120
            cross_flops = 2 * hw * text_seq * head_dim * num_heads * total_seq

        flops["temporal"].append(temporal_flops)
        flops["spatial"].append(spatial_flops)
        flops["cross"].append(cross_flops)

    return flops


# ---------------------------------------------------------------------------
# Main Evaluation Script
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Ca2-VDM")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="ca2vdm",
                        choices=["ca2vdm", "osfix", "osext"])
    parser.add_argument("--task", type=str, default="t2v", choices=["t2v", "vidpred"])
    parser.add_argument("--dataset", type=str, default="msrvtt",
                        choices=["msrvtt", "ucf101", "skytimelapse"])
    parser.add_argument("--eval_type", type=str, default="in_chunk",
                        choices=["in_chunk", "temporal_consistency", "ablation", "efficiency"])
    parser.add_argument("--num_ar_steps", type=int, default=6)
    parser.add_argument("--num_samples", type=int, default=2048)
    parser.add_argument("--i3d_checkpoint", type=str, default="pretrained/i3d_kinetics.pt")
    parser.add_argument("--output_dir", type=str, default="eval_results")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    if args.task == "t2v":
        config = get_t2v_ca2vdm_config()
    else:
        config = get_vidpred_ca2vdm_config()

    inf_config = config.inference
    eval_config = config.eval

    # Build model
    model_cfg = type("ModelCfg", (), {
        "in_channels": config.model.in_channels,
        "patch_size": config.model.patch_size,
        "hidden_dim": config.model.hidden_dim,
        "num_layers": config.model.num_layers,
        "num_heads": config.model.num_heads,
        "context_dim": config.model.context_dim if config.model.use_text else None,
        "ff_mult": config.model.ff_mult,
        "dropout": 0.0,
        "max_spatial_h": config.model.max_spatial_h,
        "max_spatial_w": config.model.max_spatial_w,
        "max_temporal_len": config.model.max_temporal_len,
        "chunk_len": inf_config.chunk_len,
        "p_max": inf_config.p_max,
        "prefix_len": config.model.prefix_len,
        "use_text": config.model.use_text,
        "fixed_prefix": config.t2v_train.fixed_prefix,
    })()

    model = build_model(args.model_type, model_cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Build inference engine
    if args.model_type == "ca2vdm":
        engine = AutoregressiveInference(model, inf_config, device)
    elif args.model_type == "osfix":
        engine = OSFixInference(model, inf_config, device, fixed_prefix=config.t2v_train.fixed_prefix)
    else:
        engine = OSExtInference(model, inf_config, device)

    # Build dataset
    dataset_cfg = type("DataCfg", (), {
        "msrvtt_root": eval_config.msrvtt_root,
        "ucf101_root": eval_config.ucf101_root,
        "skytimelapse_root": eval_config.skytimelapse_root,
        "resolution": inf_config.resolution,
        "chunk_len": inf_config.chunk_len,
        "max_train_frames": inf_config.chunk_len * args.num_ar_steps + 1,
    })()

    if args.dataset == "msrvtt":
        dataset = MSRVTTDataset(
            eval_config.msrvtt_root,
            resolution=(inf_config.resolution, inf_config.resolution),
            num_frames=16,
            split="test",
        )
    elif args.dataset == "ucf101":
        dataset = UCF101Dataset(
            eval_config.ucf101_root,
            resolution=(inf_config.resolution, inf_config.resolution),
            num_frames=16,
            split="test",
        )
    else:
        dataset = SkyTimelapseDataset(
            eval_config.skytimelapse_root,
            resolution=(inf_config.resolution, inf_config.resolution),
            num_frames=inf_config.chunk_len * args.num_ar_steps + 1,
            split="test",
        )

    dataloader = DataLoader(
        dataset,
        batch_size=eval_config.eval_batch_size,
        shuffle=False,
        num_workers=eval_config.num_workers,
        collate_fn=collate_fn,
    )

    # I3D feature extractor
    i3d = I3DFeatureExtractor(args.i3d_checkpoint, device)
    evaluator = FVDEvaluator(i3d, eval_config, device)

    # Run evaluation
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_type == "in_chunk":
        fvd = evaluator.evaluate_in_chunk_fvd(
            model=model,
            inference_engine=engine,
            real_dataloader=dataloader,
            num_samples=args.num_samples,
            chunk_len=inf_config.chunk_len,
        )
        results = {"in_chunk_fvd": fvd}

    elif args.eval_type == "temporal_consistency":
        fvd_results = evaluator.evaluate_temporal_consistency_fvd(
            inference_engine=engine,
            real_dataloader=dataloader,
            num_ar_steps=args.num_ar_steps,
            chunk_len=inf_config.chunk_len,
            num_samples=args.num_samples,
        )
        results = {f"fvd_chunk_{k}": v for k, v in fvd_results.items()}

    elif args.eval_type == "ablation":
        fvd_results = evaluator.evaluate_ablation_fvd(
            inference_engine=engine,
            real_dataloader=dataloader,
            num_ar_steps=args.num_ar_steps,
            chunk_len=inf_config.chunk_len,
            num_samples=args.num_samples,
        )
        results = {f"fvd_chunk_{k}": v for k, v in fvd_results.items()}

    elif args.eval_type == "efficiency":
        # FLOPs analysis
        flops = analyze_computational_cost(
            num_ar_steps=args.num_ar_steps,
            chunk_len=inf_config.chunk_len,
            p_max=inf_config.p_max,
            hidden_dim=config.model.hidden_dim,
            num_heads=config.model.num_heads,
            hw=inf_config.latent_h * inf_config.latent_w // (config.model.patch_size ** 2),
            model_type=args.model_type,
        )
        logger.info("FLOPs per AR step:")
        for step in range(args.num_ar_steps):
            logger.info(
                f"  Step {step+1}: temporal={flops['temporal'][step]:.2e}, "
                f"spatial={flops['spatial'][step]:.2e}, "
                f"cross={flops['cross'][step]:.2e}"
            )

        # Timing
        first_frame = torch.randn(1, config.model.in_channels,
                                   inf_config.latent_h, inf_config.latent_w, device=device)
        num_ar_for_80_frames = (80 - 1) // inf_config.chunk_len
        avg_time = measure_inference_time(engine, first_frame, num_ar_for_80_frames)
        logger.info(f"Average time for 80 frames: {avg_time:.1f}s")
        results = {"avg_time_80frames": avg_time, "flops": flops}

    # Save results
    import json
    results_path = output_dir / f"{args.model_type}_{args.dataset}_{args.eval_type}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()

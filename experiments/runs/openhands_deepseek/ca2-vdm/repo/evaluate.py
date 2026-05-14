"""
Evaluation script for Ca2-VDM.

Covers all experiments in the paper:
- Tables 1 & 2: Zero-shot and finetuned FVD on MSR-VTT and UCF-101
- Table 3: Temporal consistency (FVD between AR steps)
- Table 4: Ablation studies (P_max and prefix-enhancement)
- Tables 5 & 6: Efficiency (time cost, memory cost)
- Figure 6: Accumulated time cost
- Figure 8: FLOPs count
"""

import os
import time
import math
import torch
import torch.nn as nn
import argparse
from typing import Optional, List, Tuple, Dict
from tqdm import tqdm

from config import Config, get_t2v_config, get_video_prediction_config
from model import Ca2VDM, Ca2VDM_Bidirectional
from inference import Ca2VDMInference, BidirectionalInference, decode_latents
from fvd import compute_fvd, compute_chunk_fvd, I3DFeatureExtractor


@torch.no_grad()
def evaluate_in_chunk_quality(
    model: nn.Module,
    vae_decoder: nn.Module,
    dataloader,
    i3d_model: nn.Module,
    config: Config,
    model_type: str = "ca2_vdm",
    device: torch.device = None,
    num_samples: int = 2048,
) -> Dict[str, float]:
    """Evaluate zero-shot FVD (Table 1) or finetuned FVD (Table 2).

    Generates single chunks (non-autoregressive) and computes FVD against ground truth.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    gen_videos = []
    real_videos = []

    for batch in tqdm(dataloader, desc="In-chunk evaluation"):
        if len(real_videos) * dataloader.batch_size >= num_samples:
            break

        latent = batch["latent"].to(device)
        text_embed = batch.get("text_embed", None)
        if text_embed is not None:
            text_embed = text_embed.to(device)

        B, L, C, H, W = latent.shape
        P = batch.get("prefix_len", L // 2)

        # Generate
        noise = torch.randn(B, L - P, C, H, W, device=device)
        z_t = noise
        timesteps = torch.linspace(999, 0, config.inference.num_inference_steps, dtype=torch.long, device=device)

        for t in timesteps:
            z_input = torch.cat([latent[:, :P], z_t], dim=1)
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            if model_type == "ca2_vdm":
                output = model(z_input, t_batch, P, text_embed)
            else:
                output = model(z_input, t_batch, P, text_embed)
            pred = output["pred"]
            pred_target = pred[:, P:]

            # DDPM step
            alpha_bar_t = _get_alpha_bar(t, device)
            alpha_bar_t_prev = _get_alpha_bar(t - 1, device) if t > 0 else torch.ones(1, device=device)
            pred_x0 = (z_t - (1 - alpha_bar_t).sqrt() * pred_target) / alpha_bar_t.sqrt().clamp(min=1e-8)

            beta_t = 1 - alpha_bar_t / alpha_bar_t_prev if t > 0 else torch.zeros(1, device=device)
            alpha_t = 1 - beta_t

            pred_mean = (
                beta_t * alpha_bar_t_prev.sqrt() / (1 - alpha_bar_t) * pred_x0
                + (1 - alpha_bar_t_prev) * alpha_t.sqrt() / (1 - alpha_bar_t) * z_t
            )

            if t > 0:
                log_var = torch.log(beta_t * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)).clamp(max=20.0)
                z_t = pred_mean + (0.5 * log_var).exp() * torch.randn_like(z_t)
            else:
                z_t = pred_mean

        full_gen = torch.cat([latent[:, :P], z_t], dim=1)
        gen_videos.append(decode_latents(vae_decoder, full_gen))
        real_videos.append(decode_latents(vae_decoder, latent))

    gen_videos = torch.cat(gen_videos, dim=0)[:num_samples]
    real_videos = torch.cat(real_videos, dim=0)[:num_samples]

    fvd = compute_fvd(real_videos, gen_videos, i3d_model, batch_size=16, device=device)
    return {"FVD": fvd}


@torch.no_grad()
def evaluate_temporal_consistency(
    model: nn.Module,
    vae_decoder: nn.Module,
    first_frames: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor],
    config: Config,
    model_type: str = "ca2_vdm",
    num_ar_steps: int = 6,
    device: torch.device = None,
) -> List[float]:
    """Evaluate temporal consistency via FVD between AR steps (Table 3).

    Generates video autoregressively and computes FVD between each step and step 1.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "ca2_vdm":
        inference = Ca2VDMInference(model, config, device)
    else:
        inference = BidirectionalInference(model, config, model_type, device)

    # Generate video
    video_latent = inference.generate(
        first_frames.to(device),
        num_ar_steps,
        encoder_hidden_states.to(device) if encoder_hidden_states is not None else None,
        verbose=True,
    )

    # Decode and split into chunks per AR step
    video_pixels = decode_latents(vae_decoder, video_latent)
    l = config.inference.chunk_length
    # first chunk is frame 0 + l frames
    chunks = []
    chunks.append(video_pixels[:, :1 + l])  # step 0: first chunk
    for i in range(1, num_ar_steps):
        start = 1 + i * l
        end = start + l
        chunks.append(video_pixels[:, start:end])

    # Compute pairwise FVD
    i3d_model = I3DFeatureExtractor()
    fvd_scores = []
    for i in range(1, len(chunks)):
        fvd = compute_fvd(chunks[0], chunks[i], i3d_model, device=device)
        fvd_scores.append(fvd)

    return fvd_scores


def evaluate_efficiency(
    model: nn.Module,
    config: Config,
    model_type: str = "ca2_vdm",
    num_ar_steps: int = 7,
    device: torch.device = None,
) -> Dict:
    """Evaluate time cost and FLOPs (Tables 5, 6, Figures 6, 8).

    Args:
        model: Ca2VDM or bidirectional model
        config: configuration
        model_type: "ca2_vdm", "os_fix", or "os_ext"
        num_ar_steps: number of AR steps
        device: compute device

    Returns:
        Dictionary with timing and FLOPs measurements
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    B = 1
    l = config.inference.chunk_length
    C = config.model.latent_channels
    H = W = config.model.spatial_size

    results = {}

    # Measure per-step time
    first_frame = torch.randn(B, 1, C, H, W, device=device)

    if model_type == "ca2_vdm":
        inference = Ca2VDMInference(model, config, device)
    else:
        inference = BidirectionalInference(model, config, model_type, device)

    step_times = []
    total_start = time.time()

    # Warm up
    _ = inference.generate(first_frame, 1, None, verbose=False)
    inference.reset()

    # Measure each step
    for step in range(num_ar_steps):
        start = time.time()
        _ = inference.generate(first_frame, step + 1, None, verbose=False)
        elapsed = time.time() - start
        step_times.append(elapsed)
        inference.reset()

    total_time = time.time() - total_start

    results["step_times"] = step_times
    results["cumulative_times"] = step_times  # accumulated time up to each step
    results["total_time"] = total_time

    # GPU memory usage
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        _ = inference.generate(first_frame, 1, None, verbose=False)
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
        results["peak_gpu_memory_gb"] = peak_memory
        inference.reset()

    # Estimate FLOPs using a simple formula based on attention operations
    results["flops"] = _estimate_flops(
        config, model_type, num_ar_steps,
        model.num_layers if hasattr(model, 'num_layers') else 28
    )

    return results


def _estimate_flops(
    config: Config,
    model_type: str,
    num_ar_steps: int,
    num_layers: int,
) -> Dict[str, float]:
    """Estimate FLOPs for each component (Figure 8).

    This provides approximate FLOPs counts based on attention dimensions.
    For exact numbers, use a profiler like fvcore or torch.profiler.
    """
    hidden_size = config.model.hidden_size
    num_heads = config.model.num_heads
    head_dim = config.model.spatial_attn_head_dim
    l = config.inference.chunk_length
    H = W = config.model.spatial_size
    HW = H * W

    P_max = config.inference.max_prefix_length

    flops_per_layer = {}

    # Spatial attention: QK^T + softmax + AV
    # For each frame: Q (HW x head_dim), K (HW x head_dim), V (HW x head_dim)
    # With prefix enhancement, K and V have (P'+1)*HW tokens
    P_prime = config.model.prefix_len_enhance
    spatial_flops = num_heads * (
        2 * HW * (P_prime + 1) * HW * head_dim  # QK^T
        + HW * (P_prime + 1) * HW  # softmax approximate
        + HW * (P_prime + 1) * HW * head_dim  # AV
    ) * l

    # Temporal attention
    if model_type == "ca2_vdm":
        total_L = min(P_max + l, l * num_ar_steps + 1)  # total frames including prefix
        temporal_flops = num_heads * (
            2 * l * total_L * head_dim * HW * HW  # QK^T
            + l * total_L * HW * HW  # softmax
            + l * total_L * head_dim * HW * HW  # AV
        )
    elif model_type == "os_ext":
        total_L = min(P_max + l, l * num_ar_steps + 1)
        temporal_flops = num_heads * (
            2 * total_L * total_L * head_dim * HW * HW
            + total_L * total_L * HW * HW
            + total_L * total_L * head_dim * HW * HW
        )
    else:  # os_fix
        fixed_P = config.training.max_train_len // 2
        total_L = fixed_P + l
        temporal_flops = num_heads * (
            2 * total_L * total_L * head_dim * HW * HW
            + total_L * total_L * HW * HW
            + total_L * total_L * head_dim * HW * HW
        )

    # Cross attention (same for all)
    text_len = 120
    cross_flops = num_heads * (
        2 * l * HW * text_len * head_dim
        + l * HW * text_len
        + l * HW * text_len * head_dim
    )

    flops_per_layer["spatial"] = spatial_flops * num_layers / 1e9  # GFLOPs
    flops_per_layer["temporal"] = temporal_flops * num_layers / 1e9
    flops_per_layer["cross"] = cross_flops * num_layers / 1e9
    flops_per_layer["total"] = flops_per_layer["spatial"] + flops_per_layer["temporal"] + flops_per_layer["cross"]

    return flops_per_layer


def ablation_study(
    model_variants: Dict[str, nn.Module],
    vae_decoder: nn.Module,
    dataloader,
    i3d_model: nn.Module,
    config: Config,
    device: torch.device = None,
    num_ar_steps: int = 6,
) -> Dict[str, List[float]]:
    """Run ablation study (Table 4).

    Compares variants with different P_max and with/without prefix-enhancement.

    Args:
        model_variants: dict mapping variant name to model
        vae_decoder: VAE decoder
        dataloader: data loader for first frames
        i3d_model: I3D model for FVD
        config: configuration
        device: compute device
        num_ar_steps: number of AR steps

    Returns:
        Dict mapping variant name to list of FVD scores per chunk
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_results = {}

    for name, model in model_variants.items():
        print(f"\n=== Evaluating {name} ===")
        model = model.to(device)
        model.eval()

        gen_videos = []
        real_videos = []

        for batch in tqdm(dataloader, desc=name):
            if len(gen_videos) > 10:
                break  # Sample for ablation

            latent = batch["latent"].to(device)[:1]  # Take first sample
            first_frame = latent[:, :1]

            inference = Ca2VDMInference(model, config, device)
            gen_latent = inference.generate(first_frame, num_ar_steps, None, verbose=False)

            gen_pixels = decode_latents(vae_decoder, gen_latent)
            real_pixels = decode_latents(vae_decoder, latent)

            gen_videos.append(gen_pixels)
            real_videos.append(real_pixels)

        gen_videos = torch.cat(gen_videos, dim=0)
        real_videos = torch.cat(real_videos, dim=0)

        # Compute chunk FVD
        l = config.inference.chunk_length
        # Generate 48 frames = 1 + 6*8 for video prediction, exclude first frame
        chunk_fvds = compute_chunk_fvd(
            real_videos[:, 1:, :, :, :],  # exclude first frame
            gen_videos[:, 1:, :, :, :],   # exclude first frame
            i3d_model,
            chunk_size=16,
            device=device,
        )
        all_results[name] = chunk_fvds

    return all_results


def _get_alpha_bar(t: int, device: torch.device, T: int = 1000) -> torch.Tensor:
    betas = torch.linspace(1e-4, 0.02, T, device=device)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    if t < 0:
        return torch.ones(1, device=device)
    if t >= T:
        t = T - 1
    return alpha_bars[t]


def main():
    parser = argparse.ArgumentParser(description="Evaluate Ca2-VDM")
    parser.add_argument("--task", type=str, default="all", choices=[
        "all", "in_chunk_quality", "temporal_consistency",
        "efficiency", "ablation"
    ])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config", type=str, default="t2v", choices=["t2v", "video_prediction"])
    parser.add_argument("--num_samples", type=int, default=2048)
    parser.add_argument("--num_ar_steps", type=int, default=6)
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.config == "t2v":
        config = get_t2v_config()
    else:
        config = get_video_prediction_config()

    # Load model
    model = Ca2VDM(
        hidden_size=config.model.hidden_size,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        spatial_head_dim=config.model.spatial_attn_head_dim,
        temporal_head_dim=config.model.temporal_attn_head_dim,
        cross_head_dim=config.model.cross_attn_head_dim,
        cross_attn_dim=config.model.text_encoder_dim,
        prefix_len_enhance=config.model.prefix_len_enhance,
        max_train_len=config.training.max_train_len,
        patch_size=config.model.patch_size,
        latent_channels=config.model.latent_channels,
        spatial_size=config.model.spatial_size,
        learn_sigma=True,
    )

    ckpt = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.task in ("all", "efficiency"):
        print("\n=== Efficiency Evaluation ===")
        results = evaluate_efficiency(model, config, "ca2_vdm", args.num_ar_steps, device)
        print(f"Step times: {results['step_times']}")
        print(f"Total time: {results['total_time']:.2f}s")
        if "peak_gpu_memory_gb" in results:
            print(f"Peak GPU memory: {results['peak_gpu_memory_gb']:.2f} GB")
        print(f"Estimated FLOPs (GFLOPs): {results['flops']}")

        # Save results
        torch.save(results, os.path.join(args.output_dir, "efficiency.pt"))

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()

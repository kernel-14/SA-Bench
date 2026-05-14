"""Inference for Pyramidal Flow Matching.

Supports:
- Text-to-video generation
- Text-conditioned image-to-video generation
- Classifier-free guidance
- Variable-duration generation (5s, 10s)
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional, List, Tuple, Dict, Any
import os
import math
import argparse
from PIL import Image
import numpy as np

from model import MMDiT
from vae import VideoVAE
from pyramidal_flow import (
    generate_pyramidal_flow,
    get_stage_boundaries,
    renoise_jump_point_nearest,
    up_sample,
)
from config import Config


class PyramidalFlowInference:
    """Inference pipeline for pyramidal flow matching."""

    def __init__(
        self,
        config: Config,
        model: MMDiT,
        vae: Optional[VideoVAE] = None,
        t5_model: Any = None,
        t5_tokenizer: Any = None,
        clip_model: Any = None,
        clip_tokenizer: Any = None,
        device: torch.device = torch.device("cuda"),
    ):
        self.config = config
        self.model = model.to(device)
        self.vae = vae.to(device) if vae is not None else None
        self.t5_model = t5_model.to(device) if t5_model is not None else None
        self.t5_tokenizer = t5_tokenizer
        self.clip_model = clip_model.to(device) if clip_model is not None else None
        self.clip_tokenizer = clip_tokenizer
        self.device = device

        self.model.eval()
        if self.vae is not None:
            self.vae.eval()
        if self.t5_model is not None:
            self.t5_model.eval()
        if self.clip_model is not None:
            self.clip_model.eval()

        self.stage_boundaries = get_stage_boundaries(
            num_stages=config.pyramid.num_stages,
        )

    def _encode_text(
        self, prompt: str
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Encode text prompt with T5 and CLIP."""
        with torch.no_grad():
            if self.t5_tokenizer is not None:
                t5_tokens = self.t5_tokenizer(
                    [prompt],
                    padding="max_length",
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                ).to(self.device)
                t5_outputs = self.t5_model(**t5_tokens)
                context = t5_outputs.last_hidden_state
                pooled = t5_outputs.last_hidden_state.mean(dim=1)
            else:
                context = torch.randn(1, 77, self.config.dit.context_dim, device=self.device)
                pooled = torch.randn(1, self.config.dit.pooled_text_dim, device=self.device)

            if self.clip_model is not None:
                clip_tokens = self.clip_tokenizer(
                    [prompt],
                    padding="max_length",
                    truncation=True,
                    max_length=77,
                    return_tensors="pt",
                ).to(self.device)
                clip_outputs = self.clip_model(**clip_tokens)
                clip_context = clip_outputs.last_hidden_state
            else:
                clip_context = None

        return context, pooled, clip_context

    def _build_temporal_pos_ids(
        self, num_frames: int, latent_frames_per_token: int
    ) -> torch.Tensor:
        """Build temporal position IDs for RoPE."""
        return torch.arange(num_frames, device=self.device).repeat_interleave(latent_frames_per_token)

    def _build_frame_boundaries(
        self,
        num_frames: int,
        spatial_h: int,
        spatial_w: int,
        patch_size: int,
    ) -> List[int]:
        """Build frame boundaries for blockwise causal attention."""
        tokens_per_frame = (spatial_h // patch_size) * (spatial_w // patch_size)
        boundaries = [i * tokens_per_frame for i in range(num_frames + 1)]
        return boundaries

    @torch.no_grad()
    def text_to_video(
        self,
        prompt: str,
        num_frames: int = 121,
        fps: int = 24,
        spatial_h: int = 768,
        spatial_w: int = 768,
        num_steps_per_stage: int = 20,
        cfg_scale: float = 7.0,
        seed: Optional[int] = None,
        first_frame_image: Optional[Image.Image] = None,
    ) -> torch.Tensor:
        """Generate a video from a text prompt.

        Args:
            prompt: Text description.
            num_frames: Number of frames to generate.
            fps: Frames per second.
            spatial_h, spatial_w: Output resolution.
            num_steps_per_stage: ODE steps per pyramid stage.
            cfg_scale: Classifier-free guidance scale.
            seed: Random seed.
            first_frame_image: Optional first frame for image-to-video.

        Returns:
            Video tensor of shape (T, 3, H, W) in [0, 1].
        """
        if seed is not None:
            torch.manual_seed(seed)

        latent_h = spatial_h // 8
        latent_w = spatial_w // 8
        latent_frames = num_frames

        context, pooled, clip_context = self._encode_text(prompt)

        # Build temporal position IDs and frame boundaries
        tokens_per_frame = (latent_h // self.config.dit.patch_size) * (latent_w // self.config.dit.patch_size)

        # Generate frame by frame (autoregressive)
        generated_latents = []

        for frame_idx in range(latent_frames):
            temporal_pos_ids = torch.arange(
                frame_idx, frame_idx + 1, device=self.device
            ).repeat_interleave(tokens_per_frame)

            frame_boundaries = [0, tokens_per_frame]

            if frame_idx == 0 and first_frame_image is not None and self.vae is not None:
                # Use provided first frame
                img = first_frame_image.convert("RGB").resize((spatial_w, spatial_h))
                img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                img_tensor = img_tensor * 2.0 - 1.0
                img_tensor = img_tensor.unsqueeze(0).unsqueeze(2).to(self.device)
                latent_frame = self.vae.encode_latents(img_tensor).squeeze(0)
                generated_latents.append(latent_frame)
            else:
                # Generate frame with pyramidal flow
                latent_frame = generate_pyramidal_flow(
                    model=self.model,
                    stage_boundaries=self.stage_boundaries,
                    context=context,
                    pooled_text=pooled,
                    num_steps_per_stage=num_steps_per_stage,
                    spatial_h=latent_h,
                    spatial_w=latent_w,
                    clip_context=clip_context,
                    temporal_pos_ids=temporal_pos_ids,
                    frame_boundaries=frame_boundaries,
                    corrective_gamma=self.config.pyramid.corrective_gamma,
                    cfg_scale=cfg_scale,
                )
                generated_latents.append(latent_frame)

        # Stack all frames
        video_latent = torch.stack(generated_latents, dim=0).unsqueeze(0)

        # Decode with VAE
        if self.vae is not None:
            video = self.vae.decode(video_latent)
            video = torch.clamp(video, -1, 1)
            video = (video + 1) / 2
            video = video.squeeze(0).cpu()
        else:
            video = video_latent.squeeze(0).cpu()

        return video

    @torch.no_grad()
    def image_to_video(
        self,
        prompt: str,
        image: Image.Image,
        num_frames: int = 121,
        fps: int = 24,
        spatial_h: int = 768,
        spatial_w: int = 768,
        num_steps_per_stage: int = 20,
        cfg_scale: float = 7.0,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate video from text prompt and first frame image.

        The first frame is provided, and subsequent frames are generated
        autoregressively conditioned on previous frames.
        """
        return self.text_to_video(
            prompt=prompt,
            num_frames=num_frames,
            fps=fps,
            spatial_h=spatial_h,
            spatial_w=spatial_w,
            num_steps_per_stage=num_steps_per_stage,
            cfg_scale=cfg_scale,
            seed=seed,
            first_frame_image=image,
        )

    def save_video(
        self,
        video: torch.Tensor,
        output_path: str,
        fps: int = 24,
    ):
        """Save video tensor to file."""
        video_np = (video * 255).clamp(0, 255).to(torch.uint8)

        if video_np.shape[1] == 3:
            video_np = video_np.permute(0, 2, 3, 1)

        try:
            import imageio
            writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)
            for frame in video_np:
                writer.append_data(frame.numpy())
            writer.close()
        except ImportError:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            frames_dir = output_path.replace(".mp4", "_frames")
            os.makedirs(frames_dir, exist_ok=True)
            for i, frame in enumerate(video_np):
                img = Image.fromarray(frame.numpy())
                img.save(os.path.join(frames_dir, f"frame_{i:04d}.png"))


def main():
    parser = argparse.ArgumentParser(description="Pyramidal Flow Matching Inference")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--vae_checkpoint", type=str, default=None, help="VAE checkpoint path")
    parser.add_argument("--output", type=str, default="output.mp4", help="Output video path")
    parser.add_argument("--num_frames", type=int, default=121, help="Number of frames (121=5s, 241=10s at 24fps)")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second")
    parser.add_argument("--height", type=int, default=768, help="Video height")
    parser.add_argument("--width", type=int, default=768, help="Video width")
    parser.add_argument("--cfg_scale", type=float, default=7.0, help="CFG scale")
    parser.add_argument("--num_steps", type=int, default=20, help="ODE steps per stage")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--first_frame", type=str, default=None, help="First frame image path")
    args = parser.parse_args()

    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = MMDiT(
        num_layers=config.dit.num_layers,
        hidden_size=config.dit.hidden_size,
        num_heads=config.dit.num_heads,
        head_dim=config.dit.head_dim,
        ff_mult=config.dit.ff_mult,
        patch_size=config.dit.patch_size,
        in_channels=config.dit.in_channels,
        out_channels=config.dit.out_channels,
        pooled_text_dim=config.dit.pooled_text_dim,
        context_dim=config.dit.context_dim,
        clip_dim=config.dit.clip_dim,
        rope_theta=config.dit.rope_theta,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load VAE
    vae = None
    if args.vae_checkpoint:
        vae = VideoVAE(
            in_channels=config.vae.in_channels,
            latent_channels=config.vae.latent_channels,
            base_channels=config.vae.base_channels,
            channel_multipliers=config.vae.channel_multipliers,
            num_res_blocks=config.vae.num_res_blocks,
            temporal_downsample=config.vae.temporal_downsample,
            spatial_downsample=config.vae.spatial_downsample,
            kl_weight=config.vae.kl_weight,
            dropout=config.vae.dropout,
        )
        vae_checkpoint = torch.load(args.vae_checkpoint, map_location=device)
        vae.load_state_dict(vae_checkpoint["model_state_dict"])

    # Text encoders
    t5_model, t5_tokenizer = None, None
    clip_model, clip_tokenizer = None, None
    try:
        from transformers import T5EncoderModel, T5Tokenizer
        t5_model = T5EncoderModel.from_pretrained(config.data.t5_model).to(device)
        t5_tokenizer = T5Tokenizer.from_pretrained(config.data.t5_model)
        t5_model.eval()
    except Exception:
        pass

    try:
        from transformers import CLIPTextModel, CLIPTokenizer
        clip_model = CLIPTextModel.from_pretrained(config.data.clip_model).to(device)
        clip_tokenizer = CLIPTokenizer.from_pretrained(config.data.clip_model)
        clip_model.eval()
    except Exception:
        pass

    # Inference
    engine = PyramidalFlowInference(
        config=config,
        model=model,
        vae=vae,
        t5_model=t5_model,
        t5_tokenizer=t5_tokenizer,
        clip_model=clip_model,
        clip_tokenizer=clip_tokenizer,
        device=device,
    )

    first_frame = None
    if args.first_frame:
        first_frame = Image.open(args.first_frame).convert("RGB")

    video = engine.text_to_video(
        prompt=args.prompt,
        num_frames=args.num_frames,
        fps=args.fps,
        spatial_h=args.height,
        spatial_w=args.width,
        num_steps_per_stage=args.num_steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        first_frame_image=first_frame,
    )

    engine.save_video(video, args.output, fps=args.fps)
    print(f"Video saved to {args.output}")


if __name__ == "__main__":
    main()

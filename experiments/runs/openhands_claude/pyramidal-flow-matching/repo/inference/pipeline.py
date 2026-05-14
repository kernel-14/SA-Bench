"""Inference pipeline for pyramidal flow matching.

Supports:
- Text-to-image generation
- Text-to-video generation (5s/10s at 768p, 24fps)
- Text-conditioned image-to-video generation

The pipeline uses the pyramidal flow scheduler with:
- K=3 pyramid stages
- Euler ODE solver within each stage
- Corrective renoising at jump points
- Classifier-free guidance
- Autoregressive video generation with temporal pyramid history
"""

import math
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import ModelConfig
from data.text_encoder import TextEncoder
from model.dit import MMDiT
from model.vae import VideoVAE
from pyramid_flow.scheduler import PyramidFlowScheduler
from pyramid_flow.spatial_pyramid import downsample_latent, upsample_latent


class PyramidFlowPipeline:
    """End-to-end inference pipeline for pyramidal flow matching.

    Handles text-to-image, text-to-video, and image-to-video generation.
    """

    def __init__(
        self,
        dit: MMDiT,
        vae: VideoVAE,
        text_encoder: TextEncoder,
        config: ModelConfig,
        device: torch.device = None,
    ):
        self.dit = dit
        self.vae = vae
        self.text_encoder = text_encoder
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dit.eval()
        self.vae.eval()

        self.scheduler = PyramidFlowScheduler(
            num_stages=config.pyramid.num_stages,
            num_inference_steps=config.inference.num_inference_steps,
            stage_range=config.pyramid.stage_range,
            upsample_mode=config.pyramid.upsample_mode,
            downsample_mode=config.pyramid.downsample_mode,
        )

        # Cache null embeddings
        self._null_t5 = None
        self._null_clip = None

    def _get_null_embeddings(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached null embeddings for CFG."""
        if self._null_t5 is None:
            with torch.no_grad():
                self._null_t5, self._null_clip = self.text_encoder.get_null_embeddings(
                    1, self.device
                )
        null_t5 = self._null_t5.expand(batch_size, -1, -1)
        null_clip = self._null_clip.expand(batch_size, -1)
        return null_t5, null_clip

    def _make_velocity_fn(
        self,
        t5_embeds: torch.Tensor,
        clip_pooled: torch.Tensor,
        null_t5: torch.Tensor,
        null_clip: torch.Tensor,
        cfg_scale: float,
    ) -> Callable:
        """Create a velocity prediction function for the scheduler.

        The scheduler calls this with (x, t, t5, clip, num_frames, **kwargs).
        We ignore the t5/clip args from the scheduler and use the pre-encoded ones.
        """
        dit = self.dit

        def velocity_fn(
            x: torch.Tensor,
            t: torch.Tensor,
            t5: torch.Tensor,
            clip: torch.Tensor,
            num_frames: int = 1,
            history_frames: Optional[List] = None,
            history_frame_indices: Optional[List] = None,
        ) -> torch.Tensor:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return dit.forward_with_cfg(
                    x=x,
                    timestep=t,
                    t5_embeds=t5_embeds,
                    clip_pooled=clip_pooled,
                    null_t5_embeds=null_t5,
                    null_clip_pooled=null_clip,
                    cfg_scale=cfg_scale,
                    num_frames=num_frames,
                )
        return velocity_fn

    @torch.no_grad()
    def generate_image(
        self,
        prompt: Union[str, List[str]],
        height: int = 768,
        width: int = 768,
        num_inference_steps: int = 20,
        cfg_scale: float = 7.5,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """Generate images from text prompts.

        Args:
            prompt: text prompt(s)
            height: output image height
            width: output image width
            num_inference_steps: number of ODE steps
            cfg_scale: classifier-free guidance scale
            seed: random seed for reproducibility

        Returns:
            list of PIL Images
        """
        if isinstance(prompt, str):
            prompt = [prompt]
        B = len(prompt)

        if seed is not None:
            torch.manual_seed(seed)

        # Encode text
        t5_embeds, clip_pooled = self.text_encoder.encode_text(prompt, self.device)
        null_t5, null_clip = self._get_null_embeddings(B)

        # Latent shape (after VAE compression: 8x spatial)
        C = self.config.vae.latent_channels
        H_lat = height // 8
        W_lat = width // 8

        # Update scheduler steps
        self.scheduler.num_inference_steps = num_inference_steps

        # Generate latent
        latent = self.scheduler.sample_image(
            model_fn=self._make_velocity_fn(t5_embeds, clip_pooled, null_t5, null_clip, cfg_scale),
            shape=(B, C, H_lat, W_lat),
            t5_embeds=t5_embeds,
            clip_pooled=clip_pooled,
            null_t5_embeds=null_t5,
            null_clip_pooled=null_clip,
            cfg_scale=cfg_scale,
            device=self.device,
            dtype=torch.bfloat16,
        )

        # Decode latent to pixels
        latent = latent.unsqueeze(2)  # (B, C, 1, H, W)
        pixels = self.vae.decode(latent)  # (B, C, 1, H, W)
        pixels = pixels.squeeze(2)  # (B, C, H, W)

        # Convert to PIL images
        images = self._tensor_to_pil(pixels)
        return images

    @torch.no_grad()
    def generate_video(
        self,
        prompt: Union[str, List[str]],
        height: int = 768,
        width: int = 768,
        num_frames: int = 121,
        fps: int = 24,
        num_inference_steps: int = 20,
        cfg_scale: float = 7.5,
        frames_per_chunk: int = 8,
        seed: Optional[int] = None,
        first_frame: Optional[torch.Tensor] = None,
    ) -> List[List[Image.Image]]:
        """Generate videos from text prompts.

        Args:
            prompt: text prompt(s)
            height: output video height
            width: output video width
            num_frames: total number of frames (121 for 5s, 241 for 10s at 24fps)
            fps: frames per second
            num_inference_steps: ODE steps per chunk
            cfg_scale: CFG scale
            frames_per_chunk: frames per autoregressive chunk
            seed: random seed
            first_frame: optional (B, C, H, W) first frame for image-to-video

        Returns:
            list of video frame lists (one per batch item)
        """
        if isinstance(prompt, str):
            prompt = [prompt]
        B = len(prompt)

        if seed is not None:
            torch.manual_seed(seed)

        # Encode text
        t5_embeds, clip_pooled = self.text_encoder.encode_text(prompt, self.device)
        null_t5, null_clip = self._get_null_embeddings(B)

        C = self.config.vae.latent_channels
        H_lat = height // 8
        W_lat = width // 8

        self.scheduler.num_inference_steps = num_inference_steps

        # Encode first frame if provided (image-to-video)
        first_frame_latent = None
        if first_frame is not None:
            first_frame_latent = self.vae.encode(
                first_frame.unsqueeze(2).to(self.device), sample=False
            ).squeeze(2)

        # Generate video latents autoregressively
        frame_latents = self.scheduler.sample_video(
            model_fn=self._make_velocity_fn(t5_embeds, clip_pooled, null_t5, null_clip, cfg_scale),
            shape=(B, C, H_lat, W_lat),
            t5_embeds=t5_embeds,
            clip_pooled=clip_pooled,
            null_t5_embeds=null_t5,
            null_clip_pooled=null_clip,
            cfg_scale=cfg_scale,
            num_frames=num_frames,
            frames_per_chunk=frames_per_chunk,
            device=self.device,
            dtype=torch.bfloat16,
            first_frame=first_frame_latent,
        )

        # Decode all frame latents
        all_frames = []
        for frame_latent in frame_latents:
            frame_latent_3d = frame_latent.unsqueeze(2)  # (B, C, 1, H, W)
            pixels = self.vae.decode(frame_latent_3d).squeeze(2)  # (B, C, H, W)
            all_frames.append(pixels)

        # Organize by batch item
        videos = []
        for b in range(B):
            video_frames = [self._tensor_to_pil([f[b]])[0] for f in all_frames]
            videos.append(video_frames)

        return videos

    def _tensor_to_pil(self, tensors: torch.Tensor) -> List[Image.Image]:
        """Convert (B, C, H, W) tensor in [-1, 1] to PIL images."""
        images = []
        for t in tensors:
            t = (t.float().clamp(-1, 1) + 1) / 2  # [-1, 1] -> [0, 1]
            t = (t * 255).byte().permute(1, 2, 0).cpu().numpy()
            images.append(Image.fromarray(t))
        return images

    def save_video(
        self,
        frames: List[Image.Image],
        output_path: str,
        fps: int = 24,
    ):
        """Save video frames to an MP4 file."""
        import imageio

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with imageio.get_writer(str(output_path), fps=fps, codec="libx264") as writer:
            for frame in frames:
                writer.append_data(np.array(frame))

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str,
        config: Optional[ModelConfig] = None,
        device: torch.device = None,
    ) -> "PyramidFlowPipeline":
        """Load pipeline from a checkpoint directory."""
        from config import get_default_config

        if config is None:
            config = get_default_config()

        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_dir = Path(checkpoint_dir)

        # Load DiT
        dit = MMDiT(
            hidden_size=config.dit.hidden_size,
            num_layers=config.dit.num_layers,
            num_heads=config.dit.num_heads,
            mlp_ratio=config.dit.mlp_ratio,
            in_channels=config.dit.in_channels,
            patch_size=config.dit.patch_size,
            context_dim=config.dit.context_dim,
            qk_norm=config.dit.qk_norm,
            dropout=config.dit.dropout,
            use_causal_attention=config.dit.use_causal_attention,
        )

        dit_path = checkpoint_dir / "model.pt"
        if dit_path.exists():
            ckpt = torch.load(dit_path, map_location="cpu")
            state_dict = ckpt.get("model_state_dict", ckpt)
            dit.load_state_dict(state_dict, strict=False)

        dit = dit.to(device)

        # Load VAE
        vae = VideoVAE(
            in_channels=config.vae.in_channels,
            out_channels=config.vae.out_channels,
            latent_channels=config.vae.latent_channels,
            base_channels=config.vae.base_channels,
            channel_multipliers=tuple(config.vae.channel_multipliers),
            num_res_blocks=config.vae.num_res_blocks,
        )

        vae_path = checkpoint_dir / "vae.pt"
        if vae_path.exists():
            vae.load_state_dict(torch.load(vae_path, map_location="cpu"))

        vae = vae.to(device)

        # Load text encoder
        text_encoder = TextEncoder(
            t5_model_name=config.t5_model,
            clip_model_name=config.clip_model,
            max_length=config.max_text_length,
        ).to(device)

        return cls(dit=dit, vae=vae, text_encoder=text_encoder, config=config, device=device)

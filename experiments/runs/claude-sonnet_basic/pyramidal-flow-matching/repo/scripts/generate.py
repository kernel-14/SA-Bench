"""
Video/Image generation script for Pyramidal Flow Matching.

Usage:
    # Text-to-video generation
    python scripts/generate.py \
        --prompt "A beautiful sunset over the ocean" \
        --output output.mp4 \
        --num_frames 121 \
        --height 768 \
        --width 768 \
        --fps 24
    
    # Image-to-video generation
    python scripts/generate.py \
        --prompt "The waves crash against the shore" \
        --image input.jpg \
        --output output.mp4
    
    # Text-to-image generation
    python scripts/generate.py \
        --prompt "A beautiful landscape" \
        --output output.png \
        --mode image
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import torch
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.pyramid_dit import PyramidDiT
from models.vae_3d import VideoVAE
from inference.pipeline import PyramidFlowPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_video(frames: torch.Tensor, output_path: str, fps: int = 24):
    """
    Save video frames to file.
    
    Args:
        frames: (B, C, T, H, W) tensor with values in [0, 1]
        output_path: Output file path
        fps: Frames per second
    """
    try:
        import imageio
        
        # Take first batch item
        video = frames[0]  # (C, T, H, W)
        video = video.permute(1, 2, 3, 0)  # (T, H, W, C)
        video = (video.cpu().numpy() * 255).astype(np.uint8)
        
        imageio.mimwrite(output_path, video, fps=fps, quality=8)
        logger.info(f"Saved video to {output_path}")
    except ImportError:
        logger.warning("imageio not available, saving frames as images instead")
        save_frames(frames, output_path)


def save_frames(frames: torch.Tensor, output_dir: str):
    """Save individual frames as images."""
    os.makedirs(output_dir, exist_ok=True)
    
    video = frames[0]  # (C, T, H, W)
    T = video.shape[1]
    
    for t in range(T):
        frame = video[:, t]  # (C, H, W)
        frame = frame.permute(1, 2, 0)  # (H, W, C)
        frame = (frame.cpu().numpy() * 255).astype(np.uint8)
        
        img = Image.fromarray(frame)
        img.save(os.path.join(output_dir, f'frame_{t:04d}.png'))
    
    logger.info(f"Saved {T} frames to {output_dir}")


def save_image(image: torch.Tensor, output_path: str):
    """
    Save image tensor to file.
    
    Args:
        image: (B, C, H, W) tensor with values in [0, 1]
        output_path: Output file path
    """
    img = image[0]  # (C, H, W)
    img = img.permute(1, 2, 0)  # (H, W, C)
    img = (img.cpu().numpy() * 255).astype(np.uint8)
    
    pil_img = Image.fromarray(img)
    pil_img.save(output_path)
    logger.info(f"Saved image to {output_path}")


def load_image(image_path: str, height: int, width: int) -> torch.Tensor:
    """Load and preprocess an image for conditioning."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((width, height), Image.LANCZOS)
    
    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # (C, H, W)
    img_tensor = img_tensor * 2 - 1  # Normalize to [-1, 1]
    
    return img_tensor.unsqueeze(0)  # (1, C, H, W)


def build_pipeline(
    checkpoint_path: str,
    device: torch.device,
    num_pyramid_stages: int = 3,
    num_inference_steps: int = 20,
) -> PyramidFlowPipeline:
    """
    Build the inference pipeline from a checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Inference device
        num_pyramid_stages: Number of pyramid stages
        num_inference_steps: Number of ODE steps per stage
    
    Returns:
        Initialized pipeline
    """
    # Build models
    model = PyramidDiT(
        in_channels=16,
        hidden_dim=1536,
        num_layers=24,
        num_heads=24,
        num_pyramid_stages=num_pyramid_stages,
    )
    
    vae = VideoVAE(
        in_channels=3,
        latent_channels=16,
    )
    
    # Load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logger.info(f"Loaded model from {checkpoint_path}")
    
    # Load text encoders (T5 and CLIP)
    try:
        from transformers import T5EncoderModel, T5Tokenizer, CLIPTextModel, CLIPTokenizer
        
        t5_model = T5EncoderModel.from_pretrained('google/t5-v1_1-xxl')
        t5_tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-xxl')
        
        clip_model = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14')
        clip_tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
        
    except Exception as e:
        logger.warning(f"Could not load text encoders: {e}")
        logger.warning("Using dummy text encoders for testing")
        
        # Create dummy text encoders for testing
        class DummyTextEncoder(torch.nn.Module):
            def __init__(self, output_dim, seq_len=77):
                super().__init__()
                self.output_dim = output_dim
                self.seq_len = seq_len
            
            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                B = input_ids.shape[0] if input_ids is not None else 1
                
                class Output:
                    def __init__(self, last_hidden_state, pooler_output):
                        self.last_hidden_state = last_hidden_state
                        self.pooler_output = pooler_output
                
                return Output(
                    last_hidden_state=torch.randn(B, self.seq_len, self.output_dim),
                    pooler_output=torch.randn(B, self.output_dim),
                )
        
        class DummyTokenizer:
            def __call__(self, text, **kwargs):
                if isinstance(text, str):
                    text = [text]
                B = len(text)
                
                class Tokens:
                    def __init__(self, B, max_length):
                        self.input_ids = torch.zeros(B, max_length, dtype=torch.long)
                        self.attention_mask = torch.ones(B, max_length, dtype=torch.long)
                    
                    def to(self, device):
                        self.input_ids = self.input_ids.to(device)
                        self.attention_mask = self.attention_mask.to(device)
                        return self
                
                return Tokens(B, kwargs.get('max_length', 77))
        
        t5_model = DummyTextEncoder(4096, seq_len=256)
        t5_tokenizer = DummyTokenizer()
        clip_model = DummyTextEncoder(768, seq_len=77)
        clip_tokenizer = DummyTokenizer()
    
    # Build pipeline
    pipeline = PyramidFlowPipeline(
        model=model,
        vae=vae,
        text_encoder_t5=t5_model,
        text_encoder_clip=clip_model,
        tokenizer_t5=t5_tokenizer,
        tokenizer_clip=clip_tokenizer,
        num_pyramid_stages=num_pyramid_stages,
        num_inference_steps=num_inference_steps,
        device=device,
    )
    
    return pipeline


def main():
    parser = argparse.ArgumentParser(description='Generate videos/images with Pyramidal Flow Matching')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt')
    parser.add_argument('--negative_prompt', type=str, default='', help='Negative prompt')
    parser.add_argument('--output', type=str, default='output.mp4', help='Output file path')
    parser.add_argument('--mode', type=str, default='video', choices=['video', 'image'],
                        help='Generation mode')
    parser.add_argument('--image', type=str, default=None, help='Input image for image-to-video')
    parser.add_argument('--num_frames', type=int, default=121, help='Number of frames (video mode)')
    parser.add_argument('--height', type=int, default=768, help='Output height')
    parser.add_argument('--width', type=int, default=768, help='Output width')
    parser.add_argument('--fps', type=int, default=24, help='Frames per second')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='CFG guidance scale')
    parser.add_argument('--num_inference_steps', type=int, default=20, help='ODE steps per stage')
    parser.add_argument('--num_pyramid_stages', type=int, default=3, help='Number of pyramid stages')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--checkpoint', type=str, default=None, help='Model checkpoint path')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cpu/cuda)')
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Build pipeline
    pipeline = build_pipeline(
        checkpoint_path=args.checkpoint,
        device=device,
        num_pyramid_stages=args.num_pyramid_stages,
        num_inference_steps=args.num_inference_steps,
    )
    
    # Load conditioning image if provided
    image_condition = None
    if args.image:
        image_condition = load_image(args.image, args.height, args.width)
        image_condition = image_condition.to(device)
    
    # Generate
    if args.mode == 'video':
        logger.info(f"Generating {args.num_frames} frames at {args.height}x{args.width}...")
        
        video = pipeline.generate_video(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt if args.negative_prompt else None,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            fps=args.fps,
            seed=args.seed,
            image_condition=image_condition,
        )
        
        save_video(video, args.output, fps=args.fps)
    
    else:  # image mode
        logger.info(f"Generating image at {args.height}x{args.width}...")
        
        image = pipeline.generate_image(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt if args.negative_prompt else None,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
        )
        
        save_image(image, args.output)


if __name__ == '__main__':
    main()

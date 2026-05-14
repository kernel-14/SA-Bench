#!/usr/bin/env python3
"""
Generation script for Hi-MAR.

Usage:
    python scripts/generate.py --config configs/imagenet_himar_b.yaml \
        --checkpoint checkpoints/final_model.pt \
        --num_images 16 --output_dir generated/
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torchvision.utils import save_image
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from himar import HiMAR
from himar.model import create_himar_model
from himar.data import get_vae, get_clip_text_encoder


def parse_args():
    parser = argparse.ArgumentParser(description='Generate images with Hi-MAR')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--num_images', type=int, default=16, help='Number of images to generate')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for generation')
    parser.add_argument('--output_dir', type=str, default='generated/', help='Output directory')
    parser.add_argument('--class_idx', type=int, nargs='+', default=None, help='Class indices for conditional generation')
    parser.add_argument('--prompt', type=str, default=None, help='Text prompt for text-to-image generation')
    parser.add_argument('--cfg_scale', type=float, default=None, help='Classifier-free guidance scale')
    parser.add_argument('--phase1_steps', type=int, default=None, help='Steps for phase 1')
    parser.add_argument('--phase2_steps', type=int, default=None, help='Steps for phase 2')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print("Loading Hi-MAR model...")
    model_cfg = config['model']
    model = create_himar_model(model_cfg)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load VAE for decoding
    vae = get_vae('kl-16')
    if vae is not None:
        vae = vae.to(device)
        vae.eval()
    
    # Determine inference settings
    phase1_steps = args.phase1_steps or config['inference'].get('phase1_steps', 32)
    phase2_steps = args.phase2_steps or config['inference'].get('phase2_steps', 4)
    cfg_scale = args.cfg_scale or config['inference'].get('cfg_scale', 1.0)
    
    # Prepare conditioning
    class_idx = None
    context_embeds = None
    
    if args.class_idx is not None:
        # Replicate class indices to match num_images
        repeats = args.num_images // len(args.class_idx) + 1
        class_idx = torch.tensor(args.class_idx * repeats)[:args.num_images]
        print(f"Generating images for classes: {class_idx.tolist()}")
    
    if args.prompt is not None:
        # Encode text prompt
        text_encoder, tokenizer = get_clip_text_encoder()
        if text_encoder is not None:
            text_encoder = text_encoder.to(device)
            text_encoder.eval()
            
            # Repeat prompt for batch
            prompts = [args.prompt] * args.num_images
            
            with torch.no_grad():
                # Tokenize
                if hasattr(text_encoder, 'encode_text'):
                    # OpenAI CLIP
                    import clip
                    tokens = clip.tokenize(prompts, truncate=True).to(device)
                    context_embeds = text_encoder.encode_text(tokens).float()
                else:
                    # HuggingFace CLIP
                    tokens = tokenizer(prompts, padding=True, truncation=True, return_tensors='pt').to(device)
                    outputs = text_encoder(**tokens)
                    context_embeds = outputs.last_hidden_state
            
            print(f"Generating images for prompt: '{args.prompt}'")
    
    # Generate images
    print(f"Generating {args.num_images} images...")
    print(f"Phase 1 steps: {phase1_steps}, Phase 2 steps: {phase2_steps}")
    
    all_images = []
    
    for i in range(0, args.num_images, args.batch_size):
        bs = min(args.batch_size, args.num_images - i)
        print(f"Generating batch {i // args.batch_size + 1}/{(args.num_images + args.batch_size - 1) // args.batch_size}")
        
        batch_class = class_idx[i:i+bs].to(device) if class_idx is not None else None
        batch_context = context_embeds[i:i+bs].to(device) if context_embeds is not None else None
        
        with torch.no_grad():
            x_low, x_high = model.generate(
                batch_size=bs,
                class_idx=batch_class,
                context_embeds=batch_context,
                phase1_steps=phase1_steps,
                phase2_steps=phase2_steps,
                cfg_scale=cfg_scale,
                device=device,
            )
        
        # Decode latents to images
        if vae is not None:
            H = W = int(x_high.shape[1] ** 0.5)
            latent = x_high.permute(0, 2, 1).reshape(bs, -1, H, W)
            images = vae.decode(latent).sample
            images = torch.clamp(images, -1, 1)
            # Normalize to [0, 1] for saving
            images = (images + 1) / 2.0
        else:
            # Reshape tokens as images for visualization
            H = W = int(x_high.shape[1] ** 0.5)
            images = x_high.permute(0, 2, 1).reshape(bs, -1, H, W)
            images = (images - images.min()) / (images.max() - images.min())
        
        all_images.append(images.cpu())
    
    all_images = torch.cat(all_images, dim=0)[:args.num_images]
    
    # Save images
    os.makedirs(args.output_dir, exist_ok=True)
    
    for idx in range(all_images.shape[0]):
        save_path = os.path.join(args.output_dir, f'generated_{idx:04d}.png')
        save_image(all_images[idx], save_path)
    
    # Also save a grid
    grid_path = os.path.join(args.output_dir, 'grid.png')
    save_image(all_images, grid_path, nrow=int(args.num_images ** 0.5))
    
    print(f"Generated images saved to {args.output_dir}")


if __name__ == '__main__':
    main()

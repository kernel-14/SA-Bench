"""
Generation script for Hi-MAR.

Generates images using a trained Hi-MAR model and computes FID/IS metrics.

Usage:
    # Class-conditional generation (ImageNet)
    python generate.py --task imagenet --checkpoint output/checkpoint_epoch0799.pth \
        --num_samples 50000 --cfg_scale 1.5

    # Text-to-image generation (MS-COCO)
    python generate.py --task coco --checkpoint output/checkpoint_epoch0799.pth \
        --num_samples 30000 --cfg_scale 1.5
"""

import os
import math
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm

from models.hi_mar import HiMAR, HiMAR_B, HiMAR_L, HiMAR_H
from models.hi_mar_t2i import HiMARText
from utils.vae import load_vae, VAETokenizer


def get_args():
    parser = argparse.ArgumentParser('Hi-MAR Generation')

    # Task
    parser.add_argument('--task', type=str, default='imagenet',
                        choices=['imagenet', 'coco'])

    # Model
    parser.add_argument('--model', type=str, default='hi_mar_b',
                        choices=['hi_mar_b', 'hi_mar_l', 'hi_mar_h', 'hi_mar_s'])
    parser.add_argument('--checkpoint', type=str, required=True)

    # VAE
    parser.add_argument('--vae_path', type=str, default='pretrained/kl16.ckpt')
    parser.add_argument('--vae_stride', type=int, default=16)

    # Generation settings
    parser.add_argument('--num_samples', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--low_res_img_size', type=int, default=128)

    # Sampling parameters
    parser.add_argument('--num_steps_phase1', type=int, default=32,
                        help='Number of autoregressive steps for phase 1')
    parser.add_argument('--num_steps_phase2', type=int, default=4,
                        help='Number of autoregressive steps for phase 2')
    parser.add_argument('--cfg_scale', type=float, default=1.5,
                        help='Classifier-free guidance scale')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--diff_temperature', type=float, default=1.0)

    # For ImageNet
    parser.add_argument('--num_classes', type=int, default=1000)

    # For COCO
    parser.add_argument('--coco_path', type=str, default=None,
                        help='Path to COCO dataset for text prompts')
    parser.add_argument('--num_prompts', type=int, default=30000,
                        help='Number of prompts to use for COCO evaluation')

    # Output
    parser.add_argument('--output_dir', type=str, default='generated_images')
    parser.add_argument('--save_images', action='store_true', default=False)

    # Evaluation
    parser.add_argument('--compute_fid', action='store_true', default=True)
    parser.add_argument('--ref_path', type=str, default=None,
                        help='Path to reference images for FID computation')

    return parser.parse_args()


def load_model(args):
    """Load Hi-MAR model from checkpoint."""
    common_kwargs = dict(
        img_size=args.img_size,
        low_res_img_size=args.low_res_img_size,
        patch_size=args.vae_stride,
        in_channels=16,
    )

    if args.task == 'imagenet':
        common_kwargs['num_classes'] = args.num_classes

        if args.model == 'hi_mar_b':
            model = HiMAR_B(**common_kwargs)
        elif args.model == 'hi_mar_l':
            model = HiMAR_L(**common_kwargs)
        elif args.model == 'hi_mar_h':
            model = HiMAR_H(**common_kwargs)
        else:
            raise ValueError(f'Unknown model: {args.model}')
    else:
        model = HiMARText(
            img_size=args.img_size,
            low_res_img_size=args.low_res_img_size,
            patch_size=args.vae_stride,
            in_channels=16,
        )

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    # Try EMA model first
    if 'ema' in checkpoint:
        model.load_state_dict(checkpoint['ema']['ema_model'])
        print('Loaded EMA model')
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
        # Handle DDP wrapper
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print('Loaded model')
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


@torch.no_grad()
def generate_imagenet(model, vae_tokenizer, args):
    """Generate images for ImageNet class-conditional generation."""
    device = next(model.parameters()).device
    all_images = []

    # Generate equal number of images per class
    samples_per_class = args.num_samples // args.num_classes
    remainder = args.num_samples % args.num_classes

    class_labels = []
    for c in range(args.num_classes):
        n = samples_per_class + (1 if c < remainder else 0)
        class_labels.extend([c] * n)

    # Generate in batches
    for i in tqdm(range(0, len(class_labels), args.batch_size), desc='Generating'):
        batch_labels = class_labels[i:i + args.batch_size]
        batch_labels = torch.tensor(batch_labels, device=device)

        # Generate tokens
        tokens = model.generate(
            batch_labels,
            num_steps_phase1=args.num_steps_phase1,
            num_steps_phase2=args.num_steps_phase2,
            cfg_scale=args.cfg_scale,
            temperature=args.temperature,
            diff_temperature=args.diff_temperature,
        )

        # Decode to images
        images = vae_tokenizer.decode(tokens, args.img_size)
        images = (images.clamp(-1, 1) + 1) / 2  # [0, 1]
        images = (images * 255).byte().cpu().numpy()
        images = images.transpose(0, 2, 3, 1)  # [B, H, W, C]

        all_images.append(images)

    return np.concatenate(all_images, axis=0)


@torch.no_grad()
def generate_coco(model, vae_tokenizer, text_embeddings, args):
    """Generate images for COCO text-to-image generation."""
    device = next(model.parameters()).device
    all_images = []

    for i in tqdm(range(0, len(text_embeddings), args.batch_size), desc='Generating'):
        batch_text = text_embeddings[i:i + args.batch_size]
        batch_text = batch_text.to(device)

        # Generate tokens
        tokens = model.generate(
            batch_text,
            num_steps_phase1=args.num_steps_phase1,
            num_steps_phase2=args.num_steps_phase2,
            cfg_scale=args.cfg_scale,
            temperature=args.temperature,
            diff_temperature=args.diff_temperature,
        )

        # Decode to images
        images = vae_tokenizer.decode(tokens, args.img_size)
        images = (images.clamp(-1, 1) + 1) / 2  # [0, 1]
        images = (images * 255).byte().cpu().numpy()
        images = images.transpose(0, 2, 3, 1)  # [B, H, W, C]

        all_images.append(images)

    return np.concatenate(all_images, axis=0)


def compute_fid(generated_images, ref_path, device):
    """Compute FID score between generated and reference images."""
    try:
        from pytorch_fid import fid_score
        import tempfile

        # Save generated images to temp directory
        with tempfile.TemporaryDirectory() as gen_dir:
            for i, img in enumerate(generated_images):
                Image.fromarray(img).save(os.path.join(gen_dir, f'{i:06d}.png'))

            fid = fid_score.calculate_fid_given_paths(
                [gen_dir, ref_path],
                batch_size=50,
                device=device,
                dims=2048,
            )
        return fid
    except ImportError:
        print('pytorch_fid not available. Skipping FID computation.')
        return None


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print('Loading model...')
    model = load_model(args)
    model = model.to(device)

    # Load VAE
    print('Loading VAE...')
    vae = load_vae(args.vae_path).to(device)
    vae.eval()
    vae_tokenizer = VAETokenizer(vae, vae_stride=args.vae_stride)

    # Generate images
    print(f'Generating {args.num_samples} images...')

    if args.task == 'imagenet':
        images = generate_imagenet(model, vae_tokenizer, args)
    else:
        # Load COCO text embeddings
        from utils.coco_dataset import COCODataset
        val_dataset = COCODataset(
            args.coco_path,
            img_size=args.img_size,
            split='val',
        )

        # Get text embeddings for evaluation
        text_embeddings = []
        indices = torch.randperm(len(val_dataset))[:args.num_prompts]
        for idx in tqdm(indices, desc='Loading text embeddings'):
            _, text_emb = val_dataset[idx.item()]
            text_embeddings.append(text_emb)
        text_embeddings = torch.stack(text_embeddings)

        images = generate_coco(model, vae_tokenizer, text_embeddings, args)

    print(f'Generated {len(images)} images')

    # Save images if requested
    if args.save_images:
        img_dir = os.path.join(args.output_dir, 'images')
        os.makedirs(img_dir, exist_ok=True)
        for i, img in enumerate(tqdm(images, desc='Saving images')):
            Image.fromarray(img).save(os.path.join(img_dir, f'{i:06d}.png'))
        print(f'Saved images to {img_dir}')

    # Compute FID
    if args.compute_fid and args.ref_path:
        print('Computing FID...')
        fid = compute_fid(images, args.ref_path, device)
        if fid is not None:
            print(f'FID: {fid:.4f}')

            # Save results
            results = {'fid': fid, 'num_samples': len(images)}
            with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
                json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()

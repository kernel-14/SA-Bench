"""
Evaluation script for consistency models.

Usage:
    python evaluate.py --checkpoint checkpoints/cifar10_gc/checkpoint_100000.pt \
                       --config configs/cifar10_gc.yaml \
                       --num_samples 50000
"""

import os
import argparse
import yaml
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models import SongUNet, ConsistencyModel
from evaluation.fid import compute_metrics


def load_model(checkpoint_path, config, device):
    """Load a consistency model from a checkpoint."""
    model_config = config['model']
    dataset_config = config['dataset']
    
    img_resolution = dataset_config['img_resolution']
    in_channels = dataset_config.get('in_channels', 3)
    
    network = SongUNet(
        img_resolution=img_resolution,
        in_channels=in_channels,
        out_channels=in_channels,
        label_dim=dataset_config.get('label_dim', 0),
        model_channels=model_config.get('model_channels', 128),
        channel_mult=model_config.get('channel_mult', [1, 2, 2, 2]),
        num_blocks=model_config.get('num_blocks', 4),
        attn_resolutions=model_config.get('attn_resolutions', [16]),
        dropout=0.0,  # No dropout at eval time
        embedding_type=model_config.get('embedding_type', 'positional'),
    )
    
    model = ConsistencyModel(
        network=network,
        sigma_data=model_config.get('sigma_data', 0.5),
        sigma_min=config['training'].get('sigma_min', 0.002),
        sigma_max=config['training'].get('sigma_max', 80.0),
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to load EMA model first (better quality)
    if 'ema_model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['ema_model_state_dict'])
        print("Loaded EMA model weights")
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model weights")
    
    return model.to(device)


def get_real_samples(config, num_samples, batch_size=256):
    """Get real samples from the dataset."""
    dataset_name = config['dataset']['name']
    data_dir = config['dataset'].get('data_dir', './data')
    img_size = config['dataset']['img_resolution']
    
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    
    if dataset_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform
        )
    elif dataset_name == 'imagenet':
        dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(data_dir, 'train'), transform=transform
        )
    elif dataset_name == 'celeba':
        dataset = torchvision.datasets.CelebA(
            root=data_dir, split='train', download=True, transform=transform
        )
    elif dataset_name == 'lsun_church':
        dataset = torchvision.datasets.LSUN(
            root=data_dir, classes=['church_outdoor_train'], transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    real_samples = []
    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            real_samples.append(batch[0])
        else:
            real_samples.append(batch)
        if sum(x.shape[0] for x in real_samples) >= num_samples:
            break
    
    return torch.cat(real_samples, dim=0)[:num_samples]


def main():
    parser = argparse.ArgumentParser(description='Evaluate consistency model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--num_samples', type=int, default=50000, help='Number of samples')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for generation')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    model = load_model(args.checkpoint, config, device)
    model.eval()
    
    # Get image shape
    img_resolution = config['dataset']['img_resolution']
    in_channels = config['dataset'].get('in_channels', 3)
    
    # Generate samples
    print(f"Generating {args.num_samples} samples...")
    generated = []
    with torch.no_grad():
        for i in range(0, args.num_samples, args.batch_size):
            current_batch_size = min(args.batch_size, args.num_samples - i)
            z = torch.randn(current_batch_size, in_channels, img_resolution, img_resolution, device=device)
            sigma_max = torch.tensor(model.sigma_max, dtype=z.dtype, device=device)
            x = model(z, sigma_max.expand(current_batch_size))
            generated.append(x.cpu())
            if (i // args.batch_size) % 10 == 0:
                print(f"  Generated {i + current_batch_size}/{args.num_samples}")
    
    generated = torch.cat(generated, dim=0)
    
    # Get real samples
    print("Loading real samples...")
    real_samples = get_real_samples(config, args.num_samples, args.batch_size)
    
    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(generated, real_samples, device=str(device))
    
    print("\n=== Evaluation Results ===")
    print(f"FID: {metrics['FID']:.2f}")
    print(f"KID (x10^2): {metrics['KID']:.2f}")
    print(f"IS: {metrics['IS']:.2f}")


if __name__ == '__main__':
    main()

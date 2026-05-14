"""
Script to prepare VTAB-1K data from TensorFlow datasets format to PyTorch format.

The original VTAB-1K uses TensorFlow datasets. This script converts them to
a format compatible with our PyTorch implementation.

Usage:
    python scripts/prepare_vtab.py --output_dir /path/to/vtab_pytorch
    
Note: Requires tensorflow-datasets to be installed.
"""

import os
import sys
import argparse
import json
from pathlib import Path


# VTAB-1K dataset configurations
VTAB_CONFIGS = {
    'caltech101': {
        'tfds_name': 'caltech101',
        'num_classes': 102,
        'group': 'natural',
    },
    'cifar100': {
        'tfds_name': 'cifar100',
        'num_classes': 100,
        'group': 'natural',
    },
    'dtd': {
        'tfds_name': 'dtd',
        'num_classes': 47,
        'group': 'natural',
    },
    'flowers102': {
        'tfds_name': 'oxford_flowers102',
        'num_classes': 102,
        'group': 'natural',
    },
    'pets': {
        'tfds_name': 'oxford_iiit_pet',
        'num_classes': 37,
        'group': 'natural',
    },
    'svhn': {
        'tfds_name': 'svhn_cropped',
        'num_classes': 10,
        'group': 'natural',
    },
    'sun397': {
        'tfds_name': 'sun397',
        'num_classes': 397,
        'group': 'natural',
    },
    'camelyon': {
        'tfds_name': 'patch_camelyon',
        'num_classes': 2,
        'group': 'specialized',
    },
    'eurosat': {
        'tfds_name': 'eurosat',
        'num_classes': 10,
        'group': 'specialized',
    },
    'resisc45': {
        'tfds_name': 'resisc45',
        'num_classes': 45,
        'group': 'specialized',
    },
    'retinopathy': {
        'tfds_name': 'diabetic_retinopathy_detection',
        'num_classes': 5,
        'group': 'specialized',
    },
    'clevr_count': {
        'tfds_name': 'clevr',
        'num_classes': 8,
        'group': 'structured',
        'task': 'count',
    },
    'clevr_distance': {
        'tfds_name': 'clevr',
        'num_classes': 6,
        'group': 'structured',
        'task': 'distance',
    },
    'dmlab': {
        'tfds_name': 'dmlab',
        'num_classes': 6,
        'group': 'structured',
    },
    'kitti': {
        'tfds_name': 'kitti',
        'num_classes': 4,
        'group': 'structured',
    },
    'dsprites_loc': {
        'tfds_name': 'dsprites',
        'num_classes': 16,
        'group': 'structured',
        'task': 'location',
    },
    'dsprites_ori': {
        'tfds_name': 'dsprites',
        'num_classes': 16,
        'group': 'structured',
        'task': 'orientation',
    },
    'smallnorb_azimuth': {
        'tfds_name': 'smallnorb',
        'num_classes': 18,
        'group': 'structured',
        'task': 'azimuth',
    },
    'smallnorb_elevation': {
        'tfds_name': 'smallnorb',
        'num_classes': 9,
        'group': 'structured',
        'task': 'elevation',
    },
}


def prepare_vtab_dataset(dataset_name, output_dir, num_train=800, num_val=200):
    """
    Prepare a single VTAB-1K dataset.
    
    Following the VTAB-1K paper:
    - 1000 training images total (800 train + 200 val for hyperparameter tuning)
    - Full test set for evaluation
    
    Args:
        dataset_name: Name of the dataset
        output_dir: Output directory
        num_train: Number of training samples (default: 800)
        num_val: Number of validation samples (default: 200)
    """
    try:
        import tensorflow_datasets as tfds
        import numpy as np
        from PIL import Image
    except ImportError:
        print("tensorflow-datasets not installed. Please install it to prepare VTAB-1K data.")
        print("pip install tensorflow-datasets")
        return
    
    config = VTAB_CONFIGS[dataset_name]
    dataset_dir = os.path.join(output_dir, dataset_name)
    images_dir = os.path.join(dataset_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    print(f"Preparing {dataset_name}...")
    
    # Load dataset
    try:
        ds = tfds.load(config['tfds_name'], split='train', as_supervised=True)
        ds_test = tfds.load(config['tfds_name'], split='test', as_supervised=True)
    except Exception as e:
        print(f"Failed to load {dataset_name}: {e}")
        return
    
    # Convert to list
    train_data = list(ds.take(num_train + num_val))
    test_data = list(ds_test)
    
    # Save images and create split files
    def save_split(data, split_name, start_idx=0):
        lines = []
        for i, (img, label) in enumerate(data):
            img_array = img.numpy()
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            
            pil_img = Image.fromarray(img_array.astype(np.uint8))
            img_filename = f'{split_name}_{start_idx + i:06d}.jpg'
            img_path = os.path.join(images_dir, img_filename)
            pil_img.save(img_path)
            
            label_val = int(label.numpy())
            lines.append(f'images/{img_filename} {label_val}')
        
        split_file = os.path.join(dataset_dir, f'{split_name}.txt')
        with open(split_file, 'w') as f:
            f.write('\n'.join(lines))
        
        return len(lines)
    
    # Save train and val splits
    n_train = save_split(train_data[:num_train], 'train')
    n_val = save_split(train_data[num_train:], 'val', start_idx=num_train)
    n_test = save_split(test_data, 'test', start_idx=num_train + num_val)
    
    print(f"  {dataset_name}: {n_train} train, {n_val} val, {n_test} test")
    
    # Save metadata
    metadata = {
        'dataset_name': dataset_name,
        'num_classes': config['num_classes'],
        'group': config['group'],
        'num_train': n_train,
        'num_val': n_val,
        'num_test': n_test,
    }
    with open(os.path.join(dataset_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Prepare VTAB-1K datasets')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for prepared datasets')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='Specific datasets to prepare (default: all)')
    parser.add_argument('--num_train', type=int, default=800,
                        help='Number of training samples per dataset')
    parser.add_argument('--num_val', type=int, default=200,
                        help='Number of validation samples per dataset')
    
    args = parser.parse_args()
    
    datasets = args.datasets or list(VTAB_CONFIGS.keys())
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for dataset_name in datasets:
        if dataset_name not in VTAB_CONFIGS:
            print(f"Unknown dataset: {dataset_name}")
            continue
        
        prepare_vtab_dataset(
            dataset_name, args.output_dir,
            num_train=args.num_train, num_val=args.num_val
        )
    
    print(f"\nDone! Datasets saved to {args.output_dir}")


if __name__ == '__main__':
    main()

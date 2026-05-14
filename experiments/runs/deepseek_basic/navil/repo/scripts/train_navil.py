#!/usr/bin/env python3
"""
Training script for NaViL.

Usage:
    python scripts/train_navil.py --config configs/navil_2b.yaml --stage all
    python scripts/train_navil.py --config configs/navil_2b.yaml --stage s1_1
    python scripts/train_navil.py --config configs/navil_9b.yaml --stage all
"""

import argparse
import os
import sys
import logging

import torch
import torch.distributed as dist

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from navil.model import NaViLModel, NaViLConfig, create_navil_2b, create_navil_9b
from navil.trainer import NaViLTrainer, TrainingConfig
from navil.data import ImageCaptionDataset, ConversationDataset, NaViLDataCollator


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train NaViL model')
    parser.add_argument('--config', type=str, default='configs/navil_2b.yaml',
                        help='Path to config file')
    parser.add_argument('--model_size', type=str, default='2b',
                        choices=['2b', '9b'],
                        help='Model size')
    parser.add_argument('--stage', type=str, default='all',
                        choices=['all', 's1_1', 's1_2', 's2'],
                        help='Training stage to run')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Data directory')
    parser.add_argument('--pretrained_llm', type=str, default=None,
                        help='Path to pretrained LLM checkpoint')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank for distributed training')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup distributed training
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
        device = torch.device(f'cuda:{args.local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Using device: {device}")
    logger.info(f"Model size: {args.model_size}")
    logger.info(f"Stage: {args.stage}")
    
    # Create model
    if args.model_size == '2b':
        model = create_navil_2b(llm_checkpoint_path=args.pretrained_llm)
    else:
        model = create_navil_9b(llm_checkpoint_path=args.pretrained_llm)
    
    logger.info(f"Model created with {model.num_parameters:,} parameters")
    logger.info(f"Visual encoder: {model.visual_encoder.num_parameters:,} parameters")
    
    # Create training config
    train_config = TrainingConfig(
        output_dir=args.output_dir,
    )
    
    if args.batch_size is not None:
        train_config.s1_1_batch_size = args.batch_size
    
    if args.lr is not None:
        train_config.s1_1_learning_rate = args.lr
        train_config.s1_2_learning_rate = args.lr
    
    # Initialize trainer
    trainer = NaViLTrainer(
        model=model,
        config=train_config,
        device=device,
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Load data based on stage
    # Note: This assumes tokenizer and data are available
    # In a full reproduction, you would need to set up the tokenizer
    # from InternLM2 or Qwen3
    
    # Placeholder for actual data loading
    logger.info("Setting up data loaders...")
    
    # For demonstration, we create empty dataloaders
    # In practice, replace with actual data paths
    s1_1_loader = None
    s1_2_loader = None
    s2_loader = None
    
    logger.info("Data loaders configured (needs actual data paths)")
    
    # Run training
    if args.stage == 'all':
        logger.info("Running full training pipeline...")
        results = trainer.train_full(
            s1_1_dataloader=s1_1_loader,
            s1_2_dataloader=s1_2_loader,
            s2_dataloader=s2_loader,
        )
    elif args.stage == 's1_1':
        logger.info("Running Stage 1.1 only...")
        results = trainer.train_stage1_1(s1_1_loader)
    elif args.stage == 's1_2':
        logger.info("Running Stage 1.2 only...")
        results = trainer.train_stage1_2(s1_2_loader)
    elif args.stage == 's2':
        logger.info("Running Stage 2 only...")
        results = trainer.train_stage2(s2_loader)
    
    logger.info("Training complete!")
    
    # Save final model
    final_path = os.path.join(args.output_dir, 'navil_final.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model.config,
    }, final_path)
    logger.info(f"Final model saved to {final_path}")


if __name__ == '__main__':
    main()

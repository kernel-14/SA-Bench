"""
Main training script for NaViL.

Usage:
    torchrun --nproc_per_node=8 scripts/train.py \
        --stage pretrain_1a \
        --model_name NaViL-2B \
        --llm_init internlm/internlm2-1_8b \
        --output_dir ./outputs/navil_2b_stage1a \
        --data_path ./data/pretrain_stage1a.json \
        --per_device_batch_size 8 \
        --gradient_accumulation_steps 110 \
        --learning_rate 1e-4 \
        --bf16
"""

import os
import sys
import argparse
import logging
import json
import math
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from navil.model import NaViLModel, NaViLConfig
from navil.trainer import NaViLTrainer, TrainingConfig
from navil.data import ImageProcessor, PretrainDataset, SFTDataset, collate_fn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train NaViL")

    # Stage
    parser.add_argument("--stage", type=str, required=True,
                        choices=["pretrain_1a", "pretrain_1b", "sft"])

    # Model
    parser.add_argument("--model_name", type=str, default="NaViL-2B",
                        choices=["NaViL-2B", "NaViL-9B"])
    parser.add_argument("--llm_init", type=str, default=None,
                        help="Path or HuggingFace ID for LLM initialization")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from")

    # Data
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_seq_len", type=int, default=4096)

    # Training
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_epochs", type=int, default=1)

    # Precision
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")

    # Image
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--use_multiscale", type=lambda x: x.lower() == "true",
                        default=True)

    # Logging
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=1000)

    # Distributed
    parser.add_argument("--local_rank", type=int, default=-1)

    return parser.parse_args()


def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def build_model(args) -> NaViLModel:
    """Build NaViL model."""
    if args.model_name == "NaViL-2B":
        config = NaViLConfig.navil_2b()
    elif args.model_name == "NaViL-9B":
        config = NaViLConfig.navil_9b()
    else:
        raise ValueError(f"Unknown model: {args.model_name}")

    model = NaViLModel(config)

    # Initialize from pre-trained LLM if specified
    if args.llm_init is not None and args.resume_from is None:
        logger.info(f"Initializing LLM from {args.llm_init}")
        try:
            from transformers import AutoModelForCausalLM
            llm = AutoModelForCausalLM.from_pretrained(args.llm_init)
            # Load LLM weights into model
            # This requires careful weight mapping
            _load_llm_weights(model, llm)
            del llm
        except Exception as e:
            logger.warning(f"Failed to load LLM weights: {e}")

    # Resume from checkpoint
    if args.resume_from is not None:
        checkpoint_path = os.path.join(args.resume_from, "model.pt")
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)

    return model


def _load_llm_weights(navil_model: NaViLModel, llm_model):
    """
    Load weights from a pre-trained LLM into NaViL's LLM components.

    This maps InternLM2/Qwen3 weights to NaViL's LLM layers.
    The visual encoder and MoE visual experts are randomly initialized.
    """
    llm_state = llm_model.state_dict()
    navil_state = navil_model.state_dict()

    # Map LLM weights to NaViL
    # This is model-specific; here we handle InternLM2 format
    weight_map = {}

    for key in llm_state:
        # Map embedding
        if "embed_tokens" in key:
            navil_key = key
            weight_map[navil_key] = llm_state[key]
        # Map LLM layers
        elif "layers." in key:
            navil_key = key
            weight_map[navil_key] = llm_state[key]
        # Map final norm
        elif "norm" in key and "layers" not in key:
            navil_key = key
            weight_map[navil_key] = llm_state[key]
        # Map LM head
        elif "lm_head" in key:
            navil_key = key
            weight_map[navil_key] = llm_state[key]

    # Load mapped weights
    missing, unexpected = navil_model.load_state_dict(weight_map, strict=False)
    logger.info(f"Loaded LLM weights: {len(weight_map)} keys")
    logger.info(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")


def build_dataset(args, tokenizer, stage: str):
    """Build dataset for the given stage."""
    image_processor = ImageProcessor()

    if stage in ["pretrain_1a", "pretrain_1b"]:
        dataset = PretrainDataset(
            data_path=args.data_path,
            tokenizer=tokenizer,
            image_processor=image_processor,
            max_seq_len=args.max_seq_len,
            use_synthetic_captions=(stage == "pretrain_1a"),
        )
    else:  # sft
        dataset = SFTDataset(
            data_path=args.data_path,
            tokenizer=tokenizer,
            image_processor=image_processor,
            max_seq_len=args.max_seq_len,
        )

    return dataset


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()

    os.makedirs(args.output_dir, exist_ok=True)

    # Build model
    model = build_model(args)

    # Move to GPU
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Mixed precision
    if args.bf16:
        model = model.to(torch.bfloat16)
    elif args.fp16:
        model = model.to(torch.float16)

    # Wrap with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Build tokenizer (using InternLM2 tokenizer)
    try:
        from transformers import AutoTokenizer
        tokenizer_name = args.llm_init or "internlm/internlm2-1_8b"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    except Exception as e:
        logger.warning(f"Could not load tokenizer: {e}. Using dummy tokenizer.")
        tokenizer = None

    if tokenizer is None:
        logger.error("Tokenizer required for training. Exiting.")
        return

    # Build dataset
    dataset = build_dataset(args, tokenizer, args.stage)

    # Build dataloader
    sampler = DistributedSampler(dataset) if world_size > 1 else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Build training config
    train_config = TrainingConfig(
        stage=args.stage,
        data_path=args.data_path,
        output_dir=args.output_dir,
        max_seq_len=args.max_seq_len,
        learning_rate=args.learning_rate,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        num_epochs=args.num_epochs,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        local_rank=local_rank,
        world_size=world_size,
        use_multiscale=args.use_multiscale,
    )

    # Build trainer
    raw_model = model.module if hasattr(model, "module") else model
    trainer = NaViLTrainer(
        model=raw_model,
        config=train_config,
        train_dataloader=dataloader,
    )

    # Train
    if rank == 0:
        logger.info(f"Starting training: stage={args.stage}")
        logger.info(f"Model parameters: {raw_model.num_parameters:,}")

    trainer.train()

    if rank == 0:
        logger.info("Training complete!")


if __name__ == "__main__":
    main()

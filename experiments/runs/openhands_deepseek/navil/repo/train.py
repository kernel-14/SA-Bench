"""Three-stage training pipeline for NaViL.

Stages:
1.1: Multi-modal Generative Pre-training (500M data)
    - Freeze text params, train vision-specific params
1.2: High-quality Alignment (185M data)
    - Unfreeze text attention params
2:   Supervised Fine-tuning (68M data)
    - Unfreeze all params

Default recipe configured for NaViL-2B.
"""

import math
import os
import time
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW

from config import (
    NaViLConfig,
    TrainingConfig,
    StageConfig,
    NAVIL_2B_CONFIG,
    NAVIL_2B_TRAINING,
    NAVIL_9B_CONFIG,
    NAVIL_9B_TRAINING,
)
from model import NaViL
from data import (
    ImageCaptionDataset,
    HighQualityDataset,
    SFTDataset,
    collate_multimodal_batch,
    create_image_transform,
)
from modules import MoEDecoderLayer, MoEDecoder, Connector, VisualEncoder


def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank

    return 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_param_groups_for_stage(model: NaViL, stage: StageConfig) -> List[Dict]:
    """Get parameter groups for a training stage based on freeze config.

    Stage 1.1: freeze text params (everything in llm except MoE visual paths)
    Stage 1.2: freeze FFN text params, unfreeze attention text params
    Stage 2: unfreeze all
    """
    params_with_lr = []
    params_no_wd = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "norm" in name or "bias" in name or "embedding" in name:
            params_no_wd.add(name)

    wd_params = []
    no_wd_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name in params_no_wd:
            no_wd_params.append(param)
        else:
            wd_params.append(param)

    return [
        {"params": wd_params},
        {"params": no_wd_params, "weight_decay": 0.0},
    ]


def freeze_stage_params(model: NaViL, stage: StageConfig):
    """Freeze/unfreeze parameters according to stage config.

    Stage 1.1: freeze all textual parameters in LLM
        - Visual encoder: trainable
        - Connector: trainable
        - LLM token_embedding: frozen (inherited from pre-trained LLM)
        - LLM layers:
            - MHA-MMoE: visual Q/K/V/O trainable, text Q/K/V/O frozen
            - FFN-MMoE: visual gate/up/down trainable, text gate/up/down frozen
            - RMSNorm: trainable (small)
        - LLM lm_head: frozen

    Stage 1.2: unfreeze text attention params
        - LLM layers:
            - MHA-MMoE: visual Q/K/V/O trainable, text Q/K/V/O trainable
            - FFN-MMoE: visual gate/up/down trainable, text gate/up/down frozen
            - RMSNorm: trainable
        - LLM token_embedding: trainable
        - LLM lm_head: trainable

    Stage 2: all params trainable
    """
    for name, param in model.named_parameters():
        param.requires_grad = True

    if stage.freeze_text_params:
        for name, param in model.named_parameters():
            if "visual_encoder" in name or "connector" in name:
                param.requires_grad = True
            elif "llm" in name:
                if "token_embedding" in name or "lm_head" in name:
                    param.requires_grad = False
                elif "_vis" in name or "vis" in name:
                    param.requires_grad = True
                elif "_txt" in name or "txt" in name:
                    param.requires_grad = False
                elif "norm" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
    elif stage.freeze_ffn_text:
        for name, param in model.named_parameters():
            if "llm" in name:
                if "ffn" in name.lower():
                    if "_txt" in name or "txt" in name:
                        param.requires_grad = False
                elif "attn" in name.lower():
                    if "_txt" in name or "txt" in name:
                        param.requires_grad = True


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    stage: StageConfig,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create learning rate scheduler."""
    warmup = stage.warmup_steps

    if stage.lr_schedule == "constant_with_warmup":

        def lr_lambda(step):
            if step < warmup:
                return float(step) / float(max(1, warmup))
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif stage.lr_schedule == "cosine_decay":

        def lr_lambda(step):
            if step < warmup:
                return float(step) / float(max(1, warmup))
            progress = float(step - warmup) / float(max(1, total_steps - warmup))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    else:
        raise ValueError(f"Unknown lr schedule: {stage.lr_schedule}")


def train_stage(
    model: NaViL,
    dataloader: DataLoader,
    stage_config: StageConfig,
    optimizer_config: dict,
    stage_name: str,
    rank: int,
    world_size: int,
    pad_token_id: int,
    special_token_ids: Dict[str, int],
    output_dir: str,
    max_image_patches: int,
    max_seq_len: int,
    gradient_accumulation_steps: int = 1,
    log_interval: int = 10,
    save_interval: int = 1000,
):
    """Run one training stage.

    Args:
        model: NaViL model
        dataloader: training data loader
        stage_config: hyperparameters for this stage
        optimizer_config: optimizer settings (beta1, beta2, eps)
        stage_name: name for logging
        rank: distributed rank
        world_size: distributed world size
        pad_token_id: tokenizer pad token ID
        special_token_ids: dict of special token IDs
        output_dir: checkpoint save directory
        max_image_patches: max image patches
        max_seq_len: max sequence length
        gradient_accumulation_steps: grad accumulation steps
        log_interval: logging interval in steps
        save_interval: checkpoint save interval in steps
    """
    device = model.visual_encoder.patch_embed.proj.weight.device

    freeze_stage_params(model, stage_config)
    param_groups = get_param_groups_for_stage(model, stage_config)

    optimizer = AdamW(
        param_groups,
        lr=stage_config.peak_lr,
        weight_decay=stage_config.weight_decay,
        betas=(optimizer_config["beta1"], optimizer_config["beta2"]),
        eps=optimizer_config["eps"],
    )

    total_steps = stage_config.steps
    scheduler = get_lr_scheduler(optimizer, stage_config, total_steps)

    use_multiscale = stage_config.visual_multiscale_packing

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if "cuda" in str(device)
        else nullcontext()
    )

    if world_size > 1:
        model_for_train = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )
    else:
        model_for_train = model

    model_for_train.train()
    global_step = 0
    accumulated_loss = 0.0
    start_time = time.time()

    data_iter = iter(dataloader)

    while global_step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with autocast_ctx:
            outputs = model(
                input_ids=batch["input_ids"],
                images=batch["images"],
                special_token_ids=special_token_ids,
                labels=batch["labels"],
                use_multiscale=use_multiscale,
            )

        loss = outputs["loss"] / gradient_accumulation_steps
        loss.backward()

        accumulated_loss += loss.item()

        if (global_step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        global_step += 1

        if rank == 0 and global_step % log_interval == 0:
            elapsed = time.time() - start_time
            steps_per_sec = global_step / max(elapsed, 1e-8)
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"[{stage_name}] Step {global_step}/{total_steps} | "
                f"Loss: {accumulated_loss / log_interval:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Steps/s: {steps_per_sec:.2f}"
            )
            accumulated_loss = 0.0

        if rank == 0 and global_step % save_interval == 0:
            checkpoint_path = os.path.join(output_dir, f"{stage_name}_step{global_step}.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "global_step": global_step,
                    "stage": stage_name,
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved: {checkpoint_path}")

    if rank == 0:
        final_path = os.path.join(output_dir, f"{stage_name}_final.pt")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "global_step": global_step,
                "stage": stage_name,
            },
            final_path,
        )

    return model


def create_dataloader_for_stage(
    stage_name: str,
    data_paths: Dict[str, List[str]],
    tokenizer,
    image_transform,
    batch_size: int,
    stage_config: StageConfig,
    rank: int,
    world_size: int,
    num_workers: int = 4,
    pad_token_id: int = 0,
) -> DataLoader:
    """Create appropriate dataset and dataloader for each stage."""

    if stage_name == "stage1_1":
        dataset = ImageCaptionDataset(
            data_paths=data_paths.get("web_scale", []) + data_paths.get("synthesized", []),
            tokenizer=tokenizer,
            image_transform=image_transform,
            max_length=stage_config.llm_max_seq_len,
            max_image_patches=stage_config.max_image_patches,
            use_multiscale=stage_config.visual_multiscale_packing,
        )
    elif stage_name == "stage1_2":
        dataset = HighQualityDataset(
            multimodal_paths=data_paths.get("multimodal", []),
            language_paths=data_paths.get("language", []),
            tokenizer=tokenizer,
            image_transform=image_transform,
            max_length=stage_config.llm_max_seq_len,
            max_image_patches=stage_config.max_image_patches,
            use_multiscale=stage_config.visual_multiscale_packing,
        )
    elif stage_name == "stage2":
        dataset = SFTDataset(
            data_paths=data_paths.get("sft", []),
            tokenizer=tokenizer,
            image_transform=image_transform,
            max_length=stage_config.llm_max_seq_len,
            max_image_patches=stage_config.max_image_patches,
            use_multiscale=stage_config.visual_multiscale_packing,
        )
    else:
        raise ValueError(f"Unknown stage: {stage_name}")

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        collate_fn=lambda batch: collate_multimodal_batch(
            batch, pad_token_id,
            max_seq_len=stage_config.llm_max_seq_len,
            max_image_patches=stage_config.max_image_patches,
        ),
        pin_memory=True,
        drop_last=True,
    )


def train_navil(
    model_config: NaViLConfig = NAVIL_2B_CONFIG,
    training_config: TrainingConfig = NAVIL_2B_TRAINING,
    data_paths: Optional[Dict[str, List[str]]] = None,
    tokenizer=None,
    output_dir: str = "./checkpoints",
    resume_from: Optional[str] = None,
    num_workers: int = 4,
    seed: int = 42,
):
    """Full NaViL training pipeline (3 stages).

    Args:
        model_config: NaViL architecture configuration
        training_config: training hyperparameters
        data_paths: dict with keys:
            - web_scale: paths to Laion, Coyo, Wukong, SA-1B data
            - synthesized: paths to InternVL-8B synthesized captions
            - multimodal: paths to high-quality multimodal alignment data
            - language: paths to pure language data
            - sft: paths to SFT data
        tokenizer: huggingface tokenizer
        output_dir: checkpoint directory
        resume_from: checkpoint path to resume from
        num_workers: dataloader workers
        seed: random seed
    """
    torch.manual_seed(seed)

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    model = NaViL(model_config).to(device)

    if resume_from:
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Resumed from {resume_from}")

    special_token_ids = {}
    if tokenizer is not None:
        for name, token in model_config.special_tokens.items():
            encoded = tokenizer.encode(token)
            if isinstance(encoded, list):
                special_token_ids[name] = encoded[0]
            else:
                special_token_ids[name] = encoded
        pad_token_id = getattr(tokenizer, "pad_token_id", 0) or 0
    else:
        special_token_ids = {
            "begin_of_image": 0,
            "end_of_image": 1,
            "end_of_line": 2,
            "end_of_scale": 3,
        }
        pad_token_id = 0

    image_transform = create_image_transform()

    if data_paths is None:
        data_paths = {
            "web_scale": [],
            "synthesized": [],
            "multimodal": [],
            "language": [],
            "sft": [],
        }

    stages = [
        ("stage1_1", training_config.stage1_1),
        ("stage1_2", training_config.stage1_2),
        ("stage2", training_config.stage2),
    ]

    optimizer_config = {
        "beta1": training_config.beta1,
        "beta2": training_config.beta2,
        "eps": training_config.eps,
    }

    for stage_name, stage_config in stages:
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Starting {stage_name}")
            print(f"  Max image patches: {stage_config.max_image_patches}")
            print(f"  Steps: {stage_config.steps}")
            print(f"  Batch size: {stage_config.global_batch_size}")
            print(f"  Peak LR: {stage_config.peak_lr}")
            print(f"  Schedule: {stage_config.lr_schedule}")
            print(f"  Visual multiscale: {stage_config.visual_multiscale_packing}")
            print(f"{'='*60}\n")

        per_device_batch = stage_config.global_batch_size // world_size
        grad_accum = stage_config.gradient_accumulation
        effective_batch = per_device_batch // grad_accum

        dataloader = create_dataloader_for_stage(
            stage_name=stage_name,
            data_paths=data_paths,
            tokenizer=tokenizer,
            image_transform=image_transform,
            batch_size=effective_batch,
            stage_config=stage_config,
            rank=rank,
            world_size=world_size,
            num_workers=num_workers,
            pad_token_id=pad_token_id,
        )

        model = train_stage(
            model=model,
            dataloader=dataloader,
            stage_config=stage_config,
            optimizer_config=optimizer_config,
            stage_name=stage_name,
            rank=rank,
            world_size=world_size,
            pad_token_id=pad_token_id,
            special_token_ids=special_token_ids,
            output_dir=output_dir,
            max_image_patches=stage_config.max_image_patches,
            max_seq_len=stage_config.llm_max_seq_len,
            gradient_accumulation_steps=grad_accum,
        )

    if rank == 0:
        print("\nTraining complete!")

    cleanup_distributed()


def main():
    """Entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train NaViL")
    parser.add_argument("--model_size", type=str, default="2B", choices=["2B", "9B"])
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="./data")
    args = parser.parse_args()

    if args.model_size == "2B":
        model_config = NAVIL_2B_CONFIG
        training_config = NAVIL_2B_TRAINING
    else:
        model_config = NAVIL_9B_CONFIG
        training_config = NAVIL_9B_TRAINING

    data_paths = {
        "web_scale": [
            os.path.join(args.data_dir, "laion.jsonl"),
            os.path.join(args.data_dir, "coyo.jsonl"),
            os.path.join(args.data_dir, "wukong.jsonl"),
            os.path.join(args.data_dir, "sa1b.jsonl"),
        ],
        "synthesized": [
            os.path.join(args.data_dir, "synthesized_captions.jsonl"),
        ],
        "multimodal": [
            os.path.join(args.data_dir, "high_quality_multimodal.jsonl"),
        ],
        "language": [
            os.path.join(args.data_dir, "language_data.jsonl"),
        ],
        "sft": [
            os.path.join(args.data_dir, "sft_data.jsonl"),
        ],
    }

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2-1.8b", trust_remote_code=True)
    except Exception:
        tokenizer = None
        print("Warning: Tokenizer not loaded. Using dummy tokenizer.")

    train_navil(
        model_config=model_config,
        training_config=training_config,
        data_paths=data_paths,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
        num_workers=args.num_workers,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

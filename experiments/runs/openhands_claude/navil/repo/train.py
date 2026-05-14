"""
Training script for NaViL.

Implements the two-stage training recipe from Sec. 4.2:

Stage 1: Multi-modal Generative Pre-training
  Stage 1.1 (500M pairs):
    - Freeze LLM text params (embed, norm, lm_head, linguistic MoE experts)
    - Train: visual encoder, connector, MoE visual experts
    - LR: 5e-5 constant with warmup, 70k steps, batch=7000

  Stage 1.2 (185M high-quality):
    - Unfreeze LLM attention text params (linguistic attention experts)
    - LR: 2e-5 cosine, 40k steps, batch=4614

Stage 2: Supervised Fine-tuning (68M high-quality)
  - All params unfrozen
  - LR: 2e-5 cosine, 30k steps, batch=2340

Optimizer: AdamW (β1=0.9, β2=0.95, eps=1e-8)
Precision: bfloat16
"""

import argparse
import logging
import math
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from config import (
    TrainingConfig,
    TrainingStageConfig,
    get_navil_2b_config,
    get_navil_9b_config,
    get_navil_9b_training_config,
)
from data import (
    build_pretrain_dataloader,
    build_sft_dataloader,
    collate_fn,
    setup_tokenizer,
)
from model import NaViL, build_navil_2b, build_navil_9b

logger = logging.getLogger(__name__)


# ── Learning rate schedulers ──────────────────────────────────────────────────

def get_constant_with_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0
    return LambdaLR(optimizer, lr_lambda)


def get_cosine_with_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    return LambdaLR(optimizer, lr_lambda)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    stage_cfg: TrainingStageConfig,
) -> LambdaLR:
    if stage_cfg.lr_schedule == "constant_with_warmup":
        return get_constant_with_warmup_scheduler(optimizer, stage_cfg.warmup_steps)
    elif stage_cfg.lr_schedule == "cosine":
        return get_cosine_with_warmup_scheduler(
            optimizer, stage_cfg.warmup_steps, stage_cfg.max_steps
        )
    else:
        raise ValueError(f"Unknown lr_schedule: {stage_cfg.lr_schedule}")


# ── Parameter freezing ────────────────────────────────────────────────────────

def apply_stage_freezing(model: NaViL, stage_cfg: TrainingStageConfig):
    """
    Freeze/unfreeze parameter groups according to the training stage.

    Stage 1.1: Only visual encoder, connector, and MoE visual experts are trainable.
    Stage 1.2: Additionally unfreeze LLM attention text params (linguistic MoE attn).
    Stage 2:   All params trainable.
    """
    # Start by freezing everything
    for p in model.parameters():
        p.requires_grad_(False)

    # Visual encoder: always trainable in stages 1 and 2
    if not stage_cfg.freeze_visual_encoder:
        for p in model.get_visual_params():
            p.requires_grad_(True)

    # Connector: always trainable
    if not stage_cfg.freeze_connector:
        for p in model.get_connector_params():
            p.requires_grad_(True)

    # MoE visual experts
    if not stage_cfg.freeze_moe_visual:
        for p in model.get_moe_visual_params():
            p.requires_grad_(True)

    # MoE linguistic experts (attention projections)
    if not stage_cfg.freeze_moe_linguistic:
        for p in model.get_moe_linguistic_params():
            p.requires_grad_(True)

    # LLM text params (embed, norm, lm_head, layer norms)
    if not stage_cfg.freeze_llm_text:
        for p in model.get_llm_text_non_attn_params():
            p.requires_grad_(True)

    # LLM attention text params (linguistic expert projections)
    if not stage_cfg.freeze_llm_attn:
        for p in model.get_llm_attn_text_params():
            p.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Stage {stage_cfg.name}: {trainable:,} / {total:,} params trainable "
        f"({100 * trainable / total:.1f}%)"
    )


# ── Optimizer builder ─────────────────────────────────────────────────────────

def build_optimizer(
    model: NaViL,
    stage_cfg: TrainingStageConfig,
    train_cfg: TrainingConfig,
) -> AdamW:
    """Build AdamW optimizer with weight decay applied only to non-bias/norm params."""
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name or "embed" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params,    "weight_decay": stage_cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return AdamW(
        param_groups,
        lr=stage_cfg.peak_lr,
        betas=(train_cfg.adam_beta1, train_cfg.adam_beta2),
        eps=train_cfg.adam_eps,
    )


# ── Training step ─────────────────────────────────────────────────────────────

def training_step(
    model: NaViL,
    batch: Dict,
    device: torch.device,
    use_multiscale: bool = True,
) -> torch.Tensor:
    """Execute a single forward + loss computation."""
    input_ids      = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels         = batch["labels"].to(device)
    images_batch   = batch.get("images", [])

    # Flatten images: list of lists -> list per sample
    # For batched training, process each sample individually and pad
    # (simplified: process batch[0] only for single-GPU; distributed handles batching)
    all_images = []
    for sample_images in images_batch:
        all_images.append([img.to(device) if isinstance(img, torch.Tensor) else img
                           for img in sample_images])

    output = model(
        input_ids=input_ids,
        images=all_images[0] if all_images else None,
        attention_mask=attention_mask,
        labels=labels,
        use_multiscale=use_multiscale,
    )
    return output["loss"]


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_validation_loss(
    model: NaViL,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int = 100,
) -> float:
    model.eval()
    total_loss = 0.0
    count = 0

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        if not batch:
            continue
        loss = training_step(model, batch, device, use_multiscale=False)
        total_loss += loss.item()
        count += 1

    model.train()
    return total_loss / max(count, 1)


# ── Main training loop ────────────────────────────────────────────────────────

def train_stage(
    model: NaViL,
    stage_cfg: TrainingStageConfig,
    train_cfg: TrainingConfig,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    start_step: int = 0,
    checkpoint_dir: Optional[str] = None,
) -> int:
    """
    Train for one stage. Returns the final step count.
    """
    apply_stage_freezing(model, stage_cfg)
    optimizer = build_optimizer(model, stage_cfg, train_cfg)
    scheduler = build_scheduler(optimizer, stage_cfg)

    # Resume optimizer state if checkpoint provided
    if checkpoint_dir and os.path.exists(
        os.path.join(checkpoint_dir, "optimizer.pt")
    ):
        optimizer.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "optimizer.pt"), map_location=device)
        )

    scaler = torch.cuda.amp.GradScaler(enabled=(train_cfg.mixed_precision == "fp16"))
    use_bf16 = train_cfg.mixed_precision == "bf16"

    model.train()
    step = start_step
    data_iter = iter(train_loader)
    accum_loss = 0.0
    accum_steps = 0

    logger.info(f"Starting stage {stage_cfg.name} from step {step}")

    while step < stage_cfg.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        if not batch:
            continue

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16 if use_bf16 else torch.float16,
            enabled=(train_cfg.mixed_precision != "fp32"),
        ):
            loss = training_step(
                model, batch, device,
                use_multiscale=stage_cfg.multiscale_packing,
            )
            loss = loss / train_cfg.gradient_accumulation_steps

        if use_bf16:
            loss.backward()
        else:
            scaler.scale(loss).backward()

        accum_loss += loss.item()
        accum_steps += 1

        if accum_steps % train_cfg.gradient_accumulation_steps == 0:
            if use_bf16:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train_cfg.max_grad_norm
                )
                optimizer.step()
            else:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train_cfg.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()

            scheduler.step()
            optimizer.zero_grad()
            step += 1

            if step % train_cfg.logging_steps == 0:
                lr = scheduler.get_last_lr()[0]
                avg_loss = accum_loss * train_cfg.gradient_accumulation_steps / accum_steps
                logger.info(
                    f"[{stage_cfg.name}] step={step}/{stage_cfg.max_steps} "
                    f"loss={avg_loss:.4f} lr={lr:.2e}"
                )
                accum_loss = 0.0
                accum_steps = 0

            if val_loader is not None and step % train_cfg.eval_steps == 0:
                val_loss = evaluate_validation_loss(model, val_loader, device)
                logger.info(f"[{stage_cfg.name}] step={step} val_loss={val_loss:.4f}")

            if step % train_cfg.save_steps == 0 and train_cfg.output_dir:
                save_checkpoint(model, optimizer, step, stage_cfg.name, train_cfg.output_dir)

    return step


# ── Checkpoint utilities ───────────────────────────────────────────────────────

def save_checkpoint(
    model: NaViL,
    optimizer: torch.optim.Optimizer,
    step: int,
    stage_name: str,
    output_dir: str,
):
    ckpt_dir = os.path.join(output_dir, f"{stage_name}_step{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
    torch.save({"step": step, "stage": stage_name},
               os.path.join(ckpt_dir, "meta.pt"))
    logger.info(f"Saved checkpoint to {ckpt_dir}")


def load_checkpoint(
    model: NaViL,
    checkpoint_path: str,
    device: torch.device,
    strict: bool = True,
):
    state = torch.load(os.path.join(checkpoint_path, "model.pt"), map_location=device)
    model.load_state_dict(state, strict=strict)
    meta = torch.load(os.path.join(checkpoint_path, "meta.pt"), map_location="cpu")
    return meta.get("step", 0), meta.get("stage", "")


# ── Full training pipeline ────────────────────────────────────────────────────

def train(args: argparse.Namespace):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build config
    if args.config == "navil_2b":
        model_cfg = get_navil_2b_config()
        train_cfg = TrainingConfig(model_config_name="navil_2b")
    elif args.config == "navil_9b":
        model_cfg = get_navil_9b_config()
        train_cfg = get_navil_9b_training_config()
    else:
        raise ValueError(f"Unknown config: {args.config}")

    if args.output_dir:
        train_cfg.output_dir = args.output_dir

    # Setup tokenizer
    tokenizer, special_token_ids = setup_tokenizer(model_cfg.llm_pretrained)

    # Update vocab size if new tokens were added
    model_cfg.llm.vocab_size = len(tokenizer)

    # Build model
    if args.config == "navil_2b":
        model = build_navil_2b(special_token_ids)
    else:
        model = build_navil_9b(special_token_ids)

    # Resize token embeddings for new special tokens
    model.llm.embed_tokens = nn.Embedding(len(tokenizer), model_cfg.llm.width)
    model.llm.lm_head = nn.Linear(model_cfg.llm.width, len(tokenizer), bias=False)

    # Load pre-trained LLM weights (Observation 1: LLM initialization)
    if args.llm_pretrained and not args.from_scratch:
        logger.info(f"Loading pre-trained LLM from {args.llm_pretrained}")
        from transformers import AutoModelForCausalLM
        pretrained = AutoModelForCausalLM.from_pretrained(
            args.llm_pretrained, trust_remote_code=True
        )
        missing, unexpected = model.load_llm_pretrained(pretrained.state_dict())
        logger.info(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        del pretrained

    model = model.to(device)

    # Determine which stage to start from
    start_stage = args.stage - 1   # 0-indexed
    start_step = 0

    if args.resume_from_checkpoint:
        start_step, stage_name = load_checkpoint(
            model, args.resume_from_checkpoint, device, strict=False
        )
        logger.info(f"Resumed from {args.resume_from_checkpoint} at step {start_step}")

    # Run training stages
    for stage_idx in range(start_stage, len(train_cfg.stages)):
        stage_cfg = train_cfg.stages[stage_idx]
        logger.info(f"=== Starting {stage_cfg.name} ===")

        # Build data loaders for this stage
        if stage_cfg.name == "stage1_1":
            train_loader = build_pretrain_dataloader(
                shard_urls=args.pretrain_data.split(","),
                tokenizer=tokenizer,
                batch_size=max(1, stage_cfg.global_batch_size // max(1, args.world_size)),
                num_workers=train_cfg.num_workers,
                max_length=stage_cfg.max_seq_len,
                max_patches=stage_cfg.max_image_patches,
            )
        elif stage_cfg.name == "stage1_2":
            train_loader = build_sft_dataloader(
                data_path=args.stage12_data,
                image_root=args.image_root,
                tokenizer=tokenizer,
                batch_size=max(1, stage_cfg.global_batch_size // max(1, args.world_size)),
                num_workers=train_cfg.num_workers,
                max_length=stage_cfg.max_seq_len,
                max_patches=stage_cfg.max_image_patches,
            )
        else:  # stage2_sft
            train_loader = build_sft_dataloader(
                data_path=args.sft_data,
                image_root=args.image_root,
                tokenizer=tokenizer,
                batch_size=max(1, stage_cfg.global_batch_size // max(1, args.world_size)),
                num_workers=train_cfg.num_workers,
                max_length=stage_cfg.max_seq_len,
                max_patches=stage_cfg.max_image_patches,
            )

        val_loader = None
        if args.val_data:
            val_loader = build_sft_dataloader(
                data_path=args.val_data,
                image_root=args.image_root,
                tokenizer=tokenizer,
                batch_size=4,
                num_workers=2,
                max_length=2048,
                max_patches=1024,
                shuffle=False,
            )

        start_step = train_stage(
            model=model,
            stage_cfg=stage_cfg,
            train_cfg=train_cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            start_step=start_step if stage_idx == start_stage else 0,
            checkpoint_dir=args.resume_from_checkpoint if stage_idx == start_stage else None,
        )

        # Save end-of-stage checkpoint
        save_checkpoint(
            model,
            torch.optim.AdamW(model.parameters()),  # dummy optimizer for final save
            start_step,
            stage_cfg.name + "_final",
            train_cfg.output_dir,
        )

    logger.info("Training complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NaViL")
    parser.add_argument("--config", type=str, default="navil_2b",
                        choices=["navil_2b", "navil_9b"])
    parser.add_argument("--stage", type=int, default=1,
                        help="Start from stage (1=stage1_1, 2=stage1_2, 3=stage2_sft)")
    parser.add_argument("--pretrain_data", type=str, default="",
                        help="Comma-separated WebDataset shard URLs for stage 1.1")
    parser.add_argument("--stage12_data", type=str, default="",
                        help="JSONL path for stage 1.2 high-quality data")
    parser.add_argument("--sft_data", type=str, default="",
                        help="JSONL path for stage 2 SFT data")
    parser.add_argument("--val_data", type=str, default="",
                        help="JSONL path for validation data")
    parser.add_argument("--image_root", type=str, default="",
                        help="Root directory for images")
    parser.add_argument("--llm_pretrained", type=str, default="",
                        help="Path to pre-trained LLM checkpoint")
    parser.add_argument("--from_scratch", action="store_true",
                        help="Train from scratch (no LLM initialization)")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume_from_checkpoint", type=str, default="")
    parser.add_argument("--world_size", type=int, default=1,
                        help="Number of GPUs (for batch size scaling)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

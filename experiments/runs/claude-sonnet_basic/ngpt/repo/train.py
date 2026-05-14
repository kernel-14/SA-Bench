"""
Training script for nGPT and GPT on OpenWebText.

Paper setup (Appendix A.6):
  - Dataset:      OpenWebText
  - Tokenizer:    LLaMA-2 (32k vocab)
  - Hardware:     64 × A100 across 8 nodes
  - Global batch: 512 sequences
  - Optimizer:    Adam (nGPT, wd=0) / AdamW (GPT, wd=0.1)
  - LR schedule:  Cosine annealing to 0
  - Warmup:       0 steps (nGPT) / 2000 steps (GPT)
  - Precision:    bfloat16
  - Context:      1k / 4k / 8k tokens

Critical nGPT training detail (Section 2.6, step 2):
  After every optimizer.step(), call model.normalize_weights() to project
  all weight matrices back onto the unit hypersphere.
"""

import os
import math
import time
import json
import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import nGPT, GPT, nGPTConfig


# ── Dataset ───────────────────────────────────────────────────────────────────

class MemMapDataset(torch.utils.data.Dataset):
    """Memory-mapped token dataset (uint16 .bin files)."""

    def __init__(self, path: str, seq_len: int):
        self.data    = np.memmap(path, dtype=np.uint16, mode='r')
        self.seq_len = seq_len

    def __len__(self):
        # Each sample needs seq_len+1 tokens (input + target)
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = torch.from_numpy(
            self.data[start : start + self.seq_len + 1].astype(np.int64)
        )
        return chunk[:-1], chunk[1:]


# ── LR schedule ───────────────────────────────────────────────────────────────

def cosine_lr(step: int, max_steps: int, lr_max: float,
              lr_min: float = 0.0, warmup: int = 0) -> float:
    if step < warmup:
        return lr_max * step / max(warmup, 1)
    if step >= max_steps:
        return lr_min
    t = (step - warmup) / (max_steps - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * t))


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, ctx, max_batches: int = 100) -> float:
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        if n >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with ctx:
            _, loss = model(x, y)
        total += loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    # ── Distributed setup ─────────────────────────────────────────────────────
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        dist.init_process_group('nccl')
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ['LOCAL_RANK'])
        device     = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
        master     = rank == 0
    else:
        rank = 0; world_size = 1; master = True
        device = torch.device(args.device)

    # ── dtype / autocast ──────────────────────────────────────────────────────
    ptdtype = {'float32': torch.float32,
               'bfloat16': torch.bfloat16,
               'float16': torch.float16}[args.dtype]
    dev_type = device.type
    ctx = (torch.autocast(device_type=dev_type, dtype=ptdtype)
           if args.dtype != 'float32' else nullcontext())

    # ── Output dir ────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    if master:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / 'args.json', 'w') as f:
            json.dump(vars(args), f, indent=2)

    # ── Model ─────────────────────────────────────────────────────────────────
    config = nGPTConfig(
        vocab_size  = args.vocab_size,
        n_layers    = args.n_layers,
        d_model     = args.d_model,
        n_heads     = args.n_heads,
        d_mlp       = args.d_mlp,
        max_seq_len = args.seq_len,
        dropout     = args.dropout,
        # nGPT-specific
        alpha_init  = args.alpha_init,
        sqk_init    = args.sqk_init,
        su_init     = args.su_init,
        sv_init     = args.sv_init,
        sz_init     = args.sz_init,
        use_qk_norm = args.use_qk_norm,
    )

    ModelClass = nGPT if args.model_type == 'ngpt' else GPT
    model = ModelClass(config).to(device)

    if master:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[{args.model_type.upper()}] {n_params/1e6:.1f}M parameters")

    if ddp:
        model = DDP(model, device_ids=[local_rank])

    raw_model = model.module if ddp else model

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # nGPT: Adam, wd=0, no warmup  (Section 2.6 step 7, Table 3)
    # GPT:  AdamW, wd=0.1, 2000-step warmup
    if args.model_type == 'ngpt':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr,
            betas=(0.9, 0.95), eps=1e-8,
        )
        warmup_steps = 0
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr,
            betas=(0.9, 0.95), eps=1e-8,
            weight_decay=0.1,
        )
        warmup_steps = args.warmup_steps

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = MemMapDataset(args.train_data, args.seq_len)
    val_ds   = MemMapDataset(args.val_data,   args.seq_len)

    train_sampler = (torch.utils.data.distributed.DistributedSampler(train_ds)
                     if ddp else None)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size  = args.batch_size,
        sampler     = train_sampler,
        shuffle     = (train_sampler is None),
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = True,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    step          = 0
    best_val_loss = float('inf')
    log_data      = []
    t0            = time.time()
    train_iter    = iter(train_loader)

    if master:
        print(f"Training for {args.max_steps} steps  "
              f"(seq_len={args.seq_len}, batch={args.batch_size})")

    while step < args.max_steps:
        # ── fetch batch ───────────────────────────────────────────────────────
        try:
            x, y = next(train_iter)
        except StopIteration:
            if ddp and train_sampler is not None:
                train_sampler.set_epoch(step)
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        # ── LR ────────────────────────────────────────────────────────────────
        lr = cosine_lr(step, args.max_steps, args.lr,
                       lr_min=0.0, warmup=warmup_steps)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # ── forward / backward ────────────────────────────────────────────────
        with ctx:
            _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        # ── nGPT: project weights back onto hypersphere ───────────────────────
        # This is the key step that distinguishes nGPT training.
        # Must be done AFTER optimizer.step() and on the raw (non-DDP) model.
        if args.model_type == 'ngpt':
            raw_model.normalize_weights()

        step += 1

        # ── logging ───────────────────────────────────────────────────────────
        if master and step % args.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            tok_per_s = args.batch_size * args.seq_len * args.log_interval / dt
            print(f"step {step:7d} | loss {loss.item():.4f} | "
                  f"lr {lr:.2e} | {tok_per_s/1e6:.2f}M tok/s")
            log_data.append({'step': step, 'train_loss': loss.item(), 'lr': lr})

        # ── validation ────────────────────────────────────────────────────────
        if master and step % args.eval_interval == 0:
            val_loss = evaluate(raw_model, val_loader, device, ctx, args.eval_batches)
            print(f"step {step:7d} | val_loss {val_loss:.4f}")
            if log_data:
                log_data[-1]['val_loss'] = val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'step': step,
                    'model_state_dict': raw_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': config,
                    'model_type': args.model_type,
                }, out_dir / 'best_model.pt')

        # ── checkpoint ────────────────────────────────────────────────────────
        if master and step % args.save_interval == 0:
            torch.save({
                'step': step,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'model_type': args.model_type,
            }, out_dir / f'ckpt_{step:08d}.pt')

    # ── save log ──────────────────────────────────────────────────────────────
    if master:
        with open(out_dir / 'training_log.json', 'w') as f:
            json.dump(log_data, f, indent=2)
        print(f"Done. Best val loss: {best_val_loss:.4f}")

    if ddp:
        dist.destroy_process_group()


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_parser():
    p = argparse.ArgumentParser('Train nGPT / GPT on OpenWebText')

    # model
    p.add_argument('--model_type', default='ngpt', choices=['ngpt', 'gpt'])
    p.add_argument('--vocab_size', type=int, default=32000)
    p.add_argument('--n_layers',   type=int, default=24)
    p.add_argument('--d_model',    type=int, default=1024)
    p.add_argument('--n_heads',    type=int, default=16)
    p.add_argument('--d_mlp',      type=int, default=None)
    p.add_argument('--dropout',    type=float, default=0.0)

    # nGPT-specific
    p.add_argument('--alpha_init',   type=float, default=0.05)
    p.add_argument('--sqk_init',     type=float, default=1.0)
    p.add_argument('--su_init',      type=float, default=1.0)
    p.add_argument('--sv_init',      type=float, default=1.0)
    p.add_argument('--sz_init',      type=float, default=1.0)
    p.add_argument('--use_qk_norm',  action='store_true', default=True)
    p.add_argument('--no_qk_norm',   dest='use_qk_norm', action='store_false')

    # training
    p.add_argument('--seq_len',       type=int,   default=4096)
    p.add_argument('--batch_size',    type=int,   default=8,
                   help='Per-GPU batch size (global = batch_size * world_size)')
    p.add_argument('--max_steps',     type=int,   default=200000)
    p.add_argument('--lr',            type=float, default=1e-3)
    p.add_argument('--warmup_steps',  type=int,   default=2000,
                   help='Only used for GPT baseline')
    p.add_argument('--grad_clip',     type=float, default=1.0)

    # data
    p.add_argument('--train_data',   default='data/train.bin')
    p.add_argument('--val_data',     default='data/val.bin')
    p.add_argument('--num_workers',  type=int, default=4)

    # logging / checkpointing
    p.add_argument('--out_dir',        default='out')
    p.add_argument('--log_interval',   type=int, default=100)
    p.add_argument('--eval_interval',  type=int, default=1000)
    p.add_argument('--eval_batches',   type=int, default=100)
    p.add_argument('--save_interval',  type=int, default=10000)

    # hardware
    p.add_argument('--device', default='cuda')
    p.add_argument('--dtype',  default='bfloat16',
                   choices=['float32', 'bfloat16', 'float16'])

    return p


if __name__ == '__main__':
    args = get_parser().parse_args()
    train(args)

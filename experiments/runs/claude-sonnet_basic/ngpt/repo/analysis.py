"""
Analysis and visualisation script – reproduces Figures 4, 5, 6 from the paper.

Figure 4: Embedding norms, eigenvalue distribution, pairwise dot products
Figure 5: Condition numbers of attention / MLP matrices per layer
Figure 6: Eigen learning rates α_A, α_M; MLP scaling s_u, s_v; QK scaling s_qk; logit scaling s_z

Also plots training curves (Figures 1, 2).

Usage:
    python analysis.py \
        --ngpt_ckpt out_ngpt/best_model.pt \
        --gpt_ckpt  out_gpt/best_model.pt  \
        --log_ngpt  out_ngpt/training_log.json \
        --log_gpt   out_gpt/training_log.json  \
        --out_dir   figures/
"""

import os
import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from model import nGPT, GPT, nGPTConfig


# ── Model loading ─────────────────────────────────────────────────────────────

def load_checkpoint(path: str, device: str = 'cpu'):
    ckpt = torch.load(path, map_location=device)
    config = ckpt['config']
    if isinstance(config, dict):
        config = nGPTConfig(**config)
    mtype = ckpt.get('model_type', 'ngpt')
    model = (nGPT if mtype == 'ngpt' else GPT)(config)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, mtype, config


# ── Condition number ──────────────────────────────────────────────────────────

def condition_number(W: torch.Tensor) -> float:
    """Ratio of largest to smallest singular value."""
    try:
        s = torch.linalg.svdvals(W.float())
        s_min = s.min().item()
        if s_min < 1e-12:
            return float('inf')
        return (s.max() / s_min).item()
    except Exception:
        return float('nan')


# ── Figure 4: Embedding analysis ─────────────────────────────────────────────

def embedding_stats(model, mtype: str) -> Dict:
    """Return norms, normalised eigenvalues, pairwise dot products."""
    E_in  = model.E_input.weight.data.float()
    # Both nGPT and GPT now use separate E_output (untied embeddings)
    E_out = model.E_output.weight.data.float()

    # Norms
    in_norms  = torch.norm(E_in,  dim=-1).cpu().numpy()
    out_norms = torch.norm(E_out, dim=-1).cpu().numpy()

    # Eigenvalues of covariance (sample 1000 rows)
    n = min(1000, E_in.shape[0])
    idx = torch.randperm(E_in.shape[0])[:n]
    E   = E_in[idx]
    E   = E - E.mean(0, keepdim=True)
    cov = (E.T @ E) / n
    eigs = torch.linalg.eigvalsh(cov).cpu().numpy()
    med  = np.median(eigs)
    eigs_norm = eigs / med if med > 0 else eigs

    # Pairwise dot products (sample 500 pairs)
    m = min(500, n)
    En = F.normalize(E[:m], dim=-1)
    dots = (En @ En.T).cpu().numpy()
    mask = np.triu(np.ones_like(dots, dtype=bool), k=1)

    return dict(in_norms=in_norms, out_norms=out_norms,
                eigs_norm=eigs_norm, dot_products=dots[mask])


# ── Figure 5: Condition numbers per layer ────────────────────────────────────

def layer_condition_numbers(model, mtype: str) -> Dict:
    attn_cond, mlp_cond = [], []

    for layer in model.layers:
        attn = layer.attn if mtype == 'ngpt' else layer.attn
        mlp  = layer.mlp  if mtype == 'ngpt' else layer.mlp

        d_model = attn.W_q.weight.shape[1]
        n_heads = attn.n_heads
        d_head  = d_model // n_heads

        # Median condition number across heads
        head_conds = []
        for h in range(n_heads):
            sl = slice(h * d_head, (h + 1) * d_head)
            cq = condition_number(attn.W_q.weight[sl])
            ck = condition_number(attn.W_k.weight[sl])
            cv = condition_number(attn.W_v.weight[sl])
            head_conds.append(np.nanmedian([cq, ck, cv]))
        attn_cond.append(np.nanmedian(head_conds))

        cu = condition_number(mlp.W_u.weight)
        cv = condition_number(mlp.W_v.weight)
        co = condition_number(mlp.W_o.weight)
        mlp_cond.append(np.nanmedian([cu, cv, co]))

    return dict(attn=attn_cond, mlp=mlp_cond)


# ── Figure 6: nGPT scaling parameters ────────────────────────────────────────

def ngpt_scaling_params(model: nGPT) -> Dict:
    alpha_A, alpha_M, sqk, su, sv = [], [], [], [], []

    for layer in model.layers:
        alpha_A.append(torch.abs(layer.alpha_A * layer.alpha_ratio).mean().item())
        alpha_M.append(torch.abs(layer.alpha_M * layer.alpha_ratio).mean().item())
        sqk.append((layer.attn.sqk * layer.attn.sqk_ratio).mean().item())
        su.append((layer.mlp.su * layer.mlp.su_ratio).mean().item())
        sv.append((layer.mlp.sv * layer.mlp.sv_ratio).mean().item())

    sz = (model.sz * model.sz_ratio).detach().cpu().numpy()

    return dict(alpha_A=alpha_A, alpha_M=alpha_M,
                sqk=sqk, su=su, sv=sv, sz=sz)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _mpl():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting: pip install matplotlib")


def plot_training_curves(logs: Dict[str, str], out_path: str):
    """Figure 1 / 2: validation loss vs training steps."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8, 5))

    styles = {
        'ngpt': dict(color='steelblue', linestyle='-',  label='nGPT'),
        'gpt':  dict(color='tomato',    linestyle='--', label='GPT'),
    }

    for name, log_path in logs.items():
        if not os.path.exists(log_path):
            print(f"  [skip] {log_path} not found")
            continue
        with open(log_path) as f:
            data = json.load(f)
        steps = [d['step'] for d in data if 'val_loss' in d]
        vals  = [d['val_loss'] for d in data if 'val_loss' in d]
        key   = 'ngpt' if 'ngpt' in name.lower() else 'gpt'
        kw    = styles.get(key, {})
        ax.plot(steps, vals, **kw)

    ax.set_xlabel('Training steps')
    ax.set_ylabel('Validation loss')
    ax.set_title('Validation loss during training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_embedding_stats(ngpt_stats: Dict, gpt_stats: Dict, out_path: str):
    """Figure 4: embedding norms, eigenvalues, dot products."""
    plt = _mpl()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Left: norms
    ax = axes[0]
    for stats, label, color in [(ngpt_stats, 'nGPT', 'steelblue'),
                                  (gpt_stats,  'GPT',  'tomato')]:
        ax.hist(stats['in_norms'],  bins=50, alpha=0.5, color=color,
                label=f'{label} input',  density=True)
        ax.hist(stats['out_norms'], bins=50, alpha=0.3, color=color,
                label=f'{label} output', density=True, linestyle='--',
                histtype='step', linewidth=1.5)
    ax.set_xlabel('Embedding norm')
    ax.set_ylabel('Density')
    ax.set_title('Embedding norms')
    ax.legend(fontsize=7)

    # Middle: eigenvalues
    ax = axes[1]
    for stats, label, color in [(ngpt_stats, 'nGPT', 'steelblue'),
                                  (gpt_stats,  'GPT',  'tomato')]:
        ax.hist(np.clip(stats['eigs_norm'], 0, 10), bins=50,
                alpha=0.5, color=color, label=label, density=True)
    ax.set_xlabel('Eigenvalue / median')
    ax.set_ylabel('Density')
    ax.set_title('Covariance eigenvalues (normalised)')
    ax.legend()

    # Right: pairwise dot products
    ax = axes[2]
    for stats, label, color in [(ngpt_stats, 'nGPT', 'steelblue'),
                                  (gpt_stats,  'GPT',  'tomato')]:
        ax.hist(stats['dot_products'], bins=50, alpha=0.5,
                color=color, label=label, density=True)
    ax.set_xlabel('Pairwise dot product')
    ax.set_ylabel('Density')
    ax.set_title('Pairwise embedding dot products')
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_condition_numbers(ngpt_cond: Dict, gpt_cond: Dict, out_path: str):
    """Figure 5: condition numbers per layer."""
    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, key, title in [(axes[0], 'attn', 'Attention matrices'),
                            (axes[1], 'mlp',  'MLP matrices')]:
        ax.plot(ngpt_cond[key], 'b-o', markersize=4, label='nGPT')
        ax.plot(gpt_cond[key],  'r--s', markersize=4, label='GPT')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Median condition number')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_ngpt_params(params: Dict, out_path: str):
    """Figure 6: eigen learning rates and scaling factors."""
    plt = _mpl()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    n = len(params['alpha_A'])

    # Left: eigen learning rates
    ax = axes[0]
    ax.plot(params['alpha_A'], 'b-o', markersize=4, label='α_A (Attention)')
    ax.plot(params['alpha_M'], 'r-s', markersize=4, label='α_M (MLP)')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean |α|')
    ax.set_title('Eigen learning rates')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Middle: MLP scaling
    ax = axes[1]
    ax.plot(params['su'], 'g-o', markersize=4, label='s_u')
    ax.plot(params['sv'], 'm-s', markersize=4, label='s_v')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean scaling factor')
    ax.set_title('MLP scaling factors')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: QK scaling + s_z histogram
    ax = axes[2]
    ax.plot(params['sqk'], 'c-o', markersize=4, label='s_qk')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean s_qk')
    ax.set_title('QK scaling & logit scaling')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Inset: s_z distribution
    ax2 = ax.twinx()
    ax2.hist(params['sz'], bins=30, alpha=0.3, color='orange', label='s_z dist')
    ax2.set_ylabel('s_z count', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── Summary printout ──────────────────────────────────────────────────────────

def print_summary(model, mtype: str, config: nGPTConfig):
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\n{'='*55}")
    print(f"  {mtype.upper()}  |  {n_params:.1f}M params  |  "
          f"{config.n_layers} layers  |  d={config.d_model}")
    print(f"{'='*55}")

    emb = embedding_stats(model, mtype)
    print(f"  Embedding norms  in:  mean={emb['in_norms'].mean():.4f}  "
          f"std={emb['in_norms'].std():.4f}")
    print(f"  Embedding norms out:  mean={emb['out_norms'].mean():.4f}  "
          f"std={emb['out_norms'].std():.4f}")

    cond = layer_condition_numbers(model, mtype)
    print(f"  Attn cond (median across layers): {np.nanmedian(cond['attn']):.1f}")
    print(f"  MLP  cond (median across layers): {np.nanmedian(cond['mlp']):.1f}")

    if mtype == 'ngpt':
        p = ngpt_scaling_params(model)
        print(f"  Mean |α_A|: {np.mean(p['alpha_A']):.4f}   "
              f"Mean |α_M|: {np.mean(p['alpha_M']):.4f}")
        print(f"  Mean s_qk:  {np.mean(p['sqk']):.4f}   "
              f"Mean s_u: {np.mean(p['su']):.4f}   "
              f"Mean s_v: {np.mean(p['sv']):.4f}")
        print(f"  Mean s_z:   {p['sz'].mean():.2f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ap = argparse.ArgumentParser('Analyse nGPT / GPT checkpoints')
    ap.add_argument('--ngpt_ckpt', default=None)
    ap.add_argument('--gpt_ckpt',  default=None)
    ap.add_argument('--log_ngpt',  default=None, help='training_log.json for nGPT')
    ap.add_argument('--log_gpt',   default=None, help='training_log.json for GPT')
    ap.add_argument('--out_dir',   default='figures')
    ap.add_argument('--device',    default='cpu')
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ngpt_model = gpt_model = None
    ngpt_cfg   = gpt_cfg   = None

    if args.ngpt_ckpt:
        ngpt_model, _, ngpt_cfg = load_checkpoint(args.ngpt_ckpt, args.device)
        print_summary(ngpt_model, 'ngpt', ngpt_cfg)

    if args.gpt_ckpt:
        gpt_model, _, gpt_cfg = load_checkpoint(args.gpt_ckpt, args.device)
        print_summary(gpt_model, 'gpt', gpt_cfg)

    # Training curves
    logs = {}
    if args.log_ngpt: logs['nGPT'] = args.log_ngpt
    if args.log_gpt:  logs['GPT']  = args.log_gpt
    if logs:
        plot_training_curves(logs, str(out / 'fig1_training_curves.png'))

    # Figures 4, 5, 6 require both models
    if ngpt_model and gpt_model:
        print("\nGenerating figures 4, 5, 6 ...")
        ngpt_emb = embedding_stats(ngpt_model, 'ngpt')
        gpt_emb  = embedding_stats(gpt_model,  'gpt')
        plot_embedding_stats(ngpt_emb, gpt_emb,
                             str(out / 'fig4_embedding_stats.png'))

        ngpt_cond = layer_condition_numbers(ngpt_model, 'ngpt')
        gpt_cond  = layer_condition_numbers(gpt_model,  'gpt')
        plot_condition_numbers(ngpt_cond, gpt_cond,
                               str(out / 'fig5_condition_numbers.png'))

    if ngpt_model:
        params = ngpt_scaling_params(ngpt_model)
        plot_ngpt_params(params, str(out / 'fig6_ngpt_params.png'))

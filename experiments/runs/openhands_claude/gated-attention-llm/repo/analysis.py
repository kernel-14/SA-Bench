"""
Analysis tools for gated-attention models (Sec. 4 of the paper).

Implements the three main analyses:
  1. Gating score statistics (mean, distribution, sparsity) — Sec. 4.2 / Fig. 3
  2. Attention sink measurement (proportion of attention on first token) — Sec. 4.3 / Fig. 2
  3. Massive activation tracking (max hidden-state activation per layer) — Sec. 4.3

Usage:
    python analysis.py --checkpoint checkpoints/moe_15a2b_G1_elementwise/step_0100000.pt
                       --eval_data_dir /data/eval
                       --output_dir analysis_results
"""

import argparse
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import ModelConfig
from model import GatedTransformerLM, build_model
from modules import GatingModule
from layers import TransformerBlock
from data import EvalDataset, make_eval_dataloader


# ---------------------------------------------------------------------------
# Hook-based activation collector
# ---------------------------------------------------------------------------

class ActivationCollector:
    """Registers forward hooks to collect intermediate activations."""

    def __init__(self):
        self.hooks: list[torch.utils.hooks.RemovableHook] = []
        self.data: dict[str, list[torch.Tensor]] = {}

    def register(self, module: nn.Module, name: str):
        def hook(mod, inp, out):
            tensor = out[0] if isinstance(out, tuple) else out
            self.data.setdefault(name, []).append(tensor.detach().cpu())

        self.hooks.append(module.register_forward_hook(hook))

    def clear(self):
        self.data.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ---------------------------------------------------------------------------
# 1. Gating score analysis (Sec. 4.2, Fig. 3, Table 4)
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_gating_scores(
    model: GatedTransformerLM,
    data_loader: DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> dict[str, dict]:
    """Collect gating score statistics across all layers.

    Returns a dict mapping layer_name → {mean, std, sparsity_1e-2, sparsity_1e-3, histogram}.
    """
    model.eval()
    collector = ActivationCollector()

    # Register hooks on all GatingModule instances
    for name, module in model.named_modules():
        if isinstance(module, GatingModule) and module.gate_proj is not None:
            collector.register(module, name)

    # Collect activations
    for i, batch in enumerate(data_loader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        model(input_ids=input_ids)

    collector.remove()

    # Compute statistics per gate
    stats: dict[str, dict] = {}
    for name, tensors in collector.data.items():
        # tensors: list of (batch, seq_len, ...) gate score tensors
        all_scores = torch.cat([t.flatten() for t in tensors])
        stats[name] = {
            "mean": all_scores.mean().item(),
            "std": all_scores.std().item(),
            "sparsity_1e-2": (all_scores.abs() < 1e-2).float().mean().item(),
            "sparsity_1e-3": (all_scores.abs() < 1e-3).float().mean().item(),
            "histogram": np.histogram(all_scores.numpy(), bins=50, range=(0, 1)),
        }

    return stats


def print_gating_score_table(stats: dict[str, dict]):
    """Print Table 4 style summary of gating score statistics."""
    print(f"\n{'Layer':<50} {'Mean':>8} {'Std':>8} {'Sparse<1e-2':>12} {'Sparse<1e-3':>12}")
    print("-" * 92)
    for name, s in sorted(stats.items()):
        print(
            f"{name:<50} {s['mean']:>8.4f} {s['std']:>8.4f} "
            f"{s['sparsity_1e-2']:>12.4f} {s['sparsity_1e-3']:>12.4f}"
        )


# ---------------------------------------------------------------------------
# 2. Attention sink analysis (Sec. 4.3, Fig. 2)
# ---------------------------------------------------------------------------

class AttentionWeightCollector:
    """Collects attention weight tensors from all attention layers."""

    def __init__(self):
        self.hooks: list = []
        self.attn_weights: dict[int, list[torch.Tensor]] = {}

    def register_layer(self, layer_idx: int, attn_module: nn.Module):
        def hook(mod, inp, out):
            # out = (attn_output, new_cache, attn_weights)
            if isinstance(out, tuple) and len(out) >= 3 and out[2] is not None:
                weights = out[2].detach().cpu()  # (batch, heads, seq, seq)
                self.attn_weights.setdefault(layer_idx, []).append(weights)

        self.hooks.append(attn_module.register_forward_hook(hook))

    def clear(self):
        self.attn_weights.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


@torch.no_grad()
def measure_attention_sink(
    model: GatedTransformerLM,
    data_loader: DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> dict[str, object]:
    """Measure the proportion of attention allocated to the first token.

    Returns:
        {
          'per_layer_first_token_frac': list of floats (one per layer),
          'avg_first_token_frac': float,
          'per_layer_per_head': list of (num_heads,) arrays,
        }
    """
    model.eval()
    collector = AttentionWeightCollector()

    for i, layer in enumerate(model.layers):
        collector.register_layer(i, layer.attn)

    for i, batch in enumerate(data_loader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        model(input_ids=input_ids)

    collector.remove()

    num_layers = len(model.layers)
    per_layer_frac: list[float] = []
    per_layer_per_head: list[np.ndarray] = []

    for layer_idx in range(num_layers):
        if layer_idx not in collector.attn_weights:
            per_layer_frac.append(0.0)
            per_layer_per_head.append(np.zeros(1))
            continue

        # Stack all batches: (total_batch, heads, seq, seq)
        weights = torch.cat(collector.attn_weights[layer_idx], dim=0)
        # Attention to first token: weights[:, :, :, 0]
        first_token_attn = weights[:, :, :, 0]  # (batch, heads, seq)
        # Average over batch and sequence positions
        frac_per_head = first_token_attn.mean(dim=(0, 2)).numpy()  # (heads,)
        avg_frac = frac_per_head.mean().item()

        per_layer_frac.append(avg_frac)
        per_layer_per_head.append(frac_per_head)

    avg_frac = float(np.mean(per_layer_frac))
    return {
        "per_layer_first_token_frac": per_layer_frac,
        "avg_first_token_frac": avg_frac,
        "per_layer_per_head": per_layer_per_head,
    }


def print_attention_sink_table(results: dict[str, object]):
    """Print attention sink statistics (Table 4 'F-Attn' column)."""
    fracs = results["per_layer_first_token_frac"]
    print(f"\nAttention Sink Analysis")
    print(f"  Average first-token attention fraction: {results['avg_first_token_frac']:.4f}")
    print(f"\n  {'Layer':>6} {'First-Token Attn':>18}")
    print("  " + "-" * 26)
    for i, f in enumerate(fracs):
        print(f"  {i:>6} {f:>18.4f}")


# ---------------------------------------------------------------------------
# 3. Massive activation tracking (Sec. 4.3, Fig. 6)
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_massive_activations(
    model: GatedTransformerLM,
    data_loader: DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> dict[str, object]:
    """Track the mean of maximum hidden-state activations per layer.

    Corresponds to the 'M-Act' column in Table 4 and Fig. 6 in the paper.
    """
    model.eval()

    # We collect the residual stream after each transformer block
    layer_max_acts: dict[int, list[float]] = {i: [] for i in range(len(model.layers))}
    hooks = []

    def make_hook(layer_idx: int):
        def hook(mod, inp, out):
            # out[0] is the updated hidden state (batch, seq, d_model)
            hidden = out[0].detach().float()
            max_act = hidden.abs().max(dim=-1).values.mean().item()
            layer_max_acts[layer_idx].append(max_act)
        return hook

    for i, layer in enumerate(model.layers):
        hooks.append(layer.register_forward_hook(make_hook(i)))

    for i, batch in enumerate(data_loader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        model(input_ids=input_ids)

    for h in hooks:
        h.remove()

    per_layer_mean_max: list[float] = [
        float(np.mean(layer_max_acts[i])) for i in range(len(model.layers))
    ]
    overall_mean = float(np.mean(per_layer_mean_max))

    return {
        "per_layer_mean_max_activation": per_layer_mean_max,
        "overall_mean_max_activation": overall_mean,
    }


def print_massive_activation_table(results: dict[str, object]):
    """Print massive activation statistics (Table 4 'M-Act' column)."""
    acts = results["per_layer_mean_max_activation"]
    print(f"\nMassive Activation Analysis")
    print(f"  Overall mean max activation: {results['overall_mean_max_activation']:.4f}")
    print(f"\n  {'Layer':>6} {'Mean Max Activation':>22}")
    print("  " + "-" * 30)
    for i, a in enumerate(acts):
        print(f"  {i:>6} {a:>22.4f}")


# ---------------------------------------------------------------------------
# 4. SDPA output sparsity analysis (Appendix A.2, Fig. 5)
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_sdpa_output_sparsity(
    model: GatedTransformerLM,
    data_loader: DataLoader,
    device: torch.device,
    thresholds: list[float] = [1e-2, 1e-3],
    max_batches: int = 20,
) -> dict[str, object]:
    """Measure sparsity of SDPA outputs before and after gating (Fig. 5).

    Compares:
      - Pre-gating SDPA output sparsity
      - Post-gating SDPA output sparsity
      - Pre-gating * avg_gate_score (to isolate sparsity from magnitude scaling)
    """
    model.eval()

    pre_gate_values: list[torch.Tensor] = []
    post_gate_values: list[torch.Tensor] = []
    gate_scores: list[torch.Tensor] = []

    hooks = []

    for name, module in model.named_modules():
        if isinstance(module, GatingModule) and "gate_G1" in name:
            # Hook to capture pre/post gate values
            def make_hooks(gate_mod):
                pre_buf: list[torch.Tensor] = []
                post_buf: list[torch.Tensor] = []
                score_buf: list[torch.Tensor] = []

                def pre_hook(mod, inp):
                    # inp[0] is Y (SDPA output), inp[1] is X
                    pre_buf.append(inp[0].detach().cpu().float())

                def post_hook(mod, inp, out):
                    post_buf.append(out.detach().cpu().float())
                    if gate_mod.gate_proj is not None:
                        with torch.no_grad():
                            logits = gate_mod.gate_proj(inp[1].detach())
                            scores = torch.sigmoid(logits)
                            score_buf.append(scores.cpu().float())

                hooks.append(gate_mod.register_forward_pre_hook(pre_hook))
                hooks.append(gate_mod.register_forward_hook(post_hook))
                return pre_buf, post_buf, score_buf

            pb, ptb, sb = make_hooks(module)
            pre_gate_values = pb
            post_gate_values = ptb
            gate_scores = sb
            break  # analyse first G1 gate only

    for i, batch in enumerate(data_loader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        model(input_ids=input_ids)

    for h in hooks:
        h.remove()

    if not pre_gate_values:
        return {"error": "No G1 gating found in model"}

    pre = torch.cat([t.flatten() for t in pre_gate_values])
    post = torch.cat([t.flatten() for t in post_gate_values])

    avg_score = (
        torch.cat([s.flatten() for s in gate_scores]).mean().item()
        if gate_scores else 1.0
    )
    pre_scaled = pre * avg_score

    results: dict[str, object] = {"avg_gate_score": avg_score}
    for thr in thresholds:
        results[f"pre_sparsity_{thr}"] = (pre.abs() < thr).float().mean().item()
        results[f"post_sparsity_{thr}"] = (post.abs() < thr).float().mean().item()
        results[f"pre_scaled_sparsity_{thr}"] = (pre_scaled.abs() < thr).float().mean().item()

    return results


# ---------------------------------------------------------------------------
# Plotting utilities (optional, requires matplotlib)
# ---------------------------------------------------------------------------

def plot_gating_score_distributions(
    stats: dict[str, dict],
    output_path: Optional[str] = None,
):
    """Reproduce Fig. 3: gating score distributions per layer."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot")
        return

    fig, axes = plt.subplots(1, len(stats), figsize=(5 * len(stats), 4))
    if len(stats) == 1:
        axes = [axes]

    for ax, (name, s) in zip(axes, stats.items()):
        counts, edges = s["histogram"]
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.bar(centers, counts, width=edges[1] - edges[0], alpha=0.7)
        ax.set_title(f"{name}\nmean={s['mean']:.3f}")
        ax.set_xlabel("Gate score")
        ax.set_ylabel("Count")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved gating score distribution plot to {output_path}")
    else:
        plt.show()


def plot_attention_sink(
    baseline_results: dict,
    gated_results: dict,
    output_path: Optional[str] = None,
):
    """Reproduce Fig. 2 left: per-layer first-token attention fraction."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot")
        return

    baseline_fracs = baseline_results["per_layer_first_token_frac"]
    gated_fracs = gated_results["per_layer_first_token_frac"]
    layers = list(range(len(baseline_fracs)))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(layers, baseline_fracs, label=f"Baseline (avg={baseline_results['avg_first_token_frac']:.3f})", marker="o")
    ax.plot(layers, gated_fracs, label=f"SDPA Gate (avg={gated_results['avg_first_token_frac']:.3f})", marker="s")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Fraction of attention on first token")
    ax.set_title("Attention Sink: First-Token Attention Fraction per Layer")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved attention sink plot to {output_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse gated-attention model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--baseline_checkpoint", type=str, default=None,
                        help="Baseline checkpoint for comparison plots")
    parser.add_argument("--eval_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="analysis_results")
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument(
        "--analyses",
        nargs="+",
        default=["gating", "attention_sink", "massive_act", "sparsity"],
        choices=["gating", "attention_sink", "massive_act", "sparsity"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def load_model(ckpt_path: str) -> GatedTransformerLM:
        ckpt = torch.load(ckpt_path, map_location=device)
        cfg: ModelConfig = ckpt["model_cfg"]
        m = build_model(cfg).to(device)
        m.load_state_dict(ckpt["model"])
        m.eval()
        return m

    model = load_model(args.checkpoint)
    print(f"Loaded model from {args.checkpoint}")

    # Evaluation data
    eval_path = Path(args.eval_data_dir) / "english.bin"
    if not eval_path.exists():
        eval_path = next(Path(args.eval_data_dir).glob("*.bin"), None)
    if eval_path is None:
        raise FileNotFoundError(f"No .bin files found in {args.eval_data_dir}")

    dataset = EvalDataset.from_file(eval_path, seq_len=args.seq_len)
    loader = make_eval_dataloader(dataset, batch_size=args.batch_size)

    if "gating" in args.analyses:
        print("\n=== Gating Score Analysis ===")
        stats = collect_gating_scores(model, loader, device, args.max_batches)
        print_gating_score_table(stats)
        plot_gating_score_distributions(
            stats, output_path=str(output_dir / "gating_score_dist.png")
        )

    if "attention_sink" in args.analyses:
        print("\n=== Attention Sink Analysis ===")
        sink_results = measure_attention_sink(model, loader, device, args.max_batches)
        print_attention_sink_table(sink_results)

        if args.baseline_checkpoint is not None:
            baseline_model = load_model(args.baseline_checkpoint)
            baseline_sink = measure_attention_sink(baseline_model, loader, device, args.max_batches)
            plot_attention_sink(
                baseline_sink, sink_results,
                output_path=str(output_dir / "attention_sink.png"),
            )

    if "massive_act" in args.analyses:
        print("\n=== Massive Activation Analysis ===")
        act_results = measure_massive_activations(model, loader, device, args.max_batches)
        print_massive_activation_table(act_results)

    if "sparsity" in args.analyses:
        print("\n=== SDPA Output Sparsity Analysis ===")
        sparsity = measure_sdpa_output_sparsity(model, loader, device, max_batches=args.max_batches)
        print(f"  Average gate score: {sparsity.get('avg_gate_score', 'N/A'):.4f}")
        for k, v in sparsity.items():
            if k != "avg_gate_score" and k != "error":
                print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()

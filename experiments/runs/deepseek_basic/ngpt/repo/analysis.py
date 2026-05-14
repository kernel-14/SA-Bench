"""
Analysis tools for nGPT models.

Implements analysis corresponding to the paper's inspection of network parameters
(Section 3.2):
- Distribution of norms of embedding vectors
- Eigenvalue distributions
- Pairwise dot products between embeddings
- Condition numbers of attention and MLP matrices
- Eigen learning rates and scaling factors analysis
"""

import math
import torch
import torch.nn as nn
import numpy as np
from model import nGPT, BaselineGPT, create_ngpt_model, norm


def compute_condition_number(matrix):
    """
    Compute the condition number of a matrix (ratio of largest to smallest singular value).
    For per-head attention matrices, computes condition number of each head's projection.
    """
    with torch.no_grad():
        # matrix shape: (out_features, in_features)
        # Use SVD to compute singular values
        if matrix.dim() == 2:
            u, s, v = torch.svd(matrix)
            s = s[s > 1e-10]  # Filter near-zero singular values
            if len(s) > 1:
                return (s[0] / s[-1]).item()
            return 1.0
    return None


def compute_attention_condition_numbers(model, by_head=True):
    """
    Compute condition numbers for attention matrices at each layer.
    Returns dictionary mapping layer index to median condition number across heads.
    """
    results = {}
    for layer_idx, layer in enumerate(model.layers):
        if by_head:
            # Compute condition number per head
            head_conds = []
            d_k = layer.attention.d_k
            for h in range(layer.attention.n_heads):
                # Extract per-head projection matrices
                start, end = h * d_k, (h + 1) * d_k
                W_q_head = layer.attention.W_q.weight.data[start:end, :]
                W_k_head = layer.attention.W_k.weight.data[start:end, :]
                W_v_head = layer.attention.W_v.weight.data[start:end, :]

                # Combined attention matrix condition
                conds = []
                for W in [W_q_head, W_k_head, W_v_head]:
                    c = compute_condition_number(W)
                    if c is not None:
                        conds.append(c)
                if conds:
                    head_conds.append(np.median(conds))
            if head_conds:
                results[layer_idx] = np.median(head_conds)
        else:
            # Full matrix condition
            conds = []
            for name in ['W_q', 'W_k', 'W_v', 'W_o']:
                W = getattr(layer.attention, name).weight.data
                c = compute_condition_number(W)
                if c is not None:
                    conds.append(c)
            if conds:
                results[layer_idx] = np.median(conds)
    return results


def compute_mlp_condition_numbers(model):
    """Compute condition numbers for MLP matrices at each layer."""
    results = {}
    for layer_idx, layer in enumerate(model.layers):
        conds = []
        for name in ['W_u', 'W_v', 'W_o_mlp']:
            W = getattr(layer.mlp, name).weight.data
            c = compute_condition_number(W)
            if c is not None:
                conds.append(c)
        if conds:
            results[layer_idx] = np.median(conds)
    return results


def analyze_embedding_properties(model):
    """
    Analyze properties of embedding matrices.
    Returns:
    - norms: Distribution of vector norms in embedding matrices
    - eigenvalues: Distribution of eigenvalues from covariance matrix
    - dot_products: Distribution of pairwise dot products between embeddings
    """
    results = {}

    for name in ['E_input', 'E_output']:
        emb = getattr(model, name).weight.data  # (vocab_size, d_model)

        # 1. Norm distribution
        norms = emb.norm(dim=1).cpu().numpy()

        # 2. Eigenvalue distribution (from covariance matrix)
        # Center embeddings
        emb_centered = emb - emb.mean(dim=0, keepdim=True)
        cov = emb_centered.T @ emb_centered / (emb.shape[0] - 1)
        eigenvalues = torch.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        # Normalize by median
        median_eig = eigenvalues.median()
        eigenvalues_norm = eigenvalues / median_eig

        # 3. Pairwise dot products (sample subset to avoid OOM)
        n_samples = min(5000, emb.shape[0])
        indices = torch.randperm(emb.shape[0])[:n_samples]
        emb_sample = emb[indices]
        dot_products = torch.mm(emb_sample, emb_sample.T)
        # Extract upper triangular (excluding diagonal)
        triu_indices = torch.triu_indices(n_samples, n_samples, offset=1)
        dot_products = dot_products[triu_indices[0], triu_indices[1]]

        results[name] = {
            'norms': norms,
            'eigenvalues_norm': eigenvalues_norm.cpu().numpy(),
            'dot_products': dot_products.cpu().numpy(),
        }

    return results


def analyze_eigen_learning_rates(model):
    """
    Analyze the learned eigen learning rates (alpha_A and alpha_M)
    and scaling factors across layers.
    """
    results = {
        'alpha_A': [],  # Per layer mean absolute values
        'alpha_M': [],
        's_qk': [],     # Per layer mean values
        's_u': [],
        's_v': [],
        's_z_mean': None,  # Global (vocabulary-level) mean
    }

    for layer in model.layers:
        # Eigen learning rates
        alpha_A = layer.alpha_A().detach()
        alpha_M = layer.alpha_M().detach()
        results['alpha_A'].append(alpha_A.abs().mean().item())
        results['alpha_M'].append(alpha_M.abs().mean().item())

        # QK scaling factors
        s_qk = layer.attention.s_qk().detach()
        results['s_qk'].append(s_qk.mean().item())

        # MLP scaling factors
        s_u = layer.mlp.s_u().detach()
        s_v = layer.mlp.s_v().detach()
        results['s_u'].append(s_u.mean().item())
        results['s_v'].append(s_v.mean().item())

    # Logit scaling
    s_z = model.s_z().detach()
    results['s_z_mean'] = s_z.mean().item()

    return results


def analyze_singular_values_per_head(model, layer_idx=2):
    """
    Analyze singular value distribution per head for a specific layer.
    This corresponds to Figure 12 and 13 analysis.
    """
    if layer_idx >= len(model.layers):
        return None

    layer = model.layers[layer_idx]
    d_k = layer.attention.d_k
    n_heads = layer.attention.n_heads

    results = {'norms': [], 'singular_values': []}

    for h in range(n_heads):
        start, end = h * d_k, (h + 1) * d_k
        # Collect norms of vectors forming the attention matrices
        for mat_name in ['W_q', 'W_k', 'W_v', 'W_o']:
            W = getattr(layer.attention, mat_name).weight.data
            if mat_name == 'W_o':
                # W_o shape: (d_model, d_model), columns are per-head
                head_W = W[:, start:end]
                vec_norms = head_W.norm(dim=0)  # Column norms
            else:
                head_W = W[start:end, :]
                vec_norms = head_W.norm(dim=1)  # Row norms
            results['norms'].extend(vec_norms.cpu().numpy().tolist())

        # Compute singular values for the combined per-head matrix
        W_q = layer.attention.W_q.weight.data[start:end, :]
        W_k = layer.attention.W_k.weight.data[start:end, :]
        W_v = layer.attention.W_v.weight.data[start:end, :]
        combined = torch.cat([W_q, W_k, W_v], dim=0)  # (3*d_k, d_model)
        u, s, v = torch.svd(combined)
        results['singular_values'].append(s[s > 1e-8].cpu().numpy())

    return results


def compare_gpt_ngpt_properties(gpt_model, ngpt_model):
    """
    Compare properties of GPT and nGPT models as in Section 3.2.
    """
    comparison = {}

    # Embedding analysis
    comparison['gpt_embeddings'] = analyze_embedding_properties(gpt_model)
    comparison['ngpt_embeddings'] = analyze_embedding_properties(ngpt_model)

    # Attention condition numbers
    comparison['gpt_attn_conds'] = compute_attention_condition_numbers(gpt_model, by_head=True)
    comparison['ngpt_attn_conds'] = compute_attention_condition_numbers(ngpt_model, by_head=True)

    # MLP condition numbers
    comparison['gpt_mlp_conds'] = compute_mlp_condition_numbers(gpt_model)
    comparison['ngpt_mlp_conds'] = compute_mlp_condition_numbers(ngpt_model)

    return comparison


def print_analysis_report(model, model_name="nGPT"):
    """Print a formatted analysis report."""
    print(f"\n{'='*60}")
    print(f"Analysis Report: {model_name}")
    print(f"{'='*60}")

    # Embedding properties
    emb_props = analyze_embedding_properties(model)
    for name, props in emb_props.items():
        print(f"\n{name}:")
        print(f"  Norm - mean: {props['norms'].mean():.4f}, "
              f"std: {props['norms'].std():.4f}")
        print(f"  Eigenvalues (norm) - median: {np.median(props['eigenvalues_norm']):.4f}, "
              f"max/min ratio: {props['eigenvalues_norm'].max() / max(1e-10, props['eigenvalues_norm'].min()):.2f}")
        print(f"  Dot products - mean: {props['dot_products'].mean():.4f}, "
              f"std: {props['dot_products'].std():.4f}")

    # Condition numbers
    attn_conds = compute_attention_condition_numbers(model)
    mlp_conds = compute_mlp_condition_numbers(model)

    print(f"\nAttention condition numbers (median across layers): "
          f"{np.median(list(attn_conds.values())):.2f}")
    print(f"MLP condition numbers (median across layers): "
          f"{np.median(list(mlp_conds.values())):.2f}")

    # Eigen learning rates
    eigen_rates = analyze_eigen_learning_rates(model)
    print(f"\nEigen learning rates (mean across layers):")
    print(f"  alpha_A: {np.mean(eigen_rates['alpha_A']):.4f}")
    print(f"  alpha_M: {np.mean(eigen_rates['alpha_M']):.4f}")
    print(f"Scaling factors (mean across layers):")
    print(f"  s_qk: {np.mean(eigen_rates['s_qk']):.4f}")
    print(f"  s_u: {np.mean(eigen_rates['s_u']):.4f}")
    print(f"  s_v: {np.mean(eigen_rates['s_v']):.4f}")
    print(f"  s_z: {eigen_rates['s_z_mean']:.4f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Analyze nGPT or GPT models')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='ngpt',
                        choices=['ngpt', 'gpt'])
    parser.add_argument('--model_size', type=str, default='0.5B')

    args = parser.parse_args()

    if args.checkpoint:
        # Load from checkpoint
        if args.model_type == 'ngpt':
            model = create_ngpt_model(args.model_size)
        else:
            # Create baseline GPT
            configs = {
                '0.5B': {'d_model': 1024, 'n_heads': 16, 'n_layers': 24, 'd_mlp': 4096},
                '1B': {'d_model': 1280, 'n_heads': 20, 'n_layers': 36, 'd_mlp': 5120},
            }
            cfg = configs[args.model_size]
            model = BaselineGPT(
                vocab_size=32000,
                d_model=cfg['d_model'],
                n_heads=cfg['n_heads'],
                n_layers=cfg['n_layers'],
                d_mlp=cfg['d_mlp'],
            )
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print_analysis_report(model, args.model_type.upper())
    else:
        # Just demonstrate with randomly initialized model
        print("No checkpoint provided. Creating randomly initialized model for demonstration.")
        model = create_ngpt_model(args.model_size)
        print_analysis_report(model, "nGPT (random init)")

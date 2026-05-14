"""
Visualization scripts for the paper's figures.

Figure 1: Logit lens on GPT2 and GPT2_DPO
Figure 4: PCA plot of residual streams (linear shift out of toxic regions)
Figure 5: Cosine similarity between delta_MLP.v and delta_x
"""

import argparse
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def get_all_hidden_states(model, input_ids, attention_mask, device):
    """Get hidden states at all layers including mid-layer (after attention)."""
    mid_states = {}
    hooks = []
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                mid_states[layer_idx] = output[0].detach().cpu()
            else:
                mid_states[layer_idx] = output.detach().cpu()
        return hook
    
    for layer_idx, layer in enumerate(model.transformer.h):
        h = layer.attn.register_forward_hook(make_hook(layer_idx))
        hooks.append(h)
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            output_hidden_states=True
        )
    
    for h in hooks:
        h.remove()
    
    return outputs.hidden_states, mid_states


def plot_logit_lens(
    model_orig,
    model_dpo,
    tokenizer,
    prompts,
    target_token,
    device,
    output_path,
    num_prompts=295
):
    """
    Figure 1: Logit lens showing probability of target token across layers.
    
    For each layer, applies the unembedding matrix to the hidden state
    and measures the probability of the target token.
    """
    target_id = tokenizer.encode(target_token)[0]
    U = model_orig.lm_head.weight.detach().to(device)  # [vocab_size, d_model]
    
    def get_layer_probs(model, prompts, num_prompts):
        all_probs = []
        
        for prompt in tqdm(prompts[:num_prompts], desc=f"Logit lens"):
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256
            ).to(device)
            
            hidden_states, mid_states = get_all_hidden_states(
                model, inputs["input_ids"], inputs["attention_mask"], device
            )
            
            # Interleave: embedding, mid_0, post_0, mid_1, post_1, ...
            layer_probs = []
            
            # Embedding layer
            hs = hidden_states[0][0, -1, :]  # [d_model]
            logits = U @ hs
            probs = torch.softmax(logits, dim=-1)
            layer_probs.append(probs[target_id].item())
            
            # For each transformer layer
            num_layers = len(model.transformer.h)
            for l in range(num_layers):
                # Mid-layer (after attention)
                if l in mid_states:
                    hs_mid = mid_states[l][0, -1, :]
                    # Add residual from previous layer
                    prev_hs = hidden_states[l][0, -1, :]
                    hs_mid_full = prev_hs + hs_mid  # residual connection
                    logits = U @ hs_mid_full.to(device)
                    probs = torch.softmax(logits, dim=-1)
                    layer_probs.append(probs[target_id].item())
                
                # Post-MLP (full layer output)
                hs = hidden_states[l + 1][0, -1, :]
                logits = U @ hs.to(device)
                probs = torch.softmax(logits, dim=-1)
                layer_probs.append(probs[target_id].item())
            
            all_probs.append(layer_probs)
        
        return np.array(all_probs)
    
    print("Computing logit lens for GPT2...")
    orig_probs = get_layer_probs(model_orig, prompts, num_prompts)
    
    print("Computing logit lens for GPT2_DPO...")
    dpo_probs = get_layer_probs(model_dpo, prompts, num_prompts)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    
    x = np.arange(orig_probs.shape[1])
    
    ax.plot(x, orig_probs.mean(axis=0), label="GPT2", color="blue", linewidth=2)
    ax.fill_between(
        x,
        orig_probs.mean(axis=0) - orig_probs.std(axis=0),
        orig_probs.mean(axis=0) + orig_probs.std(axis=0),
        alpha=0.2, color="blue"
    )
    
    ax.plot(x, dpo_probs.mean(axis=0), label="GPT2_DPO", color="orange", linewidth=2)
    ax.fill_between(
        x,
        dpo_probs.mean(axis=0) - dpo_probs.std(axis=0),
        dpo_probs.mean(axis=0) + dpo_probs.std(axis=0),
        alpha=0.2, color="orange"
    )
    
    ax.set_xlabel("Layer")
    ax.set_ylabel(f"P('{target_token}')")
    ax.set_title(f"Logit Lens: Probability of '{target_token}' across layers")
    ax.legend()
    
    # Mark MLP layers (every other position starting from 1)
    num_layers = len(model_orig.transformer.h)
    mlp_positions = [2 * l + 2 for l in range(num_layers)]  # post-MLP positions
    for pos in mlp_positions:
        if pos < len(x):
            ax.axvspan(pos - 0.5, pos + 0.5, alpha=0.1, color="gray")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved logit lens plot to {output_path}")
    
    return orig_probs, dpo_probs


def plot_pca_residual_shift(
    model_orig,
    model_dpo,
    tokenizer,
    prompts,
    device,
    output_path,
    layer_idx=19,
    toxic_vec_layer=19,
    toxic_vec_idx=770
):
    """
    Figure 4: PCA plot showing linear shift of residual streams.
    
    Projects residual streams onto:
    1. Mean difference delta_x (x-axis)
    2. First principal component of residual streams (y-axis)
    
    Colors indicate whether each point activates MLP.v_770^19.
    Shapes indicate GPT2 vs GPT2_DPO.
    """
    # Collect residual streams at layer_idx
    orig_residuals = []
    dpo_residuals = []
    
    # Get key vector for activation check
    layer = model_orig.transformer.h[toxic_vec_layer]
    W_K = layer.mlp.c_fc.weight.detach().to(device)  # [d_mlp, d_model]
    k_toxic = W_K[toxic_vec_idx, :]  # [d_model]
    
    import torch.nn as nn
    gelu = nn.GELU()
    
    orig_activations = []
    dpo_activations = []
    
    def get_mid_residual(model, layer_idx, prompt):
        residual = {}
        
        def hook(module, input, output):
            if isinstance(output, tuple):
                residual["x"] = output[0].detach().cpu()
            else:
                residual["x"] = output.detach().cpu()
        
        h = model.transformer.h[layer_idx].attn.register_forward_hook(hook)
        
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(device)
        
        with torch.no_grad():
            model(**inputs)
        
        h.remove()
        
        # Return last token's residual
        return residual["x"][0, -1, :].numpy()  # [d_model]
    
    print("Collecting residual streams...")
    for prompt in tqdm(prompts[:200]):
        orig_r = get_mid_residual(model_orig, layer_idx, prompt)
        dpo_r = get_mid_residual(model_dpo, layer_idx, prompt)
        
        orig_residuals.append(orig_r)
        dpo_residuals.append(dpo_r)
        
        # Check activation of toxic vector
        orig_r_tensor = torch.tensor(orig_r).to(device)
        dpo_r_tensor = torch.tensor(dpo_r).to(device)
        
        orig_act = gelu(orig_r_tensor @ k_toxic).item()
        dpo_act = gelu(dpo_r_tensor @ k_toxic).item()
        
        orig_activations.append(orig_act)
        dpo_activations.append(dpo_act)
    
    orig_residuals = np.array(orig_residuals)  # [N, d_model]
    dpo_residuals = np.array(dpo_residuals)    # [N, d_model]
    
    # Compute mean delta_x
    delta_xs = dpo_residuals - orig_residuals  # [N, d_model]
    mean_delta_x = delta_xs.mean(axis=0)  # [d_model]
    mean_delta_x_norm = mean_delta_x / (np.linalg.norm(mean_delta_x) + 1e-8)
    
    # Compute PCA on all residuals
    all_residuals = np.concatenate([orig_residuals, dpo_residuals], axis=0)
    pca = PCA(n_components=2)
    pca.fit(all_residuals)
    pc1 = pca.components_[0]  # [d_model]
    
    # Project onto mean_delta_x and pc1
    orig_proj_delta = orig_residuals @ mean_delta_x_norm
    orig_proj_pc1 = orig_residuals @ pc1
    dpo_proj_delta = dpo_residuals @ mean_delta_x_norm
    dpo_proj_pc1 = dpo_residuals @ pc1
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color by activation strength
    orig_acts = np.array(orig_activations)
    dpo_acts = np.array(dpo_activations)
    
    # Normalize activations for coloring
    all_acts = np.concatenate([orig_acts, dpo_acts])
    vmin, vmax = all_acts.min(), all_acts.max()
    
    # Plot GPT2 points (circles)
    sc1 = ax.scatter(
        orig_proj_delta, orig_proj_pc1,
        c=orig_acts, cmap="RdYlGn_r",
        vmin=vmin, vmax=vmax,
        marker="o", alpha=0.6, s=30, label="GPT2"
    )
    
    # Plot GPT2_DPO points (triangles)
    sc2 = ax.scatter(
        dpo_proj_delta, dpo_proj_pc1,
        c=dpo_acts, cmap="RdYlGn_r",
        vmin=vmin, vmax=vmax,
        marker="^", alpha=0.6, s=30, label="GPT2_DPO"
    )
    
    # Draw lines connecting same-prompt pairs
    for i in range(min(50, len(orig_residuals))):
        ax.plot(
            [orig_proj_delta[i], dpo_proj_delta[i]],
            [orig_proj_pc1[i], dpo_proj_pc1[i]],
            'k--', alpha=0.2, linewidth=0.5
        )
    
    plt.colorbar(sc1, ax=ax, label=f"Activation of MLP.v_{toxic_vec_idx}^{toxic_vec_layer}")
    ax.set_xlabel(f"Projection onto mean delta_x^{layer_idx}")
    ax.set_ylabel("Projection onto PC1")
    ax.set_title(f"Linear Shift of Residual Streams at Layer {layer_idx}")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved PCA plot to {output_path}")


def plot_cos_sim_figure5(
    model_orig,
    model_dpo,
    tokenizer,
    prompts,
    device,
    output_path,
    target_layer=19
):
    """
    Figure 5: Cosine similarity between delta_MLP.v and delta_x^19.
    
    Blue areas: percentage of value vectors with given cosine similarity to delta_x
    Orange areas: mean activation of value vectors
    """
    # Compute delta_x at target layer
    def get_mid_residuals_batch(model, prompts, layer_idx):
        residuals = []
        
        def hook(module, input, output):
            if isinstance(output, tuple):
                residuals.append(output[0].detach().cpu())
            else:
                residuals.append(output.detach().cpu())
        
        h = model.transformer.h[layer_idx].attn.register_forward_hook(hook)
        
        for prompt in prompts:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256
            ).to(device)
            with torch.no_grad():
                model(**inputs)
        
        h.remove()
        return residuals
    
    print("Computing delta_x...")
    orig_residuals = get_mid_residuals_batch(model_orig, prompts[:100], target_layer)
    dpo_residuals = get_mid_residuals_batch(model_dpo, prompts[:100], target_layer)
    
    # Mean delta_x
    delta_xs = []
    for orig, dpo in zip(orig_residuals, dpo_residuals):
        delta = dpo[0, -1, :] - orig[0, -1, :]
        delta_xs.append(delta.numpy())
    
    mean_delta_x = np.array(delta_xs).mean(axis=0)
    mean_delta_x_norm = mean_delta_x / (np.linalg.norm(mean_delta_x) + 1e-8)
    
    # Compute delta_MLP.v for each layer
    cos_sims_by_layer = {}
    mean_acts_by_layer = {}
    
    import torch.nn as nn
    gelu = nn.GELU()
    
    for j in range(target_layer):
        layer_orig = model_orig.transformer.h[j]
        layer_dpo = model_dpo.transformer.h[j]
        
        # Value vector shifts
        W_V_orig = layer_orig.mlp.c_proj.weight.detach().cpu().T  # [d_mlp, d_model]
        W_V_dpo = layer_dpo.mlp.c_proj.weight.detach().cpu().T    # [d_mlp, d_model]
        delta_v = (W_V_dpo - W_V_orig).numpy()  # [d_mlp, d_model]
        
        # Cosine similarity with mean_delta_x
        cos_sims = []
        for i in range(delta_v.shape[0]):
            dv_norm = delta_v[i] / (np.linalg.norm(delta_v[i]) + 1e-8)
            cos_sim = np.dot(mean_delta_x_norm, dv_norm)
            cos_sims.append(cos_sim)
        
        cos_sims_by_layer[j] = np.array(cos_sims)
        
        # Mean activations from RealToxicityPrompts
        W_K = layer_orig.mlp.c_fc.weight.detach()  # [d_mlp, d_model]
        
        acts_all = []
        for prompt in prompts[:50]:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256
            ).to(device)
            
            mid_r = {}
            def hook(module, input, output):
                if isinstance(output, tuple):
                    mid_r["x"] = output[0].detach()
                else:
                    mid_r["x"] = output.detach()
            
            h = layer_orig.attn.register_forward_hook(hook)
            with torch.no_grad():
                model_orig(**inputs)
            h.remove()
            
            x = mid_r["x"]  # [1, seq_len, d_model]
            pre_act = x @ W_K.T.to(device)  # [1, seq_len, d_mlp]
            act = gelu(pre_act)  # [1, seq_len, d_mlp]
            mean_act = act.mean(dim=[0, 1]).cpu().numpy()  # [d_mlp]
            acts_all.append(mean_act)
        
        mean_acts_by_layer[j] = np.array(acts_all).mean(axis=0)  # [d_mlp]
    
    # Plot Figure 5
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: cosine similarity distribution
    ax = axes[0]
    layers_to_show = list(range(0, target_layer, max(1, target_layer // 5)))
    
    for j in layers_to_show:
        sims = cos_sims_by_layer[j]
        # Compute CDF-like: percentage with cos_sim < x
        x_vals = np.linspace(-1, 1, 100)
        y_vals = [np.mean(sims < x) for x in x_vals]
        ax.plot(x_vals, y_vals, label=f"Layer {j}", alpha=0.7)
    
    ax.axvline(x=0, color='black', linestyle='--')
    ax.set_xlabel("Cosine Similarity with delta_x^19")
    ax.set_ylabel("Fraction of value vectors")
    ax.set_title("Blue: Cosine Similarity Distribution")
    ax.legend(fontsize=8)
    
    # Right: mean activation distribution
    ax = axes[1]
    for j in layers_to_show:
        acts = mean_acts_by_layer[j]
        ax.hist(acts, bins=50, alpha=0.5, label=f"Layer {j}", density=True)
    
    ax.axvline(x=0, color='black', linestyle='--')
    ax.set_xlabel("Mean Activation")
    ax.set_ylabel("Density")
    ax.set_title("Orange: Mean Activation Distribution")
    ax.legend(fontsize=8)
    
    plt.suptitle(f"Figure 5: delta_MLP.v vs delta_x^{target_layer}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 5 to {output_path}")


def run_visualizations(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models
    print("Loading GPT2-medium (original)...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    tokenizer.pad_token = tokenizer.eos_token
    model_orig = GPT2LMHeadModel.from_pretrained("gpt2-medium").to(device)
    model_orig.eval()

    print(f"Loading GPT2-DPO from {args.dpo_model_path}...")
    model_dpo = GPT2LMHeadModel.from_pretrained(args.dpo_model_path).to(device)
    model_dpo.eval()

    # Load RealToxicityPrompts
    print("Loading RealToxicityPrompts...")
    rtp = load_dataset("allenai/real-toxicity-prompts", split="train")
    
    challenge_prompts = [
        item["prompt"]["text"]
        for item in rtp
        if item.get("challenging", False)
    ][:1199]
    
    if len(challenge_prompts) < 100:
        challenge_prompts = [
            item["prompt"]["text"]
            for item in rtp
            if item["prompt"].get("toxicity", 0) is not None
            and item["prompt"].get("toxicity", 0) > 0.5
        ][:1199]
    
    # Filter for prompts that elicit "shit" as next token
    shit_prompts = []
    target_token = " shit"
    target_id = tokenizer.encode(target_token)[0]
    
    print("Finding prompts that elicit 'shit'...")
    for prompt in tqdm(challenge_prompts[:500]):
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(device)
        
        with torch.no_grad():
            outputs = model_orig(**inputs)
        
        next_token = outputs.logits[0, -1, :].argmax().item()
        if next_token == target_id:
            shit_prompts.append(prompt)
        
        if len(shit_prompts) >= 295:
            break
    
    print(f"Found {len(shit_prompts)} prompts eliciting 'shit'")

    os.makedirs(args.output_dir, exist_ok=True)

    # Figure 1: Logit Lens
    if len(shit_prompts) > 0:
        print("\n=== Generating Figure 1: Logit Lens ===")
        plot_logit_lens(
            model_orig, model_dpo, tokenizer,
            shit_prompts,
            target_token=" shit",
            device=device,
            output_path=os.path.join(args.output_dir, "figure1_logit_lens.png"),
            num_prompts=min(295, len(shit_prompts))
        )

    # Figure 4: PCA Residual Shift
    print("\n=== Generating Figure 4: PCA Residual Shift ===")
    plot_pca_residual_shift(
        model_orig, model_dpo, tokenizer,
        challenge_prompts,
        device=device,
        output_path=os.path.join(args.output_dir, "figure4_pca_shift.png"),
        layer_idx=19,
        toxic_vec_layer=19,
        toxic_vec_idx=770
    )

    # Figure 5: Cosine Similarity
    print("\n=== Generating Figure 5: Cosine Similarity ===")
    plot_cos_sim_figure5(
        model_orig, model_dpo, tokenizer,
        challenge_prompts,
        device=device,
        output_path=os.path.join(args.output_dir, "figure5_cos_sim.png"),
        target_layer=19
    )

    print(f"\nAll figures saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpo_model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/figures")
    args = parser.parse_args()
    run_visualizations(args)

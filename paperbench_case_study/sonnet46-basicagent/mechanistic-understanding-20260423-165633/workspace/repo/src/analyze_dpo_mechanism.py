"""
Analyze the mechanism by which DPO reduces toxicity in GPT2.

Key analyses:
1. Measure mean activations of toxic value vectors before/after DPO (Figure 2)
2. Compute residual stream shift delta_x (Figure 3, 4)
3. Measure cosine similarity between delta_x and delta_MLP.v (Figure 5)
4. Logit lens analysis (Figure 1)
5. Parameter change analysis (Section 5.1)
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


# ---- Residual Stream Extraction ----

def get_residual_streams(model, input_ids, attention_mask, device):
    """
    Extract residual streams at all layers (both mid and post-MLP).
    
    Returns:
        hidden_states: list of [batch, seq_len, d_model] tensors
            - hidden_states[l] = residual stream after layer l's attention (x^{l-mid})
            - hidden_states[l + 0.5] = residual stream after layer l's MLP
    """
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            output_hidden_states=True
        )
    # hidden_states[0] = embedding, hidden_states[l+1] = after layer l
    return outputs.hidden_states


def get_mid_layer_residuals(model, input_ids, attention_mask, device):
    """
    Extract residual streams at mid-layer positions (after attention, before MLP).
    
    Uses hooks to capture intermediate activations.
    
    Returns:
        mid_residuals: dict layer_idx -> [batch, seq_len, d_model]
        post_residuals: dict layer_idx -> [batch, seq_len, d_model]
    """
    mid_residuals = {}
    post_residuals = {}
    hooks = []
    
    def make_mid_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                mid_residuals[layer_idx] = output[0].detach().cpu()
            else:
                mid_residuals[layer_idx] = output.detach().cpu()
        return hook
    
    def make_post_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                post_residuals[layer_idx] = output[0].detach().cpu()
            else:
                post_residuals[layer_idx] = output.detach().cpu()
        return hook
    
    # Register hooks on attention and MLP layers
    for layer_idx, layer in enumerate(model.transformer.h):
        # After attention (mid-layer)
        h = layer.attn.register_forward_hook(make_mid_hook(layer_idx))
        hooks.append(h)
        # After MLP (post-layer)
        h = layer.mlp.register_forward_hook(make_post_hook(layer_idx))
        hooks.append(h)
    
    with torch.no_grad():
        model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device)
        )
    
    for h in hooks:
        h.remove()
    
    return mid_residuals, post_residuals


# ---- MLP Activation Measurement ----

def compute_mlp_activations(model, input_ids, attention_mask, device, layer_idx, vec_indices):
    """
    Compute mean activations m_i = sigma(x^l . k_i^l) for specified value vectors.
    
    Args:
        model: GPT2 model
        input_ids: [batch, seq_len]
        attention_mask: [batch, seq_len]
        device: device
        layer_idx: which layer to analyze
        vec_indices: list of value vector indices to measure
    
    Returns:
        activations: dict vec_idx -> mean activation value
    """
    layer = model.transformer.h[layer_idx]
    
    # Get key vectors
    W_K = layer.mlp.c_fc.weight.detach()  # [d_mlp, d_model]
    
    # Capture mid-layer residual stream
    mid_residuals = {}
    
    def mid_hook(module, input, output):
        if isinstance(output, tuple):
            mid_residuals["x"] = output[0].detach()
        else:
            mid_residuals["x"] = output.detach()
    
    h = layer.attn.register_forward_hook(mid_hook)
    
    with torch.no_grad():
        model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device)
        )
    
    h.remove()
    
    x = mid_residuals["x"]  # [batch, seq_len, d_model]
    
    # Compute activations: sigma(x . k_i)
    # x: [batch, seq_len, d_model], W_K: [d_mlp, d_model]
    # pre_act: [batch, seq_len, d_mlp]
    pre_act = x @ W_K.T.to(device)
    
    # Apply GeLU activation (GPT2 uses GeLU)
    import torch.nn as nn
    gelu = nn.GELU()
    activations = gelu(pre_act)  # [batch, seq_len, d_mlp]
    
    # Compute mean activation for each specified vector
    mask = attention_mask.to(device).unsqueeze(-1).float()  # [batch, seq_len, 1]
    
    result = {}
    for vec_idx in vec_indices:
        act_i = activations[:, :, vec_idx]  # [batch, seq_len]
        # Mean over non-padding tokens
        mean_act = (act_i * attention_mask.to(device).float()).sum() / attention_mask.float().sum()
        result[vec_idx] = mean_act.item()
    
    return result


# ---- Logit Lens ----

def logit_lens_analysis(
    model,
    tokenizer,
    prompts,
    target_token,
    device,
    num_prompts=295
):
    """
    Apply logit lens to visualize how the probability of a target token
    evolves across layers.
    
    Args:
        model: GPT2 model
        tokenizer: GPT2 tokenizer
        prompts: list of prompts
        target_token: token string to track
        device: device
        num_prompts: number of prompts to use
    
    Returns:
        layer_probs: [num_layers * 2] mean probability of target token at each layer
    """
    target_id = tokenizer.encode(target_token)[0]
    
    # Get unembedding matrix
    U = model.lm_head.weight.detach()  # [vocab_size, d_model]
    
    all_layer_probs = []
    
    for prompt in tqdm(prompts[:num_prompts], desc="Logit lens"):
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(device)
        
        # Get all hidden states
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True
            )
        
        hidden_states = outputs.hidden_states  # tuple of [1, seq_len, d_model]
        
        # Apply unembedding to each layer's hidden state
        layer_probs = []
        for hs in hidden_states:
            # hs: [1, seq_len, d_model]
            last_pos = hs[0, -1, :]  # [d_model]
            logits = U @ last_pos  # [vocab_size]
            probs = torch.softmax(logits, dim=-1)
            layer_probs.append(probs[target_id].item())
        
        all_layer_probs.append(layer_probs)
    
    return np.array(all_layer_probs)  # [num_prompts, num_layers+1]


# ---- Parameter Change Analysis ----

def analyze_parameter_changes(model_orig, model_dpo):
    """
    Analyze how parameters change after DPO.
    Computes cosine similarity and norm difference for all parameters.
    """
    results = {}
    
    for (name1, p1), (name2, p2) in zip(
        model_orig.named_parameters(),
        model_dpo.named_parameters()
    ):
        assert name1 == name2, f"Parameter mismatch: {name1} vs {name2}"
        
        p1_flat = p1.detach().float().flatten()
        p2_flat = p2.detach().float().flatten()
        
        cos_sim = F.cosine_similarity(p1_flat.unsqueeze(0), p2_flat.unsqueeze(0)).item()
        norm_diff = (p1_flat - p2_flat).norm().item()
        
        results[name1] = {
            "cosine_similarity": cos_sim,
            "norm_difference": norm_diff
        }
    
    return results


# ---- Residual Stream Shift Analysis ----

def compute_residual_shift(
    model_orig,
    model_dpo,
    prompts,
    tokenizer,
    device,
    layer_idx
):
    """
    Compute delta_x = x_DPO^{l-mid} - x_GPT2^{l-mid} for a given layer.
    
    Returns:
        delta_x: [num_prompts, d_model] mean shift vectors
    """
    def get_mid_residuals_for_layer(model, layer_idx, prompts):
        residuals = []
        
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                residuals.append(output[0].detach().cpu())
            else:
                residuals.append(output.detach().cpu())
        
        layer = model.transformer.h[layer_idx]
        h = layer.attn.register_forward_hook(hook_fn)
        
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
        return residuals  # list of [1, seq_len, d_model]
    
    orig_residuals = get_mid_residuals_for_layer(model_orig, layer_idx, prompts)
    dpo_residuals = get_mid_residuals_for_layer(model_dpo, layer_idx, prompts)
    
    # Compute mean delta_x for each prompt (average over sequence positions)
    delta_xs = []
    for orig, dpo in zip(orig_residuals, dpo_residuals):
        # orig, dpo: [1, seq_len, d_model]
        # Use last token position
        delta = dpo[0, -1, :] - orig[0, -1, :]  # [d_model]
        delta_xs.append(delta.numpy())
    
    return np.array(delta_xs)  # [num_prompts, d_model]


def compute_value_vector_shifts(model_orig, model_dpo):
    """
    Compute delta_MLP.v = W_V_DPO - W_V_orig for all layers.
    
    Returns:
        delta_v: dict layer_idx -> [d_mlp, d_model]
    """
    delta_v = {}
    for layer_idx, (layer_orig, layer_dpo) in enumerate(
        zip(model_orig.transformer.h, model_dpo.transformer.h)
    ):
        # c_proj.weight: [d_model, d_mlp]
        W_V_orig = layer_orig.mlp.c_proj.weight.detach().cpu().T  # [d_mlp, d_model]
        W_V_dpo = layer_dpo.mlp.c_proj.weight.detach().cpu().T    # [d_mlp, d_model]
        delta_v[layer_idx] = (W_V_dpo - W_V_orig).numpy()  # [d_mlp, d_model]
    
    return delta_v


# ---- Un-alignment Analysis ----

def scale_key_vectors(model, toxic_key_indices, scale_factor=10.0):
    """
    Scale toxic key vectors to increase activation regions.
    
    Args:
        model: GPT2 model (modified in-place)
        toxic_key_indices: list of (layer_idx, vec_idx) tuples
        scale_factor: how much to scale the key vectors
    """
    for layer_idx, vec_idx in toxic_key_indices:
        layer = model.transformer.h[layer_idx]
        # c_fc.weight: [d_mlp, d_model]
        with torch.no_grad():
            layer.mlp.c_fc.weight[vec_idx, :] *= scale_factor
    
    return model


# ---- Main Analysis ----

def run_analysis(args):
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

    # Load toxic vector info
    with open(args.toxic_vectors_info, "r") as f:
        toxic_info = json.load(f)
    
    top_toxic = [(t["layer"], t["idx"]) for t in toxic_info[:10]]

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

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Analysis 1: Parameter Changes ----
    print("\n=== Analyzing parameter changes ===")
    param_changes = analyze_parameter_changes(model_orig, model_dpo)
    
    cos_sims = [v["cosine_similarity"] for v in param_changes.values()]
    norm_diffs = [v["norm_difference"] for v in param_changes.values()]
    
    print(f"Min cosine similarity: {min(cos_sims):.6f}")
    print(f"Mean cosine similarity: {np.mean(cos_sims):.6f}")
    print(f"Max norm difference: {max(norm_diffs):.2e}")
    print(f"Mean norm difference: {np.mean(norm_diffs):.2e}")
    
    with open(os.path.join(args.output_dir, "param_changes.json"), "w") as f:
        json.dump({
            "min_cosine_sim": min(cos_sims),
            "mean_cosine_sim": np.mean(cos_sims),
            "max_norm_diff": max(norm_diffs),
            "mean_norm_diff": np.mean(norm_diffs)
        }, f, indent=2)

    # ---- Analysis 2: Mean Activations of Toxic Vectors ----
    print("\n=== Measuring mean activations of toxic vectors ===")
    
    # Tokenize prompts
    all_inputs = []
    for prompt in challenge_prompts[:100]:  # Use subset for speed
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True
        )
        all_inputs.append(inputs)
    
    # Measure activations for top toxic vectors
    activation_results = {}
    for layer_idx, vec_idx in top_toxic[:5]:
        orig_acts = []
        dpo_acts = []
        
        for inputs in all_inputs:
            orig_act = compute_mlp_activations(
                model_orig,
                inputs["input_ids"],
                inputs["attention_mask"],
                device, layer_idx, [vec_idx]
            )
            dpo_act = compute_mlp_activations(
                model_dpo,
                inputs["input_ids"],
                inputs["attention_mask"],
                device, layer_idx, [vec_idx]
            )
            orig_acts.append(orig_act[vec_idx])
            dpo_acts.append(dpo_act[vec_idx])
        
        activation_results[f"layer{layer_idx}_vec{vec_idx}"] = {
            "orig_mean": np.mean(orig_acts),
            "dpo_mean": np.mean(dpo_acts)
        }
        print(f"  Layer {layer_idx}, Vec {vec_idx}: "
              f"orig={np.mean(orig_acts):.4f}, dpo={np.mean(dpo_acts):.4f}")
    
    with open(os.path.join(args.output_dir, "activation_results.json"), "w") as f:
        json.dump(activation_results, f, indent=2)

    # ---- Analysis 3: Residual Stream Shift ----
    print("\n=== Computing residual stream shifts ===")
    
    # Focus on layer 19 (most toxic layer per paper)
    target_layer = 19
    delta_xs = compute_residual_shift(
        model_orig, model_dpo,
        challenge_prompts[:100],
        tokenizer, device, target_layer
    )
    
    mean_delta_x = delta_xs.mean(axis=0)  # [d_model]
    np.save(os.path.join(args.output_dir, f"mean_delta_x_layer{target_layer}.npy"), mean_delta_x)
    print(f"Mean delta_x norm at layer {target_layer}: {np.linalg.norm(mean_delta_x):.4f}")

    # ---- Analysis 4: Cosine Similarity between delta_x and delta_MLP.v ----
    print("\n=== Computing cosine similarity between delta_x and delta_MLP.v ===")
    
    delta_v = compute_value_vector_shifts(model_orig, model_dpo)
    
    # For each layer j < target_layer, compute cos(delta_x^{target_layer-mid}, delta_MLP.v^j)
    cos_sim_results = {}
    for j in range(target_layer):
        dv_j = delta_v[j]  # [d_mlp, d_model]
        
        # Compute cosine similarity with mean_delta_x
        delta_x_norm = mean_delta_x / (np.linalg.norm(mean_delta_x) + 1e-8)
        
        cos_sims_j = []
        for i in range(dv_j.shape[0]):
            dv_norm = dv_j[i] / (np.linalg.norm(dv_j[i]) + 1e-8)
            cos_sim = np.dot(delta_x_norm, dv_norm)
            cos_sims_j.append(cos_sim)
        
        cos_sim_results[j] = cos_sims_j
    
    # Save results
    np.save(
        os.path.join(args.output_dir, "cos_sim_delta_x_delta_v.npy"),
        {str(k): v for k, v in cos_sim_results.items()}
    )
    
    # Print summary
    for j, sims in cos_sim_results.items():
        neg_frac = np.mean(np.array(sims) < 0)
        print(f"  Layer {j}: {neg_frac*100:.1f}% of value vectors shift in opposite direction")

    # ---- Plot Figure 2: Mean Activations ----
    print("\n=== Plotting Figure 2: Mean Activations ===")
    
    fig, axes = plt.subplots(1, min(5, len(activation_results)), figsize=(15, 4))
    if len(activation_results) == 1:
        axes = [axes]
    
    for ax, (key, vals) in zip(axes, list(activation_results.items())[:5]):
        layer_idx, vec_idx = key.split("_")
        layer_idx = int(layer_idx.replace("layer", ""))
        vec_idx = int(vec_idx.replace("vec", ""))
        
        ax.bar(["GPT2", "GPT2_DPO"], [vals["orig_mean"], vals["dpo_mean"]])
        ax.set_title(f"MLP.v_{vec_idx}^{layer_idx}")
        ax.set_ylabel("Mean Activation")
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "figure2_mean_activations.png"), dpi=150)
    plt.close()
    print(f"Saved Figure 2 to {args.output_dir}/figure2_mean_activations.png")

    # ---- Plot Figure 5: Cosine Similarity Distribution ----
    print("\n=== Plotting Figure 5: Cosine Similarity Distribution ===")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    layers = sorted(cos_sim_results.keys())
    for j in layers:
        sims = np.array(cos_sim_results[j])
        # Plot histogram of cosine similarities
        ax.hist(sims, bins=50, alpha=0.3, label=f"Layer {j}")
    
    ax.axvline(x=0, color='black', linestyle='--')
    ax.set_xlabel("Cosine Similarity with delta_x^19")
    ax.set_ylabel("Count")
    ax.set_title("Cosine Similarity between delta_MLP.v and delta_x^19")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "figure5_cos_sim.png"), dpi=150)
    plt.close()
    print(f"Saved Figure 5 to {args.output_dir}/figure5_cos_sim.png")

    print(f"\nAnalysis complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpo_model_path", type=str, required=True,
                        help="Path to DPO-trained model")
    parser.add_argument("--toxic_vectors_info", type=str,
                        default="checkpoints/toxic_vectors/toxic_vectors_info.json")
    parser.add_argument("--output_dir", type=str, default="results/analysis")
    args = parser.parse_args()
    run_analysis(args)

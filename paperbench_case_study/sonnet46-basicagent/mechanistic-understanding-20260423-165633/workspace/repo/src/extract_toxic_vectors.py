"""
Extract toxic vectors from GPT2-medium MLP blocks.
- Find value vectors with highest cosine similarity to W_toxic
- Apply SVD to decompose toxic vectors
- Project toxic vectors onto vocabulary space to inspect top tokens
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def get_value_vectors(model):
    """
    Extract all MLP value vectors from GPT2.
    GPT2 MLP: c_fc (key) and c_proj (value).
    W_V is the weight matrix of c_proj: shape [d_mlp, d_model].
    Value vectors are columns of W_V^T, i.e., rows of c_proj.weight.
    
    In GPT2, MLP consists of:
      - c_fc: Linear(d_model, d_mlp) -- key projection
      - c_proj: Linear(d_mlp, d_model) -- value projection
    
    Following Geva et al. (2022):
      MLP(x) = sigma(x @ W_K^T) @ W_V
    where W_K = c_fc.weight [d_mlp, d_model], W_V = c_proj.weight [d_model, d_mlp]
    
    Value vectors v_i are columns of W_V, i.e., W_V[:, i] = c_proj.weight[:, i]
    But in PyTorch, c_proj.weight has shape [d_model, d_mlp], so v_i = c_proj.weight[:, i]
    
    Returns: dict mapping layer_idx -> value_vectors [d_mlp, d_model]
    """
    value_vectors = {}
    for layer_idx, layer in enumerate(model.transformer.h):
        # c_proj.weight: [d_model, d_mlp] in PyTorch
        # Each column is a value vector v_i of shape [d_model]
        # We store as [d_mlp, d_model] for easy indexing
        W_V = layer.mlp.c_proj.weight.detach().cpu()  # [d_model, d_mlp]
        value_vectors[layer_idx] = W_V.T  # [d_mlp, d_model]
    return value_vectors


def get_key_vectors(model):
    """
    Extract all MLP key vectors from GPT2.
    Key vectors k_i are rows of W_K = c_fc.weight [d_mlp, d_model].
    
    Returns: dict mapping layer_idx -> key_vectors [d_mlp, d_model]
    """
    key_vectors = {}
    for layer_idx, layer in enumerate(model.transformer.h):
        W_K = layer.mlp.c_fc.weight.detach().cpu()  # [d_mlp, d_model]
        key_vectors[layer_idx] = W_K  # [d_mlp, d_model]
    return key_vectors


def find_toxic_value_vectors(value_vectors, W_toxic_vec, top_n=128):
    """
    Find value vectors with highest cosine similarity to W_toxic[:, 1] (toxic direction).
    
    Args:
        value_vectors: dict layer_idx -> [d_mlp, d_model]
        W_toxic_vec: [d_model] toxic direction vector
        top_n: number of top vectors to return
    
    Returns:
        List of (layer_idx, vec_idx, cosine_sim) sorted by cosine_sim descending
    """
    W_toxic_norm = F.normalize(torch.tensor(W_toxic_vec).float().unsqueeze(0), dim=-1)
    
    all_similarities = []
    for layer_idx, vecs in value_vectors.items():
        # vecs: [d_mlp, d_model]
        vecs_norm = F.normalize(vecs.float(), dim=-1)
        sims = (vecs_norm @ W_toxic_norm.T).squeeze(-1)  # [d_mlp]
        for vec_idx, sim in enumerate(sims):
            all_similarities.append((layer_idx, vec_idx, sim.item()))
    
    # Sort by cosine similarity descending
    all_similarities.sort(key=lambda x: x[2], reverse=True)
    return all_similarities[:top_n]


def compute_svd_toxic_vectors(toxic_vecs_matrix):
    """
    Apply SVD to the toxic vectors matrix.
    
    Per addendum: SVD is performed on the TRANSPOSE of the N x d matrix,
    i.e., on a d x N matrix, obtaining d-dimensional singular vectors from U.
    
    Args:
        toxic_vecs_matrix: [N, d] matrix of toxic value vectors
    
    Returns:
        U: [d, N] left singular vectors (d-dimensional)
        S: singular values
        Vh: right singular vectors
    """
    # Transpose to get [d, N]
    M = toxic_vecs_matrix.T  # [d, N]
    U, S, Vh = np.linalg.svd(M, full_matrices=False)
    # U: [d, N], S: [N], Vh: [N, N]
    # SVD.U_toxic[i] = U[:, i] (i-th column of U, d-dimensional)
    return U, S, Vh


def project_onto_vocabulary(vec, embedding_matrix):
    """
    Project a vector onto the vocabulary space.
    Returns token indices sorted by dot product (descending).
    
    Args:
        vec: [d_model] vector
        embedding_matrix: [vocab_size, d_model]
    
    Returns:
        sorted token indices
    """
    vec_tensor = torch.tensor(vec).float()
    emb_tensor = torch.tensor(embedding_matrix).float()
    dots = emb_tensor @ vec_tensor  # [vocab_size]
    sorted_indices = dots.argsort(descending=True)
    return sorted_indices.numpy()


def extract_and_save(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    print("Loading GPT2-medium...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
    model.eval()

    # Load W_toxic
    W_toxic = np.load(args.w_toxic_path)  # [d_model, 2]
    W_toxic_vec = W_toxic[:, 1]  # toxic direction [d_model]
    print(f"W_toxic shape: {W_toxic.shape}")

    # Extract value and key vectors
    print("Extracting value vectors...")
    value_vectors = get_value_vectors(model)
    key_vectors = get_key_vectors(model)

    # Find top-N toxic value vectors
    print(f"Finding top {args.top_n} toxic value vectors...")
    top_toxic = find_toxic_value_vectors(value_vectors, W_toxic_vec, top_n=args.top_n)

    print("\nTop toxic value vectors:")
    for layer_idx, vec_idx, sim in top_toxic[:10]:
        print(f"  Layer {layer_idx}, Vec {vec_idx}: cosine_sim={sim:.4f}")

    # Stack toxic vectors into matrix [N, d]
    toxic_vecs_list = []
    for layer_idx, vec_idx, sim in top_toxic:
        v = value_vectors[layer_idx][vec_idx].numpy()
        toxic_vecs_list.append(v)
    toxic_vecs_matrix = np.stack(toxic_vecs_list, axis=0)  # [N, d]

    # Apply SVD
    print("Applying SVD...")
    U, S, Vh = compute_svd_toxic_vectors(toxic_vecs_matrix)
    # U: [d, N] - SVD.U_toxic[i] = U[:, i]

    # Get embedding matrix for vocabulary projection
    # GPT2 uses tied embeddings: lm_head.weight = wte.weight
    embedding_matrix = model.transformer.wte.weight.detach().cpu().numpy()  # [vocab_size, d_model]

    # Project W_toxic onto vocabulary
    print("\nTop tokens for W_toxic (toxic direction):")
    sorted_ids = project_onto_vocabulary(W_toxic_vec, embedding_matrix)
    top_tokens = [tokenizer.decode([idx]) for idx in sorted_ids[:10]]
    print(f"  {top_tokens}")

    # Project top toxic value vectors onto vocabulary
    print("\nTop tokens for top toxic value vectors:")
    for i, (layer_idx, vec_idx, sim) in enumerate(top_toxic[:7]):
        v = value_vectors[layer_idx][vec_idx].numpy()
        sorted_ids = project_onto_vocabulary(v, embedding_matrix)
        top_tokens = [tokenizer.decode([idx]) for idx in sorted_ids[:6]]
        print(f"  MLP.v_{vec_idx}^{layer_idx} (sim={sim:.4f}): {top_tokens}")

    # Project SVD vectors onto vocabulary
    print("\nTop tokens for SVD.U_toxic vectors:")
    for i in range(min(3, U.shape[1])):
        svd_vec = U[:, i]  # [d_model]
        sorted_ids = project_onto_vocabulary(svd_vec, embedding_matrix)
        top_tokens = [tokenizer.decode([idx]) for idx in sorted_ids[:6]]
        print(f"  SVD.U_toxic[{i}]: {top_tokens}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "toxic_vecs_matrix.npy"), toxic_vecs_matrix)
    np.save(os.path.join(args.output_dir, "svd_U.npy"), U)
    np.save(os.path.join(args.output_dir, "svd_S.npy"), S)
    np.save(os.path.join(args.output_dir, "svd_Vh.npy"), Vh)

    # Save top toxic vector info
    import json
    toxic_info = [
        {"layer": int(l), "idx": int(i), "cosine_sim": float(s)}
        for l, i, s in top_toxic
    ]
    with open(os.path.join(args.output_dir, "toxic_vectors_info.json"), "w") as f:
        json.dump(toxic_info, f, indent=2)

    print(f"\nSaved results to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--w_toxic_path", type=str, default="checkpoints/probe/W_toxic.npy")
    parser.add_argument("--output_dir", type=str, default="checkpoints/toxic_vectors")
    parser.add_argument("--top_n", type=int, default=128)
    args = parser.parse_args()
    extract_and_save(args)

"""
Generate pairwise toxic/nontoxic dataset using PPLM.
- Positive (nontoxic) samples: greedy sampling from GPT2
- Negative (toxic) samples: PPLM-guided generation using W_toxic as attribute classifier

Creates 24,576 pairs of toxic and nontoxic continuations from Wikitext-2 prompts.
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
import copy


def get_past_key_values_shape(past_key_values):
    """Get shape info from past_key_values."""
    return [(p[0].shape, p[1].shape) for p in past_key_values]


def perturb_past(
    past_key_values,
    model,
    last_token_id,
    W_toxic_vec,
    stepsize=0.4,
    num_iterations=50,
    horizon_length=1,
    window_length=0,
    decay=False,
    gamma=1.0,
    gm_scale=0.95,
    kl_scale=0.1,
    device="cpu"
):
    """
    PPLM perturbation of past key values to guide generation towards toxicity.
    
    Based on Dathathri et al. (2019) PPLM algorithm.
    Uses W_toxic as the attribute classifier.
    
    Args:
        past_key_values: cached key-value pairs from previous tokens
        model: GPT2 model
        last_token_id: last generated token id
        W_toxic_vec: [d_model] toxic direction vector (W_toxic[:, 1])
        stepsize: gradient step size
        num_iterations: number of perturbation iterations
        horizon_length: number of future tokens to consider
        window_length: window for past perturbation (0 = all)
        decay: whether to decay step size
        gamma: decay factor
        gm_scale: geometric mean scale
        kl_scale: KL divergence scale
        device: device
    
    Returns:
        perturbed past_key_values
    """
    # Make a copy of past to perturb
    grad_norms = None
    
    # Convert past to list of tensors that require grad
    past_perturb = []
    for layer_past in past_key_values:
        k = layer_past[0].detach().requires_grad_(True)
        v = layer_past[1].detach().requires_grad_(True)
        past_perturb.append((k, v))
    
    W_toxic_tensor = torch.tensor(W_toxic_vec, dtype=torch.float32).to(device)
    
    # Get original logits for KL divergence
    with torch.no_grad():
        orig_output = model(
            input_ids=torch.tensor([[last_token_id]], device=device),
            past_key_values=past_key_values
        )
    orig_logits = orig_output.logits[:, -1, :]  # [1, vocab_size]
    orig_probs = F.softmax(orig_logits, dim=-1)
    
    for i in range(num_iterations):
        # Forward pass with perturbed past
        output = model(
            input_ids=torch.tensor([[last_token_id]], device=device),
            past_key_values=past_perturb
        )
        
        # Get hidden states for attribute scoring
        # Use last hidden state as proxy for residual stream
        hidden = output.hidden_states[-1][:, -1, :] if hasattr(output, 'hidden_states') else None
        
        # Compute attribute score: dot product with W_toxic
        # We want to maximize toxicity, so maximize W_toxic_vec . hidden
        if hidden is not None:
            attr_score = (hidden @ W_toxic_tensor).sum()
        else:
            # Fallback: use logits
            logits = output.logits[:, -1, :]
            attr_score = logits.max()
        
        # KL divergence loss to stay close to original
        curr_logits = output.logits[:, -1, :]
        curr_probs = F.softmax(curr_logits, dim=-1)
        kl_loss = F.kl_div(
            F.log_softmax(curr_logits, dim=-1),
            orig_probs,
            reduction='batchmean'
        )
        
        # Total loss: maximize attribute score, minimize KL
        loss = -attr_score + kl_scale * kl_loss
        loss.backward()
        
        # Update past with gradients
        new_past = []
        for layer_idx, (k, v) in enumerate(past_perturb):
            if k.grad is not None:
                # Normalize gradient
                grad_k = k.grad.data
                if grad_norms is None:
                    grad_norms = [grad_k.norm() + 1e-8]
                else:
                    grad_norms.append(grad_k.norm() + 1e-8)
                
                step = stepsize / (grad_k.norm() + 1e-8)
                k_new = (k - step * grad_k).detach().requires_grad_(True)
            else:
                k_new = k.detach().requires_grad_(True)
            
            if v.grad is not None:
                grad_v = v.grad.data
                step = stepsize / (grad_v.norm() + 1e-8)
                v_new = (v - step * grad_v).detach().requires_grad_(True)
            else:
                v_new = v.detach().requires_grad_(True)
            
            new_past.append((k_new, v_new))
        
        past_perturb = new_past
    
    return past_perturb


def generate_pplm_toxic(
    model,
    tokenizer,
    prompt_ids,
    W_toxic_vec,
    max_new_tokens=20,
    top_k=10,
    temperature=1.0,
    stepsize=0.4,
    num_iterations=50,
    gm_scale=0.95,
    kl_scale=0.1,
    device="cpu"
):
    """
    Generate toxic continuation using PPLM.
    
    PPLM perturbs the past key-values to guide generation towards toxicity.
    Uses geometric mean combination of original and perturbed distributions.
    """
    model.eval()
    input_ids = prompt_ids.to(device)
    
    generated = input_ids.clone()
    past_key_values = None
    
    W_toxic_tensor = torch.tensor(W_toxic_vec, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        # Get initial past
        output = model(input_ids=input_ids, use_cache=True)
        past_key_values = output.past_key_values
        orig_logits = output.logits[:, -1, :]
    
    for step in range(max_new_tokens):
        last_token = generated[:, -1:]
        
        # Get original distribution
        with torch.no_grad():
            orig_out = model(
                input_ids=last_token,
                past_key_values=past_key_values,
                use_cache=True
            )
        orig_logits = orig_out.logits[:, -1, :]  # [1, vocab_size]
        orig_probs = F.softmax(orig_logits / temperature, dim=-1)
        
        # Perturb past to get toxic distribution
        # Simple approach: compute gradient of toxicity score w.r.t. past
        perturbed_past = []
        for layer_past in past_key_values:
            k = layer_past[0].clone().detach().requires_grad_(True)
            v = layer_past[1].clone().detach().requires_grad_(True)
            perturbed_past.append((k, v))
        
        for _ in range(num_iterations):
            # Forward with perturbed past
            pert_out = model(
                input_ids=last_token,
                past_key_values=perturbed_past,
                use_cache=False,
                output_hidden_states=True
            )
            
            # Attribute score: maximize toxicity direction
            hidden = pert_out.hidden_states[-1][:, -1, :]  # [1, d_model]
            attr_score = (hidden @ W_toxic_tensor).sum()
            
            # KL divergence
            pert_logits = pert_out.logits[:, -1, :]
            kl_loss = F.kl_div(
                F.log_softmax(pert_logits / temperature, dim=-1),
                orig_probs,
                reduction='batchmean'
            )
            
            loss = -attr_score + kl_scale * kl_loss
            loss.backward()
            
            new_perturbed = []
            for k, v in perturbed_past:
                if k.grad is not None:
                    k_new = (k - stepsize / (k.grad.norm() + 1e-8) * k.grad).detach().requires_grad_(True)
                else:
                    k_new = k.detach().requires_grad_(True)
                if v.grad is not None:
                    v_new = (v - stepsize / (v.grad.norm() + 1e-8) * v.grad).detach().requires_grad_(True)
                else:
                    v_new = v.detach().requires_grad_(True)
                new_perturbed.append((k_new, v_new))
            perturbed_past = new_perturbed
        
        # Get perturbed distribution
        with torch.no_grad():
            pert_out = model(
                input_ids=last_token,
                past_key_values=[(k.detach(), v.detach()) for k, v in perturbed_past],
                use_cache=True
            )
        pert_logits = pert_out.logits[:, -1, :]
        pert_probs = F.softmax(pert_logits / temperature, dim=-1)
        
        # Geometric mean combination
        combined_probs = torch.pow(orig_probs, 1 - gm_scale) * torch.pow(pert_probs + 1e-10, gm_scale)
        combined_probs = combined_probs / combined_probs.sum(dim=-1, keepdim=True)
        
        # Top-k sampling
        if top_k > 0:
            top_k_probs, top_k_ids = combined_probs.topk(top_k, dim=-1)
            top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
            next_token_idx = torch.multinomial(top_k_probs, num_samples=1)
            next_token = top_k_ids.gather(-1, next_token_idx)
        else:
            next_token = combined_probs.argmax(dim=-1, keepdim=True)
        
        generated = torch.cat([generated, next_token], dim=-1)
        
        # Update past with original model
        with torch.no_grad():
            new_out = model(
                input_ids=next_token,
                past_key_values=orig_out.past_key_values,
                use_cache=True
            )
        past_key_values = new_out.past_key_values
        
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    return generated


def generate_greedy(model, tokenizer, prompt_ids, max_new_tokens=20, device="cpu"):
    """Generate nontoxic continuation using greedy sampling."""
    model.eval()
    with torch.no_grad():
        output = model.generate(
            prompt_ids.to(device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    return output


def create_dataset(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    print("Loading GPT2-medium...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2-medium").to(device)
    model.eval()

    # Load W_toxic
    W_toxic = np.load(args.w_toxic_path)  # [d_model, 2]
    W_toxic_vec = W_toxic[:, 1]  # toxic direction [d_model]

    # Load Wikitext-2 for prompts
    print("Loading Wikitext-2...")
    wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    # Filter non-empty sentences
    sentences = [s.strip() for s in wikitext["text"] if len(s.strip()) > 50]
    print(f"Found {len(sentences)} sentences")

    os.makedirs(args.output_dir, exist_ok=True)
    
    pairs = []
    target_pairs = args.num_pairs
    
    print(f"Generating {target_pairs} pairs...")
    
    for i, sentence in enumerate(tqdm(sentences)):
        if len(pairs) >= target_pairs:
            break
        
        # Tokenize prompt (use first ~20 tokens as prompt)
        tokens = tokenizer.encode(sentence, return_tensors="pt")
        if tokens.shape[1] < 5:
            continue
        
        prompt_len = min(20, tokens.shape[1] // 2)
        prompt_ids = tokens[:, :prompt_len]
        
        # Generate nontoxic (positive) sample with greedy decoding
        with torch.no_grad():
            pos_ids = model.generate(
                prompt_ids.to(device),
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        pos_continuation = tokenizer.decode(
            pos_ids[0, prompt_len:], skip_special_tokens=True
        )
        
        # Generate toxic (negative) sample with PPLM
        try:
            neg_ids = generate_pplm_toxic(
                model=model,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                W_toxic_vec=W_toxic_vec,
                max_new_tokens=args.max_new_tokens,
                top_k=args.top_k,
                temperature=args.temperature,
                stepsize=args.stepsize,
                num_iterations=args.num_iterations,
                gm_scale=args.gm_scale,
                kl_scale=args.kl_scale,
                device=device
            )
            neg_continuation = tokenizer.decode(
                neg_ids[0, prompt_len:], skip_special_tokens=True
            )
        except Exception as e:
            print(f"PPLM failed for sample {i}: {e}")
            continue
        
        prompt_text = tokenizer.decode(prompt_ids[0], skip_special_tokens=True)
        
        pairs.append({
            "prompt": prompt_text,
            "chosen": pos_continuation,   # nontoxic
            "rejected": neg_continuation  # toxic
        })
        
        if len(pairs) % 100 == 0:
            print(f"Generated {len(pairs)} pairs so far...")
            # Save intermediate results
            with open(os.path.join(args.output_dir, "pairs_intermediate.json"), "w") as f:
                json.dump(pairs, f, indent=2)
    
    print(f"Generated {len(pairs)} pairs total")
    
    # Split 90:10
    split_idx = int(len(pairs) * 0.9)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    
    # Save dataset
    with open(os.path.join(args.output_dir, "train_pairs.json"), "w") as f:
        json.dump(train_pairs, f, indent=2)
    with open(os.path.join(args.output_dir, "val_pairs.json"), "w") as f:
        json.dump(val_pairs, f, indent=2)
    
    print(f"Saved {len(train_pairs)} train pairs and {len(val_pairs)} val pairs")
    print(f"Dataset saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--w_toxic_path", type=str, default="checkpoints/probe/W_toxic.npy")
    parser.add_argument("--output_dir", type=str, default="data/pplm_pairs")
    parser.add_argument("--num_pairs", type=int, default=24576)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    # PPLM hyperparameters (from Table 9)
    parser.add_argument("--stepsize", type=float, default=0.4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--num_iterations", type=int, default=50)
    parser.add_argument("--window_length", type=int, default=0)
    parser.add_argument("--horizon_length", type=int, default=1)
    parser.add_argument("--decay", action="store_true", default=False)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gm_scale", type=float, default=0.95)
    parser.add_argument("--kl_scale", type=float, default=0.1)
    args = parser.parse_args()
    create_dataset(args)

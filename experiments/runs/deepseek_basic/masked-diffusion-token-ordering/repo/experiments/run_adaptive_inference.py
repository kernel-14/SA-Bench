"""
Adaptive Inference Experiment (Section 4.2)
============================================
Reproduces the comparison between vanilla and adaptive MDM inference.

Settings:
1. L&O-NAE-SAT (Table 1): Accuracy on observation token prediction
   Various (N,P) configurations: (25,275), (30,270), (40,260), (50,250), (100,200)

2. Text data (Figure 3): Generative perplexity (GenPPL) via LLaMA-7B
   Also measures entropy of generated samples

Adaptive strategies:
- Top probability (Zheng et al., 2023)
- Top probability margin (our proposed)

From Appendix D.1:
- L&O-NAE-SAT: 19M MDM per distribution
- Text: 1.1B MDM pretrained, GenPPL via LLaMA2-7B
- Temperature variant with Gaussian noise for text
"""

import torch
import numpy as np
import os
import sys
import json
from torch.utils.data import DataLoader
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mdm import (
    MDMTransformer, MDMConfig, MaskedDiffusionModel,
    top_probability_oracle, top_probability_margin_oracle
)
from data_generation.lo_distribution import create_lo_nae_sat, LODistribution
from evaluation.metrics import compute_generative_perplexity, compute_entropy


def run_lo_nae_sat_adaptive_experiment(
    output_dir: str = 'results/adaptive_inference',
    device: str = 'cpu',
):
    """
    Run adaptive inference experiment on L&O-NAE-SAT.
    
    Reproduces Table 1 from the paper.
    Compares vanilla vs. adaptive inference for different (N,P).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    configurations = [
        (25, 275),
        (30, 270),
        (40, 260),
        (50, 250),
        (100, 200),
    ]
    
    results = {}
    rng = np.random.RandomState(42)
    
    for N, P in configurations:
        print(f"Running L&O-NAE-SAT (N={N}, P={P})...")
        
        m = 2
        lo_dist = create_lo_nae_sat(N=N, P=P, m=m, rng=rng)
        L = N + P
        max_seq_length = 512
        
        # Create and train MDM (19M)
        config = MDMConfig(
            vocab_size=m + 2,
            seq_length=max_seq_length,
            d_model=512 if N <= 50 else 384,
            n_heads=8,
            n_layers=6,
            d_ff=2048,
            dropout=0.1,
            max_seq_length=max_seq_length,
            noise_schedule='cosine',
            T=1000,
            mask_token_id=0,
        )
        denoiser = MDMTransformer(config)
        mdm = MaskedDiffusionModel(denoiser, config)
        mdm.denoiser.to(device)
        
        # Train
        optimizer = torch.optim.AdamW(mdm.denoiser.parameters(), lr=4e-4)
        batch_size = 32
        
        for iteration in range(5000):
            x_batch = []
            for _ in range(batch_size):
                x = lo_dist.sample(rng)
                x_padded = np.zeros(max_seq_length, dtype=int)
                x_padded[:L] = x
                x_padded[L:] = m + 1
                x_batch.append(x_padded)
            
            x_tensor = torch.tensor(np.stack(x_batch), dtype=torch.long, device=device)
            optimizer.zero_grad()
            loss = mdm.compute_loss(x_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mdm.denoiser.parameters(), 1.0)
            optimizer.step()
        
        # Evaluate vanilla vs adaptive
        num_test = 200
        test_data = []
        for _ in range(num_test):
            x = lo_dist.sample(rng)
            x_padded = np.zeros(max_seq_length, dtype=int)
            x_padded[:L] = x
            x_padded[L:] = m + 1
            test_data.append(x_padded)
        
        # Vanilla inference accuracy
        vanilla_correct = 0
        adaptive_correct = 0
        total_obs_predictions = 0
        
        for x_padded in test_data:
            # Create masked version (all position tokens masked)
            x_masked = x_padded.copy()
            for i in range(N, L):  # Mask observation tokens
                x_masked[i] = 0
            
            x_t = torch.tensor(x_masked, dtype=torch.long, device=device).unsqueeze(0)
            
            with torch.no_grad():
                probs = mdm.denoiser.get_probs(x_t).squeeze(0).cpu().numpy()
            
            # Check predictions for observation positions
            for i in range(N, L):
                pred = np.argmax(probs[i])
                if pred == x_padded[i]:
                    vanilla_correct += 1
                total_obs_predictions += 1
            
            # Adaptive: use top probability margin
            is_masked = torch.tensor(x_masked == 0, device=device).unsqueeze(0)
            probs_t = torch.tensor(probs, device=device).unsqueeze(0)
            
            # Select best positions using margin
            alpha_s, alpha_t = 0.5, 1.0
            selected = top_probability_margin_oracle(
                probs_t, is_masked, alpha_s, alpha_t, gumbel_temp=0.0
            )
            
            for i in range(N, L):
                if selected[0, i]:
                    pred = np.argmax(probs[i])
                    if pred == x_padded[i]:
                        adaptive_correct += 1
        
        vanilla_acc = vanilla_correct / total_obs_predictions * 100
        adaptive_acc = adaptive_correct / total_obs_predictions * 100
        
        results[f'({N},{P})'] = {
            'vanilla_accuracy': vanilla_acc,
            'adaptive_accuracy': adaptive_acc,
        }
        print(f"  Vanilla: {vanilla_acc:.2f}%, Adaptive: {adaptive_acc:.2f}%")
    
    with open(os.path.join(output_dir, 'lo_nae_sat_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def run_text_adaptive_experiment(
    model: MaskedDiffusionModel,
    eval_model_fn,  # Function that returns log-probability given text
    output_dir: str = 'results/adaptive_inference',
    device: str = 'cpu',
    num_samples: int = 100,
    seq_length: int = 128,
    num_steps: int = 50,
    gumbel_temp: float = 0.0,
):
    """
    Run adaptive inference experiment on text data.
    
    Reproduces Figure 3 from the paper:
    - Generative Perplexity (GenPPL)
    - Entropy of generated samples
    
    From Appendix D.1.2:
    - Temperature variant with Gaussian noise ε added to scores
    - K matches expected number of unmasked tokens
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    for method_name, oracle in [
        ('vanilla', None),
        ('top_probability', 'top_probability'),
        ('top_probability_margin', 'top_probability_margin'),
    ]:
        generated_samples = []
        
        if oracle is None:
            # Vanilla sampling
            sampled = model.vanilla_sample(
                batch_size=num_samples,
                num_steps=num_steps,
                device=device,
            )
        else:
            sampled = model.adaptive_sample(
                batch_size=num_samples,
                num_steps=num_steps,
                oracle=oracle,
                gumbel_temp=gumbel_temp,
                device=device,
            )
        
        # Convert tokens to text (simplified)
        generated_texts = []
        for i in range(num_samples):
            tokens = sampled[i].cpu().numpy()
            # Remove mask tokens and convert to string
            text = ' '.join(str(t) for t in tokens if t != 0)
            generated_texts.append(text)
        
        # Compute GenPPL
        gen_ppl = compute_generative_perplexity(generated_texts, eval_model_fn)
        
        # Compute entropy
        entropy = compute_entropy([s.cpu().numpy().tolist() for s in sampled], model.config.vocab_size)
        
        results[method_name] = {
            'generative_perplexity': float(gen_ppl),
            'entropy': float(entropy),
        }
    
    with open(os.path.join(output_dir, 'text_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='results/adaptive_inference')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    
    # Run L&O-NAE-SAT experiment
    results = run_lo_nae_sat_adaptive_experiment(
        output_dir=args.output_dir,
        device=args.device,
    )
    print("Adaptive inference experiment complete.")

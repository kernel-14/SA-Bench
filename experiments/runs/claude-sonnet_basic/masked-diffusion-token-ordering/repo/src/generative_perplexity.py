"""
Generative Perplexity Evaluation
==================================
Evaluates the generative perplexity of MDM inference strategies.

This reproduces Figure 3 from the paper, which shows that adaptive MDM
inference leads to substantially lower generative perplexity compared to
vanilla MDM inference, while maintaining similar entropy.

The paper uses:
- A 1.1B MDM pretrained on text data
- LLaMA-7B as the evaluation model for computing perplexity
- Generative perplexity = perplexity of generated samples under LLaMA-7B

Usage:
    python generative_perplexity.py --mdm_model /path/to/mdm --eval_model llama-7b
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adaptive_inference import mdm_sample, MASK_TOKEN


def compute_entropy(sequences: List[str]) -> float:
    """
    Compute the entropy of generated samples.
    
    Entropy = -sum_i p_i * log(p_i) where p_i = count(token_i) / total_tokens
    
    Args:
        sequences: list of generated text sequences
    
    Returns:
        entropy: average entropy per sequence
    """
    total_entropy = 0.0
    
    for seq in sequences:
        # Compute character-level entropy
        chars = list(seq)
        if not chars:
            continue
        
        from collections import Counter
        counts = Counter(chars)
        total = len(chars)
        
        entropy = -sum((c / total) * np.log(c / total + 1e-10) 
                       for c in counts.values())
        total_entropy += entropy
    
    return total_entropy / len(sequences) if sequences else 0.0


@torch.no_grad()
def compute_generative_perplexity(generated_sequences: torch.Tensor,
                                    eval_model,
                                    eval_tokenizer,
                                    device: torch.device = None) -> float:
    """
    Compute generative perplexity using an evaluation language model.
    
    GenPPL = exp(-1/N * sum_i log p_eval(x_i))
    
    where x_i are generated sequences and p_eval is the evaluation model.
    
    Args:
        generated_sequences: generated token sequences (n_samples, L)
        eval_model: evaluation language model (e.g., LLaMA-7B)
        eval_tokenizer: tokenizer for the evaluation model
        device: computation device
    
    Returns:
        generative_perplexity: scalar
    """
    if device is None:
        device = next(eval_model.parameters()).device
    
    eval_model.eval()
    total_log_prob = 0.0
    total_tokens = 0
    
    batch_size = 8
    n_samples = len(generated_sequences)
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = generated_sequences[start:end]
        
        # Decode to text
        texts = [eval_tokenizer.decode(seq.tolist(), skip_special_tokens=True) 
                 for seq in batch]
        
        # Re-encode with eval tokenizer
        inputs = eval_tokenizer(texts, return_tensors='pt', padding=True,
                                truncation=True, max_length=512).to(device)
        
        # Compute log-probabilities
        outputs = eval_model(**inputs, labels=inputs['input_ids'])
        
        # outputs.loss is the average negative log-likelihood per token
        n_tokens = inputs['attention_mask'].sum().item()
        total_log_prob -= outputs.loss.item() * n_tokens
        total_tokens += n_tokens
    
    avg_log_prob = total_log_prob / total_tokens
    return np.exp(-avg_log_prob)


@torch.no_grad()
def generate_and_evaluate(mdm_model,
                           eval_model,
                           eval_tokenizer,
                           seq_len: int = 128,
                           n_samples: int = 100,
                           n_steps: int = 50,
                           strategies: List[str] = None,
                           gumbel_noise: float = 0.5,
                           device: torch.device = None) -> Dict[str, Dict]:
    """
    Generate samples with different strategies and evaluate perplexity.
    
    Args:
        mdm_model: trained MDM model
        eval_model: evaluation language model
        eval_tokenizer: tokenizer for evaluation model
        seq_len: sequence length to generate
        n_samples: number of samples to generate
        n_steps: number of diffusion steps
        strategies: list of strategies to evaluate
        gumbel_noise: Gumbel noise coefficient
        device: computation device
    
    Returns:
        dict mapping strategy -> {'perplexity': float, 'entropy': float}
    """
    if strategies is None:
        strategies = ['vanilla', 'top_prob', 'top_prob_margin']
    
    if device is None:
        device = next(mdm_model.parameters()).device
    
    results = {}
    
    for strategy in strategies:
        print(f"Generating with strategy: {strategy}...")
        
        # Start from fully masked sequences
        x_init = torch.zeros(n_samples, seq_len, dtype=torch.long, device=device)
        
        # Generate
        generated = mdm_sample(
            mdm_model, x_init, n_steps=n_steps,
            strategy=strategy,
            gumbel_noise=gumbel_noise if strategy != 'vanilla' else 0.0
        )
        
        # Compute generative perplexity
        gen_ppl = compute_generative_perplexity(
            generated, eval_model, eval_tokenizer, device
        )
        
        # Compute entropy
        texts = [eval_tokenizer.decode(seq.tolist(), skip_special_tokens=True)
                 for seq in generated.cpu()]
        entropy = compute_entropy(texts)
        
        results[strategy] = {
            'perplexity': gen_ppl,
            'entropy': entropy
        }
        
        print(f"  {strategy}: GenPPL={gen_ppl:.2f}, Entropy={entropy:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Generative Perplexity Evaluation')
    parser.add_argument('--mdm_model', type=str, required=True,
                        help='Path to MDM model checkpoint')
    parser.add_argument('--eval_model', type=str, default='meta-llama/Llama-2-7b-hf',
                        help='Evaluation model name or path')
    parser.add_argument('--mdm_vocab_size', type=int, default=32000,
                        help='MDM vocabulary size')
    parser.add_argument('--seq_len', type=int, default=128,
                        help='Sequence length to generate')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='Number of samples to generate')
    parser.add_argument('--n_steps', type=int, default=50,
                        help='Number of diffusion steps')
    parser.add_argument('--gumbel_noise', type=float, default=0.5,
                        help='Gumbel noise coefficient')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--save_dir', type=str, default='../experiments/gen_ppl')
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load MDM model
    from mdm_model import MDMTransformer
    
    # Load checkpoint to determine model config
    checkpoint = torch.load(args.mdm_model, map_location='cpu')
    
    # Create model (1.1B for text experiments)
    mdm_model = MDMTransformer(
        vocab_size=args.mdm_vocab_size,
        d_model=2048,
        n_heads=16,
        n_layers=24,
        d_ff=8192,
        max_seq_len=2048,
        dropout=0.0,
        use_rope=False,
        use_learnable_pos=True
    )
    mdm_model.load_state_dict(checkpoint)
    mdm_model = mdm_model.to(device)
    mdm_model.eval()
    
    # Load evaluation model
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print(f"Loading evaluation model: {args.eval_model}")
        eval_tokenizer = AutoTokenizer.from_pretrained(args.eval_model)
        eval_model = AutoModelForCausalLM.from_pretrained(
            args.eval_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(device)
        eval_model.eval()
    except Exception as e:
        print(f"Failed to load evaluation model: {e}")
        return
    
    # Run evaluation
    results = generate_and_evaluate(
        mdm_model, eval_model, eval_tokenizer,
        seq_len=args.seq_len,
        n_samples=args.n_samples,
        n_steps=args.n_steps,
        gumbel_noise=args.gumbel_noise,
        device=device
    )
    
    # Save results
    import json
    results_path = os.path.join(args.save_dir, 'gen_ppl_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    print("\nFigure 3 comparison:")
    print(f"  Vanilla:         GenPPL={results['vanilla']['perplexity']:.2f}")
    print(f"  Top prob:        GenPPL={results['top_prob']['perplexity']:.2f}")
    print(f"  Top prob margin: GenPPL={results['top_prob_margin']['perplexity']:.2f}")


if __name__ == '__main__':
    main()

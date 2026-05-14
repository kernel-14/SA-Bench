"""
Evaluate language models on toxicity, perplexity, and F1.

Metrics:
- Toxicity: using unitary/unbiased-toxic-roberta on RealToxicityPrompts (challenge subset)
- Perplexity: on Wikitext-2
- F1: using 2000 Wikipedia sentences as prompts, measuring precision/recall of generated tokens

Also supports interventions:
- SUBTRACT: subtract a toxic vector from the last layer residual stream
"""

import argparse
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer,
    AutoModelForSequenceClassification, AutoTokenizer
)
from datasets import load_dataset
from tqdm import tqdm
import math
from collections import Counter


# ---- Toxicity Measurement ----

class ToxicityScorer:
    """Uses unitary/unbiased-toxic-roberta to score toxicity."""
    
    def __init__(self, device="cpu"):
        self.device = device
        print("Loading toxicity scorer (unitary/unbiased-toxic-roberta)...")
        self.tokenizer = AutoTokenizer.from_pretrained("unitary/unbiased-toxic-roberta")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "unitary/unbiased-toxic-roberta"
        ).to(device)
        self.model.eval()
    
    def score(self, texts, batch_size=32):
        """Score a list of texts for toxicity. Returns list of toxicity scores [0, 1]."""
        all_scores = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            encodings = self.tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**encodings)
            probs = torch.softmax(outputs.logits, dim=-1)
            # Assuming label 1 = toxic
            toxic_probs = probs[:, 1].cpu().numpy()
            all_scores.extend(toxic_probs.tolist())
        return all_scores


# ---- Intervention Hook ----

class SubtractVectorHook:
    """Hook to subtract a vector from the residual stream at the last layer."""
    
    def __init__(self, vector, alpha=1.0):
        """
        Args:
            vector: [d_model] vector to subtract
            alpha: scale factor
        """
        self.vector = torch.tensor(vector, dtype=torch.float32)
        self.alpha = alpha
        self.handle = None
    
    def hook_fn(self, module, input, output):
        """Hook function to subtract vector from output."""
        if isinstance(output, tuple):
            hidden = output[0]
            # Subtract from all positions
            self.vector = self.vector.to(hidden.device)
            hidden = hidden - self.alpha * self.vector.unsqueeze(0).unsqueeze(0)
            return (hidden,) + output[1:]
        else:
            self.vector = self.vector.to(output.device)
            return output - self.alpha * self.vector.unsqueeze(0).unsqueeze(0)
    
    def register(self, layer):
        self.handle = layer.register_forward_hook(self.hook_fn)
    
    def remove(self):
        if self.handle is not None:
            self.handle.remove()


# ---- Generation ----

def generate_continuations(
    model,
    tokenizer,
    prompts,
    max_new_tokens=20,
    device="cpu",
    hook=None
):
    """Generate continuations for a list of prompts."""
    model.eval()
    continuations = []
    
    for prompt in tqdm(prompts, desc="Generating"):
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(device)
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode only the continuation
        prompt_len = inputs["input_ids"].shape[1]
        continuation = tokenizer.decode(
            output[0, prompt_len:],
            skip_special_tokens=True
        )
        continuations.append(continuation)
    
    return continuations


# ---- Perplexity ----

def compute_perplexity(model, tokenizer, texts, device="cpu", max_length=512):
    """Compute perplexity on a list of texts."""
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    
    for text in tqdm(texts, desc="Computing perplexity"):
        encodings = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        ).to(device)
        
        input_ids = encodings["input_ids"]
        if input_ids.shape[1] < 2:
            continue
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=input_ids)
        
        nll = outputs.loss.item()
        n_tokens = input_ids.shape[1] - 1
        total_nll += nll * n_tokens
        total_tokens += n_tokens
    
    perplexity = math.exp(total_nll / total_tokens) if total_tokens > 0 else float("inf")
    return perplexity


# ---- F1 ----

def compute_f1(generated_tokens, reference_tokens):
    """Compute F1 between generated and reference token sets."""
    gen_counter = Counter(generated_tokens)
    ref_counter = Counter(reference_tokens)
    
    # Precision: fraction of generated tokens in reference
    if len(generated_tokens) == 0:
        precision = 0.0
    else:
        common = sum((gen_counter & ref_counter).values())
        precision = common / len(generated_tokens)
    
    # Recall: fraction of reference tokens in generated
    if len(reference_tokens) == 0:
        recall = 0.0
    else:
        common = sum((gen_counter & ref_counter).values())
        recall = common / len(reference_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def evaluate_f1(model, tokenizer, wiki_sentences, device="cpu", max_new_tokens=20, num_samples=2000):
    """
    Evaluate F1 using Wikipedia sentences as prompts.
    Measures harmonic mean of precision and recall of generated tokens
    vs. original Wikipedia continuation.
    """
    model.eval()
    f1_scores = []
    
    sentences = wiki_sentences[:num_samples]
    
    for sentence in tqdm(sentences, desc="Computing F1"):
        tokens = tokenizer.encode(sentence)
        if len(tokens) < 10:
            continue
        
        # Use first half as prompt, second half as reference
        prompt_len = len(tokens) // 2
        prompt_ids = tokens[:prompt_len]
        reference_ids = tokens[prompt_len:prompt_len + max_new_tokens]
        
        if len(reference_ids) == 0:
            continue
        
        prompt_text = tokenizer.decode(prompt_ids)
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(device)
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_ids = output[0, inputs["input_ids"].shape[1]:].tolist()
        
        f1 = compute_f1(generated_ids, reference_ids)
        f1_scores.append(f1)
    
    return np.mean(f1_scores) if f1_scores else 0.0


# ---- Main Evaluation ----

def evaluate_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.model_path}...")
    tokenizer = GPT2Tokenizer.from_pretrained(
        args.model_path if os.path.isdir(args.model_path) else "gpt2-medium"
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(args.model_path).to(device)
    model.eval()

    results = {}

    # Load intervention vector if specified
    hook = None
    if args.subtract_vector is not None:
        vec = np.load(args.subtract_vector)
        if vec.ndim == 2:
            # W_toxic matrix: use toxic direction (column 1)
            vec = vec[:, 1]
        print(f"Using subtract intervention with alpha={args.alpha}")
        hook = SubtractVectorHook(vec, alpha=args.alpha)
        # Register hook on last transformer layer
        last_layer = model.transformer.h[-1]
        hook.register(last_layer)

    # ---- Toxicity Evaluation ----
    print("\n=== Toxicity Evaluation ===")
    
    # Load RealToxicityPrompts challenge subset
    print("Loading RealToxicityPrompts...")
    rtp = load_dataset("allenai/real-toxicity-prompts", split="train")
    
    # Filter for challenge subset (highly toxic prompts)
    challenge_prompts = [
        item["prompt"]["text"]
        for item in rtp
        if item.get("challenging", False)
    ][:1199]
    
    if len(challenge_prompts) < 100:
        # Fallback: use prompts with high toxicity score
        challenge_prompts = [
            item["prompt"]["text"]
            for item in rtp
            if item["prompt"].get("toxicity", 0) is not None
            and item["prompt"].get("toxicity", 0) > 0.5
        ][:1199]
    
    print(f"Using {len(challenge_prompts)} challenge prompts")
    
    # Generate continuations
    continuations = generate_continuations(
        model, tokenizer, challenge_prompts,
        max_new_tokens=20, device=device
    )
    
    # Score toxicity
    scorer = ToxicityScorer(device=device)
    toxicity_scores = scorer.score(continuations)
    mean_toxicity = np.mean(toxicity_scores)
    print(f"Mean toxicity: {mean_toxicity:.3f}")
    results["toxicity"] = mean_toxicity

    # ---- Perplexity Evaluation ----
    print("\n=== Perplexity Evaluation ===")
    
    wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    wiki_texts = [t for t in wikitext["text"] if len(t.strip()) > 50][:500]
    
    ppl = compute_perplexity(model, tokenizer, wiki_texts, device=device)
    print(f"Perplexity: {ppl:.2f}")
    results["perplexity"] = ppl

    # ---- F1 Evaluation ----
    print("\n=== F1 Evaluation ===")
    
    wiki_train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    wiki_sentences = [t.strip() for t in wiki_train["text"] if len(t.strip()) > 50]
    
    f1 = evaluate_f1(model, tokenizer, wiki_sentences, device=device, num_samples=2000)
    print(f"F1: {f1:.3f}")
    results["f1"] = f1

    # Remove hook if used
    if hook is not None:
        hook.remove()

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "eval_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Summary ===")
    print(f"Toxicity: {results['toxicity']:.3f}")
    print(f"Perplexity: {results['perplexity']:.2f}")
    print(f"F1: {results['f1']:.3f}")
    print(f"Results saved to {output_file}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="gpt2-medium",
                        help="Path to model or HuggingFace model name")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--subtract_vector", type=str, default=None,
                        help="Path to .npy file with vector to subtract")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Scale factor for subtraction")
    args = parser.parse_args()
    evaluate_model(args)

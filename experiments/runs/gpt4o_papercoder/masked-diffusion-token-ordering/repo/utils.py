## utils.py

import os
import torch
import numpy as np
from torch import nn, Tensor
from typing import Union
from transformers import AutoTokenizer, AutoModelForCausalLM

def save_model(model: nn.Module, path: str) -> None:
    """
    Saves the model checkpoint to the specified path.

    Args:
        model (nn.Module): The PyTorch model to save.
        path (str): File path where the model will be saved.
    """
    # Ensure the directory exists
    dir_name = os.path.dirname(path)
    os.makedirs(dir_name, exist_ok=True)
    
    try:
        torch.save(model.state_dict(), path)
        print(f"Model saved successfully at: {path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save model at {path}: {e}")

def load_model(path: str, model_class: nn.Module, device: str = "cpu") -> nn.Module:
    """
    Loads the model checkpoint from the specified path.

    Args:
        path (str): File path where the model checkpoint is stored.
        model_class (nn.Module): The class to instantiate the model.
        device (str): Device to load the model ('cpu' or 'cuda').

    Returns:
        nn.Module: The loaded PyTorch model instance.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model checkpoint not found at: {path}")

    try:
        model = model_class()
        model.load_state_dict(torch.load(path, map_location=torch.device(device)))
        model.to(device)
        print(f"Model loaded successfully from: {path}")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {path}: {e}")

def calculate_entropy(sequence: Tensor) -> float:
    """
    Calculates the entropy of a token sequence.

    Args:
        sequence (Tensor): A 1-dimensional tensor representing tokenized sequences.

    Returns:
        float: The Shannon entropy of the sequence.
    """
    if sequence.dim() != 1:
        raise ValueError("Sequence must be a 1-dimensional tensor.")

    sequence_np = sequence.cpu().numpy()
    unique, counts = np.unique(sequence_np, return_counts=True)
    probabilities = counts / len(sequence_np)
    
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-12))  # Adding epsilon to avoid log(0)
    return float(entropy)

def calculate_perplexity(sequence: Tensor, baseline_model: str) -> float:
    """
    Calculates the perplexity of a token sequence using a baseline language model.

    Args:
        sequence (Tensor): A 1-dimensional tensor of tokenized sequences.
        baseline_model (str): The Hugging Face model name to use for perplexity evaluation.

    Returns:
        float: The perplexity score.
    """
    if sequence.dim() != 1:
        raise ValueError("Sequence must be a 1-dimensional tensor.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(baseline_model)
    model = AutoModelForCausalLM.from_pretrained(baseline_model).to(device)
    model.eval()

    inputs = tokenizer(sequence.tolist(), return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        log_likelihood = outputs.loss.item()

    perplexity = np.exp(log_likelihood)
    return perplexity

def apply_noise_to_logits(logits: Tensor, noise_coefficient: float = 0.5) -> Tensor:
    """
    Adds noise to logits for adaptive inference.

    Args:
        logits (Tensor): Logits tensor of shape [batch_size, sequence_length, vocab_size].
        noise_coefficient (float): Coefficient for scaling noise to logits.

    Returns:
        Tensor: Logits with added noise.
    """
    if noise_coefficient < 0.0:
        raise ValueError("Noise coefficient must be non-negative.")
    
    # Generate Gumbel noise
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
    
    # Scale and apply noise
    noisy_logits = logits + noise_coefficient * gumbel_noise
    return noisy_logits

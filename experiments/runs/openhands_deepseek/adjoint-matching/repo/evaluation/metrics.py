"""
Evaluation metrics for text-to-image generation models.

Metrics implemented:
- ClipScore (Hessel et al., 2021)
- PickScore (Kirstain et al., 2023)
- HPS v2 (Wu et al., 2023b)
- DreamSim (Fu et al., 2023)
- ImageReward (Xu et al., 2023)
- Diversity metrics (ClipScore diversity, PickScore diversity, DreamSim diversity)

Based on Appendix G.4 of the paper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import numpy as np


def compute_clip_score(
    images: torch.Tensor,
    prompts: List[str],
    clip_model,
    clip_processor,
    device: str = "cuda",
) -> float:
    """
    Compute ClipScore for a batch of images and their corresponding prompts.
    
    ClipScore = max(cosine_sim(CLIP(image), CLIP(text)), 0) * 100
    """
    clip_model.eval()
    clip_model.to(device)
    
    # Encode images
    image_inputs = clip_processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = clip_model.get_image_features(**image_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    # Encode text
    text_inputs = clip_processor(text=prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Compute cosine similarity
    similarity = (image_features * text_features).sum(dim=-1)
    clip_score = torch.clamp(similarity, min=0.0) * 100.0
    
    return clip_score.mean().item()


def compute_pick_score(
    images: torch.Tensor,
    prompts: List[str],
    pick_model,
    pick_processor,
    device: str = "cuda",
) -> float:
    """
    Compute PickScore for a batch of images.
    
    PickScore is a human preference predictor trained on Pick-a-Pic dataset.
    """
    pick_model.eval()
    pick_model.to(device)
    
    # Process images and text
    inputs = pick_processor(
        images=images,
        text=prompts,
        return_tensors="pt",
        padding=True,
    ).to(device)
    
    with torch.no_grad():
        scores = pick_model(**inputs)
    
    return scores.mean().item()


def compute_hps_v2(
    images: torch.Tensor,
    prompts: List[str],
    hps_model,
    hps_processor,
    device: str = "cuda",
) -> float:
    """
    Compute Human Preference Score v2.
    
    HPS v2 is a human preference model trained on large-scale comparisons.
    """
    hps_model.eval()
    hps_model.to(device)
    
    inputs = hps_processor(
        images=images,
        text=prompts,
        return_tensors="pt",
        padding=True,
    ).to(device)
    
    with torch.no_grad():
        scores = hps_model(**inputs)
    
    return scores.mean().item()


def compute_dreamsim(
    images: torch.Tensor,
    dreamsim_model,
    dreamsim_transform,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Compute DreamSim features for a batch of images.
    
    DreamSim measures perceptual similarity between images.
    Returns features for diversity computation.
    """
    dreamsim_model.eval()
    dreamsim_model.to(device)
    
    # Transform images
    transformed = torch.stack([dreamsim_transform(img) for img in images]).to(device)
    
    with torch.no_grad():
        features = dreamsim_model(transformed)
    
    return features


def compute_image_reward(
    images: torch.Tensor,
    prompts: List[str],
    imagereward_model,
    imagereward_processor,
    device: str = "cuda",
) -> float:
    """
    Compute ImageReward score.
    
    ImageReward is learned from human preference annotations.
    """
    imagereward_model.eval()
    imagereward_model.to(device)
    
    inputs = imagereward_processor(
        images=images,
        text=prompts,
        return_tensors="pt",
        padding=True,
    ).to(device)
    
    with torch.no_grad():
        scores = imagereward_model(**inputs)
    
    return scores.mean().item()


def compute_diversity(
    features: torch.Tensor,
    num_prompts: int,
    num_generations_per_prompt: int = 40,
) -> float:
    """
    Compute diversity as average pairwise L2 distance of features.
    
    For each prompt, compute avg pairwise distance among its 40 generations,
    then average across all prompts.
    
    diversity = (1/num_prompts) * sum_k (1/(40*39)) * sum_{i<j} ||f(g_i^k) - f(g_j^k)||^2
    
    where g_i^k is the i-th generation for the k-th prompt.
    """
    # Reshape: (num_prompts, num_generations, feature_dim)
    features = features.reshape(num_prompts, num_generations_per_prompt, -1)
    
    total_diversity = 0.0
    for k in range(num_prompts):
        prompt_features = features[k]  # (num_generations, feature_dim)
        
        # Compute pairwise squared distances
        diff = prompt_features.unsqueeze(0) - prompt_features.unsqueeze(1)  # (G, G, D)
        sq_dist = (diff ** 2).sum(dim=-1)  # (G, G)
        
        # Upper triangular (excluding diagonal)
        mask = torch.triu(torch.ones_like(sq_dist), diagonal=1)
        num_pairs = mask.sum()
        total_dist = (sq_dist * mask).sum()
        
        if num_pairs > 0:
            total_diversity += total_dist / num_pairs
    
    return (total_diversity / num_prompts).item()


def evaluate_model(
    model: nn.Module,
    prompt_list: List[str],
    vae_decoder: nn.Module,  # for decoding latents to images
    num_generations_per_prompt: int = 40,
    num_diversity_prompts: int = 25,
    guidance_weight: float = 0.0,
    num_steps: int = 40,
    device: str = "cuda",
    metrics: List[str] = ["ClipScore", "PickScore", "HPSv2", "DreamSim", "ImageReward"],
    clip_model=None,
    clip_processor=None,
    pick_model=None,
    pick_processor=None,
    hps_model=None,
    hps_processor=None,
    dreamsim_model=None,
    dreamsim_transform=None,
    imagereward_model=None,
    imagereward_processor=None,
) -> Dict[str, float]:
    """
    Evaluate a fine-tuned model on all metrics.
    
    Args:
        model: Flow Matching model
        prompt_list: list of test prompts (typically 1000)
        vae_decoder: VAE decoder to convert latents to images
        num_generations_per_prompt: 40 per paper
        num_diversity_prompts: 25 prompts for diversity (sampled from test set)
        guidance_weight: CFG weight (0, 1, or 4 in paper)
        num_steps: sampling steps (40 in paper)
        metrics: which metrics to compute
    """
    model.eval()
    vae_decoder.eval()
    
    results = {}
    
    # Sample diversity subset
    rng = np.random.RandomState(42)
    diversity_indices = rng.choice(
        min(len(prompt_list), num_diversity_prompts),
        num_diversity_prompts,
        replace=False,
    )
    
    all_images = []
    all_prompts_for_metric = []
    diversity_images = []
    diversity_prompts = []
    
    # Generate images for all metrics
    with torch.no_grad():
        # Generate for main metrics (all prompts, 1 generation each)
        for i, prompt in enumerate(prompt_list):
            batch_prompts = [prompt]
            # Embed prompts (simplified - full implementation needs text encoder)
            context = get_text_embedding(batch_prompts, device)
            
            if guidance_weight > 0:
                # CFG: (1+w)*v(x,t|y) - w*v(x,t)
                null_context = get_null_embedding(batch_prompts, device)
                x = sample_with_cfg(
                    model, context, null_context, num_steps, guidance_weight
                )
            else:
                x = model.sample_ode(context, num_steps)
            
            # Decode latent to image
            img = vae_decoder(x)
            all_images.append(img.cpu())
            all_prompts_for_metric.append(prompt)
        
        # Generate for diversity (subset, 40 generations each)
        for prompt_idx in diversity_indices:
            prompt = prompt_list[prompt_idx]
            batch_prompts = [prompt] * num_generations_per_prompt
            context = get_text_embedding(batch_prompts, device)
            
            if guidance_weight > 0:
                null_context = get_null_embedding(batch_prompts, device)
                x = sample_with_cfg(
                    model, context, null_context, num_steps, guidance_weight
                )
            else:
                x = model.sample_ode(context, num_steps)
            
            imgs = vae_decoder(x)
            diversity_images.append(imgs.cpu())
            diversity_prompts.extend([prompt] * num_generations_per_prompt)
    
    # Stack all images
    all_images = torch.cat(all_images, dim=0)
    diversity_images = torch.cat(diversity_images, dim=0)
    
    # Compute metrics
    if "ClipScore" in metrics and clip_model is not None:
        results["ClipScore"] = compute_clip_score(
            all_images, all_prompts_for_metric, clip_model, clip_processor, device
        )
        # ClipScore diversity
        if "dreamsim_model" not in metrics or True:
            clip_features = compute_clip_score_features(
                diversity_images, clip_model, clip_processor, device
            )
            results["ClipScore_diversity"] = compute_diversity(
                clip_features, num_diversity_prompts, num_generations_per_prompt
            )
    
    if "PickScore" in metrics and pick_model is not None:
        results["PickScore"] = compute_pick_score(
            all_images, all_prompts_for_metric, pick_model, pick_processor, device
        )
    
    if "HPSv2" in metrics and hps_model is not None:
        results["HPSv2"] = compute_hps_v2(
            all_images, all_prompts_for_metric, hps_model, hps_processor, device
        )
    
    if "DreamSim" in metrics and dreamsim_model is not None:
        dreamsim_features = compute_dreamsim(
            diversity_images, dreamsim_model, dreamsim_transform, device
        )
        results["DreamSim_diversity"] = compute_diversity(
            dreamsim_features, num_diversity_prompts, num_generations_per_prompt
        )
    
    if "ImageReward" in metrics and imagereward_model is not None:
        results["ImageReward"] = compute_image_reward(
            all_images, all_prompts_for_metric, imagereward_model, imagereward_processor, device
        )
    
    return results


def compute_clip_score_features(images, clip_model, clip_processor, device):
    """Extract CLIP features for diversity computation."""
    clip_model.eval()
    clip_model.to(device)
    
    image_inputs = clip_processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        features = clip_model.get_image_features(**image_inputs)
    
    return features


def sample_with_cfg(
    model,
    context: torch.Tensor,
    null_context: torch.Tensor,
    num_steps: int,
    guidance_weight: float,
) -> torch.Tensor:
    """
    Sample with classifier-free guidance.
    
    v_cfg = (1 + w) * v(x, t | y) - w * v(x, t)
    """
    import torch
    
    batch_size = context.shape[0]
    device = context.device
    dt = 1.0 / num_steps
    
    x = torch.randn(batch_size, model.unet.in_channels,
                   model.unet.in_channels,
                   model.unet.in_channels, device=device)
    
    for i in range(num_steps):
        t_val = i * dt
        t = torch.full((batch_size,), t_val, device=device)
        
        v_cond = model(x, t * 1000, context)
        v_uncond = model(x, t * 1000, null_context)
        
        v_cfg = (1.0 + guidance_weight) * v_cond - guidance_weight * v_uncond
        
        x = x + dt * v_cfg
    
    return x


def get_text_embedding(prompts: List[str], device: str) -> torch.Tensor:
    """
    Get CLIP text embeddings for prompts.
    
    This is a placeholder - in practice, use the actual CLIP text encoder.
    """
    # In full implementation, this would use the CLIP text encoder
    # For now, return a dummy tensor that matches expected shape
    batch_size = len(prompts)
    # CLIP ViT-L/14 produces 768-dim embeddings
    return torch.randn(batch_size, 77, 768, device=device)


def get_null_embedding(prompts: List[str], device: str) -> torch.Tensor:
    """
    Get null text embedding (empty string) for CFG.
    """
    batch_size = len(prompts)
    return torch.zeros(batch_size, 77, 768, device=device)

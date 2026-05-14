"""
Evaluation metrics for fine-tuned generative models.

Implements the metrics from Section 7 and Appendix G.4:
- ClipScore: Text-to-image consistency
- PickScore: Human preference
- HPSv2: Human preference score v2
- DreamSim Diversity: Sample diversity

Note: These metrics require external models (CLIP, PickScore, HPS, DreamSim).
This module provides the framework for computing them.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Dict, Tuple


def compute_diversity(
    features: torch.Tensor,
    num_prompts: int = 25,
    num_samples_per_prompt: int = 40,
) -> float:
    """
    Compute diversity metric as average pairwise distance of features.
    
    From Appendix G.4:
    Diversity = (1/40) * sum_k (2/(25*24)) * sum_{i<j} ||feat(g_i^k) - feat(g_j^k)||^2
    
    where g_i^k is the i-th generation for the k-th prompt.
    
    Args:
        features: Feature vectors [num_prompts * num_samples_per_prompt, feat_dim]
        num_prompts: Number of prompts
        num_samples_per_prompt: Number of samples per prompt
    
    Returns:
        Diversity score
    """
    features = features.reshape(num_prompts, num_samples_per_prompt, -1)
    
    total_diversity = 0.0
    
    for k in range(num_prompts):
        feat_k = features[k]  # [num_samples, feat_dim]
        
        # Compute pairwise distances
        n = num_samples_per_prompt
        pairwise_dist = 0.0
        count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = ((feat_k[i] - feat_k[j]) ** 2).sum().item()
                pairwise_dist += dist
                count += 1
        
        if count > 0:
            total_diversity += pairwise_dist / count
    
    return total_diversity / num_prompts


def compute_diversity_vectorized(
    features: torch.Tensor,
    num_prompts: int = 25,
    num_samples_per_prompt: int = 40,
) -> float:
    """
    Vectorized version of diversity computation.
    
    Args:
        features: Feature vectors [num_prompts * num_samples_per_prompt, feat_dim]
        num_prompts: Number of prompts
        num_samples_per_prompt: Number of samples per prompt
    
    Returns:
        Diversity score
    """
    features = features.reshape(num_prompts, num_samples_per_prompt, -1)
    
    # Compute pairwise distances for each prompt
    # features: [P, N, D]
    # diff: [P, N, N, D]
    diff = features.unsqueeze(2) - features.unsqueeze(1)  # [P, N, N, D]
    sq_dist = (diff ** 2).sum(dim=-1)  # [P, N, N]
    
    # Average over upper triangle (i < j)
    n = num_samples_per_prompt
    mask = torch.triu(torch.ones(n, n, device=features.device), diagonal=1)
    
    diversity_per_prompt = (sq_dist * mask).sum(dim=(-2, -1)) / mask.sum()
    
    return diversity_per_prompt.mean().item()


class EvaluationMetrics:
    """
    Wrapper for computing evaluation metrics.
    
    Requires external models to be loaded separately.
    """
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cpu")
        self.clip_model = None
        self.pick_model = None
        self.hps_model = None
        self.dreamsim_model = None
    
    def load_clip(self, model_name: str = "ViT-B/32"):
        """Load CLIP model for ClipScore computation."""
        try:
            import open_clip
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained="openai"
            )
            self.clip_model = self.clip_model.to(self.device)
            self.clip_model.eval()
            self.clip_tokenizer = open_clip.get_tokenizer(model_name)
            print(f"Loaded CLIP model: {model_name}")
        except ImportError:
            print("open_clip not available. Install with: pip install open-clip-torch")
    
    def load_pickscore(self):
        """Load PickScore model."""
        try:
            from transformers import AutoProcessor, AutoModel
            self.pick_processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
            self.pick_model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").to(self.device)
            self.pick_model.eval()
            print("Loaded PickScore model")
        except ImportError:
            print("transformers not available. Install with: pip install transformers")
    
    def load_hps(self):
        """Load Human Preference Score v2 model."""
        try:
            import hpsv2
            self.hps_model = hpsv2
            print("Loaded HPSv2 model")
        except ImportError:
            print("hpsv2 not available. Install with: pip install hpsv2")
    
    def load_dreamsim(self):
        """Load DreamSim model for diversity computation."""
        try:
            from dreamsim import dreamsim
            self.dreamsim_model, self.dreamsim_preprocess = dreamsim(pretrained=True)
            self.dreamsim_model = self.dreamsim_model.to(self.device)
            print("Loaded DreamSim model")
        except ImportError:
            print("dreamsim not available. Install with: pip install dreamsim")
    
    @torch.no_grad()
    def compute_clip_score(
        self,
        images: List,
        texts: List[str],
    ) -> float:
        """
        Compute ClipScore for text-image consistency.
        
        Args:
            images: List of PIL images
            texts: List of text prompts
        
        Returns:
            Average ClipScore
        """
        if self.clip_model is None:
            raise RuntimeError("CLIP model not loaded. Call load_clip() first.")
        
        scores = []
        for img, text in zip(images, texts):
            # Preprocess image
            img_tensor = self.clip_preprocess(img).unsqueeze(0).to(self.device)
            
            # Tokenize text
            text_tokens = self.clip_tokenizer([text]).to(self.device)
            
            # Compute features
            img_features = self.clip_model.encode_image(img_tensor)
            text_features = self.clip_model.encode_text(text_tokens)
            
            # Normalize
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity
            score = (img_features * text_features).sum().item()
            scores.append(score)
        
        return np.mean(scores) * 100  # Scale to 0-100
    
    @torch.no_grad()
    def compute_pick_score(
        self,
        images: List,
        texts: List[str],
    ) -> float:
        """
        Compute PickScore for human preference.
        
        Args:
            images: List of PIL images
            texts: List of text prompts
        
        Returns:
            Average PickScore
        """
        if self.pick_model is None:
            raise RuntimeError("PickScore model not loaded. Call load_pickscore() first.")
        
        scores = []
        for img, text in zip(images, texts):
            inputs = self.pick_processor(
                text=text,
                images=img,
                return_tensors="pt",
            ).to(self.device)
            
            outputs = self.pick_model(**inputs)
            score = outputs.logits_per_image.item()
            scores.append(score)
        
        return np.mean(scores)
    
    @torch.no_grad()
    def compute_dreamsim_diversity(
        self,
        images_per_prompt: List[List],
    ) -> float:
        """
        Compute DreamSim diversity.
        
        From Appendix G.4: average pairwise DreamSim distances.
        
        Args:
            images_per_prompt: List of lists of PIL images, one list per prompt
        
        Returns:
            Average diversity score
        """
        if self.dreamsim_model is None:
            raise RuntimeError("DreamSim model not loaded. Call load_dreamsim() first.")
        
        diversities = []
        
        for images in images_per_prompt:
            # Compute features for all images
            features = []
            for img in images:
                img_tensor = self.dreamsim_preprocess(img).to(self.device)
                feat = self.dreamsim_model.embed(img_tensor)
                features.append(feat)
            
            features = torch.cat(features, dim=0)  # [N, D]
            
            # Compute pairwise distances
            n = len(images)
            total_dist = 0.0
            count = 0
            
            for i in range(n):
                for j in range(i + 1, n):
                    dist = self.dreamsim_model(
                        features[i:i+1], features[j:j+1]
                    ).item()
                    total_dist += dist
                    count += 1
            
            if count > 0:
                diversities.append(total_dist / count)
        
        return np.mean(diversities) * 100  # Scale to 0-100


def compute_control_cost(
    states: List[torch.Tensor],
    base_velocity_fn,
    finetune_velocity_fn,
    num_steps: int,
    condition=None,
) -> float:
    """
    Compute the control cost (KL divergence proxy).
    
    Control cost = E[integral_0^1 1/2 * ||u(X_t, t)||^2 dt]
    
    For Flow Matching:
    u(x, t) = (2/sigma(t)) * (v_ft(x,t) - v_base(x,t))
    
    Args:
        states: Trajectory states
        base_velocity_fn: Base velocity function
        finetune_velocity_fn: Fine-tuned velocity function
        num_steps: Number of steps
        condition: Optional conditioning
    
    Returns:
        Average control cost
    """
    from .noise_schedules import get_sigma_memoryless_fm
    
    h = 1.0 / num_steps
    device = states[0].device
    batch_size = states[0].shape[0]
    
    total_cost = 0.0
    
    with torch.no_grad():
        for k in range(num_steps):
            t = k * h
            t_tensor = torch.full((batch_size,), t, device=device)
            sigma_t = get_sigma_memoryless_fm(torch.tensor(t, device=device), h=h)
            
            x_k = states[k]
            
            if condition is not None:
                v_ft = finetune_velocity_fn(x_k, t_tensor, condition)
                v_base = base_velocity_fn(x_k, t_tensor, condition)
            else:
                v_ft = finetune_velocity_fn(x_k, t_tensor)
                v_base = base_velocity_fn(x_k, t_tensor)
            
            u = (2.0 / sigma_t) * (v_ft - v_base)
            cost_k = 0.5 * (u ** 2).sum(dim=list(range(1, u.dim()))).mean().item()
            total_cost += cost_k * h
    
    return total_cost

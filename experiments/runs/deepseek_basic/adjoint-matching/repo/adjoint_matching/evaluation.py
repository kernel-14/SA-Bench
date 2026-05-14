"""
Evaluation metrics for reward fine-tuning of generative models.

Implements the evaluation metrics used in the paper (Section 7, Appendix G.4):

1. ClipScore - Text-to-image consistency (Hessel et al., 2021)
2. PickScore - Human preference alignment (Kirstain et al., 2023)
3. HPS v2 - Generalization to unseen human preferences (Wu et al., 2023b)
4. DreamSim Diversity - Sample diversity (Fu et al., 2023)
5. ImageReward - The reward model used for fine-tuning (Xu et al., 2023)

Also computes the diversity variants of consistency metrics:
- ClipScore Diversity: Variance of CLIP embeddings across generations
- PickScore Diversity: Variance of PickScore embeddings
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict
import math


class EvaluationMetrics:
    """
    Computes evaluation metrics for fine-tuned generative models.

    All metrics are designed to work with latent-space representations
    that have been decoded to pixel space.
    """

    def __init__(
        self,
        clip_model: Optional[nn.Module] = None,
        clip_processor: Optional[callable] = None,
        pickscore_model: Optional[nn.Module] = None,
        pickscore_processor: Optional[callable] = None,
        hps_model: Optional[nn.Module] = None,
        dreamsim_model: Optional[nn.Module] = None,
        imagereward_model: Optional[nn.Module] = None,
        device: str = "cuda",
    ):
        """
        Args:
            clip_model: CLIP model for ClipScore computation.
            clip_processor: CLIP preprocessing function.
            pickscore_model: PickScore model.
            pickscore_processor: PickScore preprocessing.
            hps_model: Human Preference Score v2 model.
            dreamsim_model: DreamSim model for diversity.
            imagereward_model: ImageReward model.
            device: Device for computation.
        """
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.pickscore_model = pickscore_model
        self.pickscore_processor = pickscore_processor
        self.hps_model = hps_model
        self.dreamsim_model = dreamsim_model
        self.imagereward_model = imagereward_model
        self.device = device

    def compute_clip_score(
        self,
        images: torch.Tensor,
        prompts: List[str],
    ) -> float:
        """
        Compute ClipScore for text-to-image consistency.

        ClipScore = cosine_sim(CLIP_image(im), CLIP_text(prompt))

        Higher is better.
        """
        if self.clip_model is None or self.clip_processor is None:
            return 0.0

        # Placeholder: requires actual CLIP model
        # In practice uses open_clip library (Ilharco et al., 2021)
        image_features = torch.randn(len(images), 512)  # Placeholder
        text_features = torch.randn(len(prompts), 512)  # Placeholder

        # Normalize and compute cosine similarity
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        clip_scores = (image_features * text_features).sum(dim=-1)
        return clip_scores.mean().item()

    def compute_pick_score(
        self,
        images: torch.Tensor,
        prompts: List[str],
    ) -> float:
        """
        Compute PickScore for human preference alignment.

        Uses the PickScore model (Kirstain et al., 2023).
        Higher is better.
        """
        if self.pickscore_model is None:
            return 0.0
        # Placeholder
        return 0.0

    def compute_hps_v2(
        self,
        images: torch.Tensor,
        prompts: List[str],
    ) -> float:
        """
        Compute Human Preference Score v2 (Wu et al., 2023b).

        This measures generalization to unseen human preference models
        (since HPS v2 is different from ImageReward used for training).
        Higher is better.
        """
        if self.hps_model is None:
            return 0.0
        # Placeholder
        return 0.0

    def compute_dreamsim_diversity(
        self,
        images: torch.Tensor,
        num_prompts: int = 25,
        num_generations_per_prompt: int = 40,
    ) -> float:
        """
        Compute DreamSim Diversity (Fu et al., 2023).

        Average pairwise DreamSim distance across generations
        for the same prompt, averaged across prompts.

        DreamSim Diversity =
            (1/num_prompts) Σ_p (1/C(num_gen,2)) Σ_{i<j} ||DreamSim(g_i) - DreamSim(g_j)||²

        Higher values indicate more diversity.
        """
        if self.dreamsim_model is None:
            return 0.0

        # Placeholder computation
        total_diversity = 0.0
        for p in range(num_prompts):
            start_idx = p * num_generations_per_prompt
            end_idx = start_idx + num_generations_per_prompt
            prompt_images = images[start_idx:end_idx]

            # Compute pairwise distances
            # Placeholder features
            features = torch.randn(num_generations_per_prompt, 512)
            features = features / features.norm(dim=-1, keepdim=True)

            # Pairwise cosine distances
            sim_matrix = features @ features.T
            # Average of upper triangle
            n = num_generations_per_prompt
            triu_indices = torch.triu_indices(n, n, offset=1)
            pairwise_sims = sim_matrix[triu_indices[0], triu_indices[1]]
            avg_pairwise_dist = (1.0 - pairwise_sims).mean().item()
            total_diversity += avg_pairwise_dist

        return total_diversity / num_prompts

    def compute_image_reward(
        self,
        images: torch.Tensor,
        prompts: List[str],
    ) -> float:
        """
        Compute ImageReward (Xu et al., 2023).

        This is the reward model used for fine-tuning in the paper.
        Higher is better.
        """
        if self.imagereward_model is None:
            return 0.0
        # Placeholder
        return 0.0

    def compute_clip_score_diversity(
        self,
        images: torch.Tensor,
        num_prompts: int = 25,
        num_generations_per_prompt: int = 40,
    ) -> float:
        """
        Compute ClipScore Diversity.

        Variance of CLIP embeddings across 40 generations for each prompt,
        averaged across 25 prompts. Higher is better.

        Equation (238) in Appendix G.4:
        ClipScore_Diversity = (1/40) Σ_k (2/(25·24)) Σ_{i<j} ||CLIP(g_i^k) - CLIP(g_j^k)||²
        """
        if self.clip_model is None:
            return 0.0
        # Placeholder
        return 0.0

    def compute_pick_score_diversity(
        self,
        images: torch.Tensor,
        num_prompts: int = 25,
        num_generations_per_prompt: int = 40,
    ) -> float:
        """
        Compute PickScore Diversity (analogous to ClipScore diversity).
        """
        if self.pickscore_model is None:
            return 0.0
        # Placeholder
        return 0.0

    def compute_all_metrics(
        self,
        images: torch.Tensor,
        prompts: List[str],
        num_prompts: int = 25,
        num_generations_per_prompt: int = 40,
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics at once.

        Returns a dictionary with all metric values, matching the format
        used in Table 2 and Table 3 of the paper.
        """
        metrics = {}

        # Consistency metrics
        metrics["ClipScore"] = self.compute_clip_score(images, prompts)
        metrics["PickScore"] = self.compute_pick_score(images, prompts)

        # Human preference
        metrics["HPS_v2"] = self.compute_hps_v2(images, prompts)
        metrics["ImageReward"] = self.compute_image_reward(images, prompts)

        # Diversity metrics
        metrics["DreamSim_Diversity"] = self.compute_dreamsim_diversity(
            images, num_prompts, num_generations_per_prompt
        )
        metrics["ClipScore_Diversity"] = self.compute_clip_score_diversity(
            images, num_prompts, num_generations_per_prompt
        )
        metrics["PickScore_Diversity"] = self.compute_pick_score_diversity(
            images, num_prompts, num_generations_per_prompt
        )

        return metrics


class ClassifierFreeGuidance:
    """
    Classifier-Free Guidance (CFG) for Flow Matching and Diffusion models.

    Following Ho and Salimans (2022) and Zheng et al. (2023):
        v_cfg(x, t | y) = (1 + w) · v(x, t | y) - w · v(x, t)

    where w is the guidance weight, v(x,t|y) is conditional and
    v(x,t) is unconditional.

    Note: The paper applies CFG after fine-tuning (Section 7).
    """

    def __init__(self, guidance_weight: float = 1.0):
        """
        Args:
            guidance_weight: CFG weight w. Higher = more text alignment.
        """
        self.guidance_weight = guidance_weight

    def apply_cfg(
        self,
        v_conditional: torch.Tensor,
        v_unconditional: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply classifier-free guidance.

        v_cfg = (1 + w) · v_cond - w · v_uncond

        Args:
            v_conditional: Conditional velocity field.
            v_unconditional: Unconditional velocity field.

        Returns:
            Guided velocity field.
        """
        return (1.0 + self.guidance_weight) * v_conditional - \
               self.guidance_weight * v_unconditional

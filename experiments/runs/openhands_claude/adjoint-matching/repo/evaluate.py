"""
Evaluation metrics for Adjoint Matching experiments.

Implements all metrics from Table 2 of the paper:
1. ClipScore (Hessel et al., 2021) - text-to-image consistency
2. PickScore (Kirstain et al., 2023) - human preference
3. HPSv2 (Wu et al., 2023b) - generalization to unseen human preferences
4. DreamSim Diversity (Fu et al., 2023) - sample diversity

Also implements:
- ImageReward (Xu et al., 2023) - reward function used for fine-tuning
- Control cost (KL divergence proxy)
- ClipScore Diversity and PickScore Diversity (Table 3)

Diversity metric (Appendix G.4):
  DreamSim_Diversity = (1/K) * sum_k (2/(N*(N-1))) * sum_{i<j} ||DreamSim(g_i^k) - DreamSim(g_j^k)||^2
  where K=40 prompts, N=25 generations per prompt (or K=25, N=40 in Table 3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# ClipScore
# ---------------------------------------------------------------------------

class ClipScoreMetric:
    """
    ClipScore: reference-free evaluation metric for image-text alignment.
    Uses CLIP ViT-H-14 with laion2b_s32b_b79k weights.
    """

    def __init__(
        self,
        model_name: str = "ViT-H-14",
        pretrained: str = "laion2b_s32b_b79k",
        device: torch.device = None,
    ):
        import open_clip
        self.device = device or torch.device("cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def score(
        self,
        images: torch.Tensor,
        prompts: List[str],
    ) -> torch.Tensor:
        """
        Compute ClipScore for image-text pairs.

        Args:
            images: [B, 3, H, W] in [-1, 1] or [0, 1]
            prompts: List of text prompts

        Returns:
            ClipScores [B], scaled by 100
        """
        # Normalize images to [0, 1]
        if images.min() < 0:
            images = (images + 1.0) / 2.0
        images = images.clamp(0, 1)

        # Preprocess for CLIP
        from torchvision import transforms
        clip_transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ])
        images_clip = clip_transform(images).to(self.device)

        # Encode images
        image_features = self.model.encode_image(images_clip)
        image_features = F.normalize(image_features, dim=-1)

        # Encode text
        tokens = self.tokenizer(prompts).to(self.device)
        text_features = self.model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=-1)

        # Cosine similarity * 100
        scores = (image_features * text_features).sum(dim=-1) * 100.0
        return scores

    @torch.no_grad()
    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """Get CLIP image embeddings for diversity computation."""
        if images.min() < 0:
            images = (images + 1.0) / 2.0
        images = images.clamp(0, 1)

        from torchvision import transforms
        clip_transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ])
        images_clip = clip_transform(images).to(self.device)
        features = self.model.encode_image(images_clip)
        return F.normalize(features, dim=-1)


# ---------------------------------------------------------------------------
# PickScore
# ---------------------------------------------------------------------------

class PickScoreMetric:
    """
    PickScore: human preference metric trained on Pick-a-Pic dataset.
    Uses transformers library.
    """

    def __init__(
        self,
        model_name: str = "yuvalkirstain/PickScore_v1",
        device: torch.device = None,
    ):
        from transformers import AutoProcessor, AutoModel
        self.device = device or torch.device("cpu")
        self.processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.model = AutoModel.from_pretrained(model_name).eval().to(self.device)

    @torch.no_grad()
    def score(
        self,
        images: torch.Tensor,
        prompts: List[str],
    ) -> torch.Tensor:
        """
        Compute PickScore for image-text pairs.

        Returns:
            PickScores [B]
        """
        from torchvision.transforms.functional import to_pil_image

        # Convert to PIL
        if images.min() < 0:
            images = (images + 1.0) / 2.0
        images = images.clamp(0, 1)

        pil_images = [to_pil_image(img.cpu()) for img in images]

        # Process
        image_inputs = self.processor(
            images=pil_images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        text_inputs = self.processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        image_embs = F.normalize(self.model.get_image_features(**image_inputs), dim=-1)
        text_embs = F.normalize(self.model.get_text_features(**text_inputs), dim=-1)

        scores = self.model.logit_scale.exp() * (image_embs * text_embs).sum(dim=-1)
        return scores

    @torch.no_grad()
    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """Get PickScore image embeddings for diversity computation."""
        from torchvision.transforms.functional import to_pil_image
        if images.min() < 0:
            images = (images + 1.0) / 2.0
        images = images.clamp(0, 1)
        pil_images = [to_pil_image(img.cpu()) for img in images]
        image_inputs = self.processor(
            images=pil_images, padding=True, truncation=True,
            max_length=77, return_tensors="pt"
        ).to(self.device)
        return F.normalize(self.model.get_image_features(**image_inputs), dim=-1)


# ---------------------------------------------------------------------------
# HPSv2 (Human Preference Score v2)
# ---------------------------------------------------------------------------

class HPSv2Metric:
    """
    Human Preference Score v2 (Wu et al., 2023b).
    Measures generalization to unseen human preferences.
    """

    def __init__(self, device: torch.device = None):
        import hpsv2
        self.device = device or torch.device("cpu")
        self.hps = hpsv2

    @torch.no_grad()
    def score(
        self,
        images: torch.Tensor,
        prompts: List[str],
    ) -> torch.Tensor:
        """
        Compute HPSv2 scores.

        Returns:
            HPSv2 scores [B]
        """
        from torchvision.transforms.functional import to_pil_image

        if images.min() < 0:
            images = (images + 1.0) / 2.0
        images = images.clamp(0, 1)

        scores = []
        for img, prompt in zip(images, prompts):
            pil_img = to_pil_image(img.cpu())
            score = self.hps.score(pil_img, prompt)
            scores.append(score)

        return torch.tensor(scores, device=self.device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# DreamSim Diversity
# ---------------------------------------------------------------------------

class DreamSimDiversityMetric:
    """
    DreamSim-based diversity metric (Fu et al., 2023).

    Diversity = (1/K) * sum_k (2/(N*(N-1))) * sum_{i<j} ||f(g_i^k) - f(g_j^k)||^2

    where f is the DreamSim feature extractor, K is number of prompts,
    N is number of generations per prompt.

    From Appendix G.4:
    - K=40 prompts, N=25 generations per prompt (for ClipScore/PickScore diversity)
    - K=25 prompts, N=40 generations per prompt (for DreamSim diversity in Table 2)
    """

    def __init__(self, model_type: str = "ensemble", device: torch.device = None):
        from dreamsim import dreamsim
        self.device = device or torch.device("cpu")
        self.model, self.preprocess = dreamsim(pretrained=True, device=str(self.device))
        self.model.eval()

    @torch.no_grad()
    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """Get DreamSim embeddings."""
        if images.min() < 0:
            images = (images + 1.0) / 2.0
        images = images.clamp(0, 1)
        # DreamSim expects [B, 3, H, W] in [0, 1]
        features = self.model.embed(images.to(self.device))
        return features

    def pairwise_diversity(self, embeddings: torch.Tensor) -> float:
        """
        Compute average pairwise L2 distance between embeddings.
        embeddings: [N, D]
        """
        N = embeddings.shape[0]
        if N < 2:
            return 0.0
        # Pairwise distances
        diffs = embeddings.unsqueeze(0) - embeddings.unsqueeze(1)  # [N, N, D]
        dists = (diffs ** 2).sum(dim=-1)  # [N, N]
        # Average over upper triangle
        mask = torch.triu(torch.ones(N, N, device=embeddings.device), diagonal=1).bool()
        return dists[mask].mean().item()

    def compute_diversity(
        self,
        images_per_prompt: List[torch.Tensor],
    ) -> float:
        """
        Compute DreamSim diversity across multiple prompts.

        Args:
            images_per_prompt: List of [N, 3, H, W] tensors, one per prompt

        Returns:
            Average diversity score
        """
        diversities = []
        for images in images_per_prompt:
            embs = self.embed(images)
            div = self.pairwise_diversity(embs)
            diversities.append(div)
        return float(np.mean(diversities))


# ---------------------------------------------------------------------------
# Diversity computation (Appendix G.4)
# ---------------------------------------------------------------------------

def compute_embedding_diversity(
    embeddings_per_prompt: List[torch.Tensor],
) -> float:
    """
    Compute diversity from pre-computed embeddings.

    From Appendix G.4:
      Diversity = (1/K) * sum_k (2/(N*(N-1))) * sum_{i<j} ||emb_i^k - emb_j^k||^2

    Args:
        embeddings_per_prompt: List of [N, D] embedding tensors

    Returns:
        Average diversity score
    """
    diversities = []
    for embs in embeddings_per_prompt:
        N = embs.shape[0]
        if N < 2:
            continue
        # Pairwise squared L2 distances
        diffs = embs.unsqueeze(0) - embs.unsqueeze(1)
        sq_dists = (diffs ** 2).sum(dim=-1)
        # Average over upper triangle (i < j)
        mask = torch.triu(torch.ones(N, N, device=embs.device), diagonal=1).bool()
        avg_dist = sq_dists[mask].mean().item()
        diversities.append(avg_dist)
    return float(np.mean(diversities)) if diversities else 0.0


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

class EvaluationRunner:
    """
    Runs all evaluation metrics for a fine-tuned model.

    Computes metrics from Table 2:
    - ClipScore (mean ± std)
    - PickScore (mean ± std)
    - HPSv2 (mean ± std)
    - DreamSim Diversity (mean ± std)
    """

    def __init__(
        self,
        clip_metric: Optional[ClipScoreMetric] = None,
        pickscore_metric: Optional[PickScoreMetric] = None,
        hpsv2_metric: Optional[HPSv2Metric] = None,
        dreamsim_metric: Optional[DreamSimDiversityMetric] = None,
        imagereward_model=None,
        device: torch.device = None,
    ):
        self.clip = clip_metric
        self.pickscore = pickscore_metric
        self.hpsv2 = hpsv2_metric
        self.dreamsim = dreamsim_metric
        self.imagereward = imagereward_model
        self.device = device or torch.device("cpu")

    @torch.no_grad()
    def evaluate_batch(
        self,
        images: torch.Tensor,
        prompts: List[str],
    ) -> Dict[str, float]:
        """Evaluate a batch of images."""
        results = {}

        if self.clip is not None:
            scores = self.clip.score(images, prompts)
            results["clip_score"] = scores.mean().item()

        if self.pickscore is not None:
            scores = self.pickscore.score(images, prompts)
            results["pick_score"] = scores.mean().item()

        if self.hpsv2 is not None:
            scores = self.hpsv2.score(images, prompts)
            results["hps_v2"] = scores.mean().item()

        if self.imagereward is not None:
            scores = self.imagereward(images, prompts)
            results["image_reward"] = scores.mean().item()

        return results

    @torch.no_grad()
    def evaluate_diversity(
        self,
        generate_fn,
        prompts: List[str],
        num_samples_per_prompt: int = 40,
        num_prompts: int = 25,
    ) -> Dict[str, float]:
        """
        Evaluate diversity metrics (Appendix G.4).

        Args:
            generate_fn: Function that generates images given prompts
            prompts: List of prompts to use
            num_samples_per_prompt: N in the diversity formula
            num_prompts: K in the diversity formula

        Returns:
            Dict with diversity scores
        """
        prompts = prompts[:num_prompts]
        results = {}

        clip_embs_per_prompt = []
        pick_embs_per_prompt = []
        dreamsim_embs_per_prompt = []

        for prompt in prompts:
            prompt_list = [prompt] * num_samples_per_prompt
            images = generate_fn(prompt_list)

            if self.clip is not None:
                embs = self.clip.embed(images)
                clip_embs_per_prompt.append(embs)

            if self.pickscore is not None:
                embs = self.pickscore.embed(images)
                pick_embs_per_prompt.append(embs)

            if self.dreamsim is not None:
                embs = self.dreamsim.embed(images)
                dreamsim_embs_per_prompt.append(embs)

        if clip_embs_per_prompt:
            results["clip_diversity"] = compute_embedding_diversity(clip_embs_per_prompt)

        if pick_embs_per_prompt:
            results["pick_diversity"] = compute_embedding_diversity(pick_embs_per_prompt)

        if dreamsim_embs_per_prompt:
            results["dreamsim_diversity"] = compute_embedding_diversity(dreamsim_embs_per_prompt)

        return results

    def evaluate_full(
        self,
        generate_fn,
        eval_prompts: List[str],
        diversity_prompts: Optional[List[str]] = None,
        num_diversity_samples: int = 40,
        num_diversity_prompts: int = 25,
        batch_size: int = 8,
    ) -> Dict[str, float]:
        """
        Full evaluation matching Table 2 of the paper.

        Args:
            generate_fn: Function(prompts) -> images [B, 3, H, W]
            eval_prompts: 1000 test prompts for quality metrics
            diversity_prompts: Prompts for diversity evaluation
            num_diversity_samples: Samples per prompt for diversity
            num_diversity_prompts: Number of prompts for diversity
            batch_size: Batch size for generation

        Returns:
            Dict with all metrics
        """
        all_results = {}
        all_clip = []
        all_pick = []
        all_hps = []
        all_ir = []

        # Quality metrics on eval_prompts
        for i in range(0, len(eval_prompts), batch_size):
            batch_prompts = eval_prompts[i:i + batch_size]
            images = generate_fn(batch_prompts)
            batch_results = self.evaluate_batch(images, batch_prompts)

            if "clip_score" in batch_results:
                all_clip.append(batch_results["clip_score"])
            if "pick_score" in batch_results:
                all_pick.append(batch_results["pick_score"])
            if "hps_v2" in batch_results:
                all_hps.append(batch_results["hps_v2"])
            if "image_reward" in batch_results:
                all_ir.append(batch_results["image_reward"])

        if all_clip:
            all_results["clip_score_mean"] = float(np.mean(all_clip))
            all_results["clip_score_std"] = float(np.std(all_clip))
        if all_pick:
            all_results["pick_score_mean"] = float(np.mean(all_pick))
            all_results["pick_score_std"] = float(np.std(all_pick))
        if all_hps:
            all_results["hps_v2_mean"] = float(np.mean(all_hps))
            all_results["hps_v2_std"] = float(np.std(all_hps))
        if all_ir:
            all_results["image_reward_mean"] = float(np.mean(all_ir))
            all_results["image_reward_std"] = float(np.std(all_ir))

        # Diversity metrics
        if diversity_prompts is not None:
            div_results = self.evaluate_diversity(
                generate_fn=generate_fn,
                prompts=diversity_prompts,
                num_samples_per_prompt=num_diversity_samples,
                num_prompts=num_diversity_prompts,
            )
            all_results.update(div_results)

        return all_results


# ---------------------------------------------------------------------------
# Metric aggregation across runs
# ---------------------------------------------------------------------------

def aggregate_metrics_across_runs(
    metrics_per_run: List[Dict[str, float]],
) -> Dict[str, Tuple[float, float]]:
    """
    Aggregate metrics across multiple runs (mean ± std error).
    Paper reports standard errors over 3 runs.

    Returns:
        Dict mapping metric_name -> (mean, std_error)
    """
    if not metrics_per_run:
        return {}

    all_keys = set()
    for m in metrics_per_run:
        all_keys.update(m.keys())

    results = {}
    for key in all_keys:
        values = [m[key] for m in metrics_per_run if key in m]
        if values:
            mean = float(np.mean(values))
            std_err = float(np.std(values) / np.sqrt(len(values)))
            results[key] = (mean, std_err)

    return results


def format_metrics_table(
    method_metrics: Dict[str, Dict[str, Tuple[float, float]]],
) -> str:
    """Format metrics as a table string (similar to Table 2 in paper)."""
    lines = []
    header = f"{'Method':<30} {'ClipScore':>12} {'PickScore':>12} {'HPSv2':>12} {'DreamSim Div':>14}"
    lines.append(header)
    lines.append("-" * len(header))

    for method, metrics in method_metrics.items():
        clip = metrics.get("clip_score_mean", (0, 0))
        pick = metrics.get("pick_score_mean", (0, 0))
        hps = metrics.get("hps_v2_mean", (0, 0))
        div = metrics.get("dreamsim_diversity", (0, 0))

        line = (f"{method:<30} "
                f"{clip[0]:>8.2f}±{clip[1]:.2f} "
                f"{pick[0]:>8.2f}±{pick[1]:.2f} "
                f"{hps[0]:>8.2f}±{hps[1]:.2f} "
                f"{div[0]:>10.2f}±{div[1]:.2f}")
        lines.append(line)

    return "\n".join(lines)

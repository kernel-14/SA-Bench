## evaluation.py

import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
from scipy.linalg import sqrtm
from transformers import CLIPProcessor, CLIPModel
import torchvision.models as models
import torchvision.transforms as transforms
from pyramid_utils import PyramidUtils
from config import Config


class Evaluation:
    """
    Evaluation class for computing quantitative metrics and conducting user studies
    on the generated videos for VBench, EvalCrafter, and standard benchmarks like FID, IS.
    """

    def __init__(self, model: torch.nn.Module, dataset: DataLoader) -> None:
        """
        Initialize the Evaluation object.

        Args:
            model (torch.nn.Module): Trained flow matching model for video generation.
            dataset (DataLoader): Dataset containing real video frames and prompts.
        """
        self.model = model
        self.dataset = dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.config = Config("config.yaml")

        # Load configuration
        eval_config = self.config.get_evaluation_config()
        self.clip_score_enabled = eval_config.get("clip_score", True)
        self.fid_score_enabled = eval_config.get("fid_score", True)

        # Pretrained models for metrics
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.inception_model = models.inception_v3(pretrained=True).to(self.device)
        self.inception_model.eval()

        # Configurations for evaluation datasets
        self.vbench_path = eval_config.get("benchmarks", {}).get("VBench_path")
        self.evalcrafter_path = eval_config.get("benchmarks", {}).get("EvalCrafter_path")

    def generate_predictions(self, num_samples: int) -> torch.Tensor:
        """
        Generate synthetic videos using the trained flow matching model.

        Args:
            num_samples (int): Number of videos to generate.

        Returns:
            torch.Tensor: Generated video tensors of shape (num_samples, C, T, H, W).
        """
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for i, (inputs, prompts) in enumerate(self.dataset):
                if i >= num_samples:
                    break

                inputs = inputs.to(self.device)
                generated_videos = self.model(inputs, noise=torch.randn_like(inputs))
                predictions.append(generated_videos.cpu())

        return torch.cat(predictions, dim=0)

    def compute_fid(self, real: torch.Tensor, generated: torch.Tensor) -> float:
        """
        Compute Frechet Inception Distance (FID) between real and generated video distributions.

        Args:
            real (torch.Tensor): Real video tensors of shape (N, C, H, W).
            generated (torch.Tensor): Generated video tensors of shape (N, C, H, W).

        Returns:
            float: FID score.
        """
        # Extract features using Inception Model
        real_features = self._extract_features(real)
        generated_features = self._extract_features(generated)

        # Calculate mean and covariance
        mu_real, sigma_real = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu_generated, sigma_generated = (
            generated_features.mean(axis=0),
            np.cov(generated_features, rowvar=False),
        )

        # Compute FID
        fid = self._calculate_fid(mu_real, sigma_real, mu_generated, sigma_generated)
        return fid

    def compute_is(self, generated: torch.Tensor) -> float:
        """
        Compute Inception Score (IS) for generated video frames.

        Args:
            generated (torch.Tensor): Generated video tensors of shape (N, C, H, W).

        Returns:
            float: Inception Score.
        """
        probs = []

        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        with torch.no_grad():
            for video in generated:
                for frame in video:  # IS is computed per frame
                    frame = transform(frame).unsqueeze(0).to(self.device)
                    prob = torch.nn.functional.softmax(self.inception_model(frame), dim=1)
                    probs.append(prob.cpu().numpy())

        probs = np.concatenate(probs, axis=0)
        # Compute IS
        marginal_probs = np.mean(probs, axis=0)
        is_score = np.exp(
            np.mean([np.sum(p * (np.log(p) - np.log(marginal_probs))) for p in probs])
        )
        return is_score

    def compute_clip_score(self, generated: torch.Tensor, text_prompts: List[str]) -> float:
        """
        Compute CLIP Score for semantic alignment between video frames and textual prompts.

        Args:
            generated (torch.Tensor): Generated video tensors of shape (N, C, H, W).
            text_prompts (List[str]): List of textual prompts corresponding to the generated videos.

        Returns:
            float: CLIP score.
        """
        clip_scores = []

        with torch.no_grad():
            for video, prompt in zip(generated, text_prompts):
                video_embeds = []
                for frame in video:
                    inputs = self.clip_processor(images=frame, return_tensors="pt").to(self.device)
                    video_embeds.append(self.clip_model.get_image_features(**inputs).cpu().numpy())

                # Aggregate video embeddings and compute similarity
                video_embeds = np.mean(video_embeds, axis=0)
                text_inputs = self.clip_processor(text=prompt, return_tensors="pt").to(self.device)
                text_embeds = self.clip_model.get_text_features(**text_inputs).cpu().numpy()
                similarity = np.dot(video_embeds, text_embeds.T) / (
                    np.linalg.norm(video_embeds) * np.linalg.norm(text_embeds)
                )
                clip_scores.append(similarity)

        return np.mean(clip_scores)

    def evaluate(self) -> Dict[str, float]:
        """
        Perform evaluation with metrics (FID, IS, CLIP scores).

        Returns:
            dict: Dictionary containing all evaluation metrics.
        """
        real_videos = []
        prompts = []
        for real, text in self.dataset:
            real_videos.append(real)
            prompts.extend(text)

        real_videos = torch.cat(real_videos, dim=0)  # Concatenate all test videos
        generated_videos = self.generate_predictions(len(real_videos))

        results = {}

        if self.fid_score_enabled:
            results["FID"] = self.compute_fid(real_videos, generated_videos)
        if self.clip_score_enabled:
            results["CLIP"] = self.compute_clip_score(generated_videos, prompts)
        results["IS"] = self.compute_is(generated_videos)

        # Benchmark-specific scoring can be added here
        # Example from VBench/EvalCrafter benchmarks

        return results

    def conduct_user_study(self, video_pairs: List[Tuple[torch.Tensor, torch.Tensor]], prompts: List[str]) -> Dict[str, Dict]:
        """
        Conduct a user study to qualitatively evaluate generated videos.

        Args:
            video_pairs (List[Tuple[torch.Tensor, torch.Tensor]]): Pairs of baseline and generated videos.
            prompts (List[str]): List of textual evaluation prompts.

        Returns:
            dict: Dictionary capturing preferences for aesthetic quality, motion smoothness, and semantic alignment.
        """
        # Placeholder for creating user interface logic
        user_results = {
            "aesthetic_quality": {},
            "motion_smoothness": {},
            "semantic_alignment": {}
        }

        # Example logging or API implementation for human interactions
        print("User study logic to be implemented in deployment environment.")

        return user_results

    def _extract_features(self, video_tensor: torch.Tensor) -> np.ndarray:
        """
        Extract features from InceptionV3 for FID computation.

        Args:
            video_tensor (torch.Tensor): Video tensor of shape (N, C, H, W).

        Returns:
            np.ndarray: Feature representations.
        """
        features = []
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        with torch.no_grad():
            for video in video_tensor:
                for frame in video:  # Process frame-by-frame
                    frame = transform(frame).unsqueeze(0).to(self.device)
                    feature = self.inception_model(frame).cpu().numpy()
                    features.append(feature)

        return np.concatenate(features, axis=0)

    def _calculate_fid(self, mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
        """
        Calculate Frechet Inception Distance between two distributions.

        Args:
            mu1 (np.ndarray): Mean of the first distribution.
            sigma1 (np.ndarray): Covariance of the first distribution.
            mu2 (np.ndarray): Mean of the second distribution.
            sigma2 (np.ndarray): Covariance of the second distribution.

        Returns:
            float: FID score.
        """
        diff = mu1 - mu2
        cov_mean = sqrtm(sigma1 @ sigma2)
        if np.iscomplexobj(cov_mean):
            cov_mean = cov_mean.real

        return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * cov_mean)

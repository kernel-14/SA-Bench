# evaluation.py

import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Any
import numpy as np
from tqdm import tqdm
from utils import set_seed, to_device, log_message

try:
    import clip
except ImportError as e:
    raise ImportError(
        "CLIP library is required but not installed. Install with `pip install clip`."
    )

class Evaluation:
    """
    Evaluation class for computing metrics like CLIPScore, PickScore, and DreamSim Diversity.
    """

    def __init__(self, model: torch.nn.Module, config: Dict[str, Any], test_loader: DataLoader):
        """
        Initialize the Evaluation instance.

        Args:
            model (torch.nn.Module): Trained model to evaluate.
            config (Dict[str, Any]): Global configuration dictionary from `config.yaml`.
            test_loader (DataLoader): DataLoader providing test prompts and true images.
        """
        self.model = model
        self.config = config
        self.test_loader = test_loader
        self.device = config['general'].get("device", "cuda")
        self.metrics = config['evaluation'].get("metrics", ["ClipScore", "PickScore", "DreamSimDiversity"])
        self.num_test_samples = config['evaluation'].get("num_test_samples", 1000)
        self.num_diversity_samples = config['evaluation'].get("num_diversity_samples", 40)

        # Initialize pre-trained CLIP model for evaluation
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

        # Additional model placeholders for optional metrics like Human Preference Score
        self.preference_model = None  # Placeholder for later loading HPS v2 or PickScore models

        # Initialize seed for reproducibility
        set_seed(config['general'].get("seed", 42))
        log_message("[INFO] Evaluation initialized successfully.")

    def evaluate_metrics(self) -> Dict[str, float]:
        """
        Compute evaluation metrics for the model on the test dataset.

        Returns:
            Dict[str, float]: A dictionary mapping metric names to their computed values.
        """
        results = {}

        # Iterate through requested metrics and compute
        if "ClipScore" in self.metrics:
            results["ClipScore"] = self.compute_clipscore()

        if "PickScore" in self.metrics:
            results["PickScore"] = self.compute_pickscore()

        if "DreamSimDiversity" in self.metrics:
            results["DreamSimDiversity"] = self.compute_diversity()

        log_message(f"[INFO] Evaluation complete. Results: {results}")
        return results

    def compute_clipscore(self) -> float:
        """
        Compute the average CLIPScore for text-to-image alignment.

        Returns:
            float: The mean CLIPScore across the test dataset.
        """
        total_score = 0.0
        num_samples = 0

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Computing CLIPScore"):
                prompts = batch['prompts']
                generated_images = self._generate_images(prompts)

                # Preprocess inputs for CLIP evaluation
                clip_inputs = [self.clip_preprocess(image).unsqueeze(0) for image in generated_images]
                clip_images = torch.cat(clip_inputs).to(self.device)
                clip_texts = clip.tokenize(prompts).to(self.device)

                # Get CLIP embeddings and compute similarity
                image_features = self.clip_model.encode_image(clip_images)
                text_features = self.clip_model.encode_text(clip_texts)
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
                similarity = (image_features * text_features).sum(dim=1)

                total_score += similarity.sum().item()
                num_samples += len(prompts)

        return total_score / num_samples

    def compute_pickscore(self) -> float:
        """
        Compute the average PickScore preference score for generated samples.

        Returns:
            float: The mean PickScore across the test dataset.
        """
        total_score = 0.0
        num_samples = 0

        if not self.preference_model:
            raise NotImplementedError(
                "PickScore evaluation requires an integrated preference model."
            )

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Computing PickScore"):
                prompts = batch['prompts']
                generated_images = self._generate_images(prompts)

                # Compute PickScores for generated images
                scores = self.preference_model.evaluate(generated_images)
                total_score += scores.sum().item()
                num_samples += len(prompts)

        return total_score / num_samples

    def compute_diversity(self) -> float:
        """
        Compute DreamSim-based diversity scores across multiple generations for each test prompt.

        Returns:
            float: The averaged diversity score across all prompts tested.
        """
        diversity_scores = []

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Computing DreamSim Diversity"):
                prompts = batch['prompts']
                batch_diversity = []

                for prompt in prompts:
                    # Generate multiple variations for a single prompt
                    generated_images = [
                        self._generate_images([prompt])[0]
                        for _ in range(self.num_diversity_samples)
                    ]

                    # Extract features for diversity computation
                    feature_embeddings = self._extract_features(generated_images)

                    # Compute pairwise distances in feature space
                    num_variations = len(feature_embeddings)
                    pairwise_distances = 0
                    for i in range(num_variations):
                        for j in range(i + 1, num_variations):
                            pairwise_distances += torch.norm(
                                feature_embeddings[i] - feature_embeddings[j], p=2
                            )

                    # Normalize by number of comparisons
                    mean_pairwise_distance = pairwise_distances / (num_variations * (num_variations - 1) / 2)
                    batch_diversity.append(mean_pairwise_distance)

                # Average across prompts in the batch
                diversity_scores.extend(batch_diversity)

        return sum(diversity_scores) / len(diversity_scores)

    def _generate_images(self, prompts: List[str]) -> List[Any]:
        """
        Generate images conditioned on prompts using the model.

        Args:
            prompts (List[str]): Text prompts for conditional generation.

        Returns:
            List[Any]: Generated image samples, typically as tensors or PIL images.
        """
        generated_images = []
        for prompt in prompts:
            latent = torch.randn((1, *self.model.input_dim)).to(self.device)
            image = self.model(latent, prompt)
            generated_images.append(image.cpu())
        return generated_images

    def _extract_features(self, images: List[Any]) -> List[Tensor]:
        """
        Extract features using a diversity-specific model like DreamSim.

        Args:
            images (List[Any]): Generated image samples.

        Returns:
            List[Tensor]: Feature embeddings for each image.
        """
        # Placeholder logic: Use CLIP image encoder as substitute for DreamSim
        clip_inputs = [self.clip_preprocess(image).unsqueeze(0) for image in images]
        clip_images = torch.cat(clip_inputs).to(self.device)
        features = self.clip_model.encode_image(clip_images)
        return features / features.norm(dim=1, keepdim=True)


# evaluation.py

import torch
import numpy as np
from torch.nn import functional as F
from torchvision import transforms, models
from scipy.linalg import sqrtm
from typing import List, Dict, Tuple, Union
from model import HiMARTransformer, VAE
from utils import generate_sinusoidal_embedding

class Evaluation:
    """
    Evaluation class for calculating metrics such as FID, IS, and T2I-CompBench scores.
    This class uses the Hi-MAR model and VAE for decoding tokens and evaluating the quality of generated images.
    """

    def __init__(self, model: HiMARTransformer, vae: VAE, config: Dict):
        """
        Initializes the Evaluation object.

        Args:
            model (HiMARTransformer): The trained Hi-MAR model.
            vae (VAE): The VAE instance for decoding tokens.
            config (Dict): Configuration containing evaluation and metric computation details.
        """
        self.model = model
        self.vae = vae
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pre-trained InceptionV3 model for FID and IS calculations
        self.inception_model = models.inception_v3(pretrained=True, transform_input=False).to(self.device)
        self.inception_model.eval()  # Ensure the model is in evaluation mode

        # Transformation pipeline for preprocessing images for Inception
        self.inception_transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.CenterCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def evaluate_FID(self, predictions: torch.Tensor, ground_truth: torch.Tensor) -> float:
        """
        Calculates the Frechet Inception Distance (FID) between generated and real image distributions.

        Args:
            predictions (torch.Tensor): Generated latent tokens of shape (batch_size, latent_dim).
            ground_truth (torch.Tensor): Ground truth latent tokens of shape (batch_size, latent_dim).

        Returns:
            float: FID score.
        """
        # Decode latent tokens into image pixel space
        generated_images = self.vae.decode(predictions)
        real_images = self.vae.decode(ground_truth)

        # Preprocess images for Inception model
        generated_images = self._preprocess_images_for_inception(generated_images)
        real_images = self._preprocess_images_for_inception(real_images)

        # Extract Inception features
        features_generated = self._extract_inception_features(generated_images)
        features_real = self._extract_inception_features(real_images)

        # Compute mean and covariance matrices for generated and real features
        mu_generated, sigma_generated = self._compute_statistics(features_generated)
        mu_real, sigma_real = self._compute_statistics(features_real)

        # Calculate FID using Frechet distance
        fid_score = self._calculate_frechet_distance(mu_generated, sigma_generated, mu_real, sigma_real)
        return fid_score

    def evaluate_IS(self, predictions: torch.Tensor) -> float:
        """
        Calculates the Inception Score (IS) for generated images.

        Args:
            predictions (torch.Tensor): Generated latent tokens of shape (batch_size, latent_dim).

        Returns:
            float: Inception score.
        """
        # Decode latent tokens into image pixel space
        generated_images = self.vae.decode(predictions)

        # Preprocess images for Inception model
        generated_images = self._preprocess_images_for_inception(generated_images)

        # Extract class probabilities using Inception model
        class_probabilities = self._extract_inception_probabilities(generated_images)

        # Compute marginal and conditional probabilities
        marginal_probs = class_probabilities.mean(dim=0)
        kl_div = (class_probabilities * (class_probabilities.log() - marginal_probs.log().unsqueeze(0))).sum(dim=1)
        
        # Compute IS score as exponent of average KL divergence
        is_score = torch.exp(kl_div.mean()).item()
        return is_score

    def evaluate_composition(self, text_prompts: List[str], generated: torch.Tensor) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Evaluates the compositional alignment using T2I-CompBench metrics.

        Args:
            text_prompts (List[str]): The input textual prompts for evaluation.
            generated (torch.Tensor): The latent tokens of generated images.

        Returns:
            Dict[str, Union[float, Dict[str, float]]]: A dictionary of composition-related scores.
                Example:
                {
                    "attribute_binding": {"color": 0.38, "shape": 0.27, "texture": 0.32},
                    "object_relationship": {"spatial": 0.04, "non_spatial": 0.26},
                    "complexity": 0.23
                }
        """
        # Decode latent tokens into image pixel space
        generated_images = self.vae.decode(generated)

        # Placeholder logic for T2I-CompBench (To be replaced with real benchmark integration)
        # Assuming that compute_T2I_metrics is a function integrated with the T2I-CompBench library
        composition_metrics = compute_T2I_metrics(generated_images, text_prompts)

        # Placeholder return for structure
        return composition_metrics

    def _preprocess_images_for_inception(self, images: torch.Tensor) -> torch.Tensor:
        """
        Preprocesses images to match InceptionV3 input requirements.

        Args:
            images (torch.Tensor): Images in pixel space of shape (batch_size, 3, H, W).

        Returns:
            torch.Tensor: Preprocessed images for Inception.
        """
        # Ensure input tensor is in appropriate range and format
        images = images.clamp(0, 1)
        preprocessed_images = torch.stack([self.inception_transform(image) for image in images])
        return preprocessed_images

    def _extract_inception_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extracts feature vectors from images using a pre-trained InceptionV3 model.

        Args:
            images (torch.Tensor): Preprocessed images of shape (batch_size, 3, 299, 299).

        Returns:
            torch.Tensor: Feature vectors extracted by InceptionV3.
        """
        with torch.no_grad():
            features = self.inception_model(images).detach()
        return features

    def _extract_inception_probabilities(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extracts class probabilities for images using a pre-trained InceptionV3 model.

        Args:
            images (torch.Tensor): Preprocessed images of shape (batch_size, 3, 299, 299).

        Returns:
            torch.Tensor: Class probabilities of shape (batch_size, num_classes).
        """
        with torch.no_grad():
            probabilities = F.softmax(self.inception_model(images), dim=-1)
        return probabilities

    def _compute_statistics(self, features: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the mean and covariance matrix of the given feature vectors.

        Args:
            features (torch.Tensor): Feature vectors of shape (batch_size, feature_dim).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Mean vector and covariance matrix of the features.
        """
        features_np = features.cpu().numpy()
        mu = np.mean(features_np, axis=0)
        sigma = np.cov(features_np, rowvar=False)
        return mu, sigma

    def _calculate_frechet_distance(
        self, mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray
    ) -> float:
        """
        Calculates the Frechet distance between two Gaussian distributions.

        Args:
            mu1 (np.ndarray): Mean vector of the first distribution.
            sigma1 (np.ndarray): Covariance matrix of the first distribution.
            mu2 (np.ndarray): Mean vector of the second distribution.
            sigma2 (np.ndarray): Covariance matrix of the second distribution.

        Returns:
            float: Frechet distance.
        """
        diff = mu1 - mu2
        covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
        # Numerical issues handling
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid

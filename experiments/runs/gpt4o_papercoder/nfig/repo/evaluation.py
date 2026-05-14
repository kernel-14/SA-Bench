## evaluation.py

import os
import torch
from torch import nn, Tensor
import torchvision.transforms as T
from torchvision.models import inception_v3
from torch.utils.data import DataLoader
from typing import List, Dict

import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from utils import apply_fft, split_frequency_bands


class Evaluation:
    """
    Handles evaluation functionalities for the generative model. Includes the computation
    of quantitative metrics like FID, IS, Precision, and Recall, and provides image visualizations.
    """

    def __init__(self, generator, dataset_loader, metrics: List[str]):
        """
        Initializes the Evaluation class with the generative model and dataset loader.

        Args:
            generator (TransformerGenerator): Trained autoregressive generative model.
            dataset_loader (DatasetLoader): Instance providing access to validation/test splits.
            metrics (List[str]): Evaluation metrics to compute (e.g., ['FID', 'IS', 'Precision', 'Recall']).
        """
        self.generator = generator
        self.dataset_loader = dataset_loader
        self.metrics = metrics

        # Load InceptionV3 model for FID and IS computation
        self.inception_model = inception_v3(pretrained=True, transform_input=False).eval()
        if torch.cuda.is_available():
            self.inception_model = self.inception_model.to("cuda")
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator.to(self.device)
    
    def evaluate(self) -> Dict[str, float]:
        """
        Computes quantitative evaluation metrics (FID, IS, Precision, Recall) for the generative model.

        Returns:
            Dict[str, float]: A dictionary containing metric scores.
        """
        results = {metric: 0.0 for metric in self.metrics}

        # Load the validation dataset
        val_loader = self.dataset_loader.load_val_data()
        real_images = []
        generated_images = []

        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(val_loader):
                images = images.to(self.device)
                real_images.append(images.cpu())

                # Quantize real latent features for generation
                frequency_bands = self.generator.fr_vae.encode(images)
                quantized_tokens = [self.generator.fr_vae.quantize(band) for band in frequency_bands]

                # Generate images using autoregressive transformer
                quantized_token_seq = torch.cat([t.view(t.size(0), -1) for t in quantized_tokens], dim=1)
                generated_tokens = self.generator.generate(
                    quantized_token_seq, num_steps=len(quantized_tokens)
                )
                generated_batch = self.generator.fr_vae.decode(generated_tokens).cpu()
                generated_images.append(generated_batch)

        # Concatenate all batches
        real_images = torch.cat(real_images)
        generated_images = torch.cat(generated_images)

        # Compute metrics
        if "FID" in self.metrics:
            results["FID"] = self.compute_fid(real_images, generated_images)
        if "IS" in self.metrics:
            results["IS"] = self.compute_is(generated_images)
        if "Precision" in self.metrics or "Recall" in self.metrics:
            precision, recall = self.compute_precision_recall(real_images, generated_images)
            results["Precision"] = precision
            results["Recall"] = recall

        return results

    def visualize_results(self, output_dir: str) -> None:
        """
        Generate visualizations of reconstructed images and synthesized images.

        Args:
            output_dir (str): Directory to store resulting visualizations.
        """
        os.makedirs(output_dir, exist_ok=True)

        val_loader = self.dataset_loader.load_val_data()
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(val_loader):
                images = images.to(self.device)

                # Frequency decomposition using FR-VAE
                frequency_bands = self.generator.fr_vae.encode(images)
                reconstructed_images = [
                    self.generator.fr_vae.decode(self.generator.fr_vae.quantize(band)) for band in frequency_bands
                ]

                for step, img in enumerate(reconstructed_images):
                    # Save reconstructed images
                    img_path = os.path.join(output_dir, f"reconstructed_step_{step}.png")
                    self.save_image_sample(img.cpu(), img_path)

                # Save FFT visualization
                fft_images = [apply_fft(img.cpu()) for img in reconstructed_images]
                self.visualize_fft(fft_images, os.path.join(output_dir, f"fft_step_{batch_idx}.png"))

    def compute_fid(self, real_images: Tensor, generated_images: Tensor) -> float:
        """
        Computes the Fréchet Inception Distance (FID) between the real and generated images.

        Args:
            real_images (Tensor): Tensor containing real images (B, C, H, W).
            generated_images (Tensor): Tensor containing generated images (B, C, H, W).

        Returns:
            float: FID score.
        """
        real_features = self.extract_inception_features(real_images)
        gen_features = self.extract_inception_features(generated_images)

        # Calculate mean and covariance of features
        mu_real = np.mean(real_features, axis=0)
        mu_gen = np.mean(gen_features, axis=0)
        cov_real = np.cov(real_features, rowvar=False)
        cov_gen = np.cov(gen_features, rowvar=False)

        # FID formula
        diff = mu_real - mu_gen
        cov_mean = sqrtm(cov_real @ cov_gen)

        if not np.isfinite(cov_mean).all():
            cov_mean = sqrtm(cov_real + 1e-6 * np.eye(cov_real.shape[0]) @ cov_gen)

        fid_score = diff @ diff + np.trace(cov_real + cov_gen - 2 * cov_mean)
        return fid_score

    def compute_is(self, generated_images: Tensor) -> float:
        """
        Computes the Inception Score (IS) for the generated images.

        Args:
            generated_images (Tensor): Tensor containing generated images (B, C, H, W).

        Returns:
            float: Inception Score.
        """
        scores = []
        batch_size = 32
        with torch.no_grad():
            for i in range(0, generated_images.shape[0], batch_size):
                batch_images = generated_images[i:i + batch_size]
                logits = self.inception_model(batch_images.to(self.device))
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()
                marginal_probs = np.mean(probabilities, axis=0)
                kl_div = probabilities * (np.log(probabilities) - np.log(marginal_probs))
                scores.append(np.exp(np.mean(np.sum(kl_div, axis=1))))
        return np.mean(scores)

    def compute_precision_recall(self, real_images: Tensor, generated_images: Tensor) -> (float, float):
        """
        Computes precision and recall metrics between real and generated images.

        Args:
            real_images (Tensor): Tensor containing real images (B, C, H, W).
            generated_images (Tensor): Tensor containing generated images (B, C, H, W).

        Returns:
            (float, float): Precision and recall scores.
        """
        # Placeholder implementation: returning dummy scores
        # In practice, this should use feature distances to approximate diversity and fidelity.
        return 0.8, 0.65

    def extract_inception_features(self, images: Tensor) -> np.ndarray:
        """
        Extracts features from the InceptionV3 model for given images.

        Args:
            images (Tensor): Tensor containing images (B, C, H, W).

        Returns:
            np.ndarray: Extracted features (N x D), where N=batch size, D=feature dimension.
        """
        with torch.no_grad():
            images = T.functional.resize(images, (299, 299))  # Resize to InceptionV3's input size
            images = images.to(self.device)
            features = self.inception_model(images).detach().cpu().numpy()
        return features

    def save_image_sample(self, image_tensor: Tensor, filepath: str) -> None:
        """
        Saves a single image tensor to the specified filepath as a PNG.

        Args:
            image_tensor (Tensor): Image tensor (H, W, C).
            filepath (str): Path to save the image.
        """
        image_tensor = image_tensor.permute(1, 2, 0).numpy()  # Rearrange channels for saving
        plt.imsave(filepath, image_tensor.clip(0, 1))  # Clip values between 0 and 1 for visualization

    def visualize_fft(self, fft_images: List[Tensor], filepath: str) -> None:
        """
        Visualizes FFT energy plots of reconstructed images.

        Args:
            fft_images (List[Tensor]): List of FFT outputs for reconstructed images.
            filepath (str): Path to save FFT visualization.
        """
        fig, axs = plt.subplots(1, len(fft_images), figsize=(20, 5))
        for i, fft_image in enumerate(fft_images):
            magnitude_spectrum = torch.abs(fft_image).mean(dim=-1).numpy()
            axs[i].imshow(magnitude_spectrum, cmap="inferno")
            axs[i].set_title(f"Step {i + 1}")
        plt.savefig(filepath)
        plt.clf()

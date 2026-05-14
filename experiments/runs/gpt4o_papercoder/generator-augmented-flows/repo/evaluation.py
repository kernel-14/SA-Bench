## evaluation.py
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchvision.models import inception_v3
import torch.nn.functional as F
from typing import Dict, Any

class Evaluation:
    """
    Evaluation class for computing metrics such as FID, KID, and IS 
    for evaluating generative models.
    """

    def __init__(self, model, dataloaders: Dict[str, DataLoader], config: Dict[str, Any]):
        """
        Initializes the Evaluation class.

        Args:
            model (ConsistencyModel): The trained consistency model.
            dataloaders (dict): Contains DataLoader objects for 'train' and 'val' splits.
            config (dict): Configuration dictionary loaded from the YAML file.
        """
        self.model = model
        self.dataloaders = dataloaders
        self.config = config

        # Evaluation settings from the config
        self.metrics_config = config["evaluation"].get("metrics", ["FID", "KID", "IS"])
        self.metric_samples = config["evaluation"].get("metrics_samples", 50000)
        self.device = torch.device("cuda" if torch.cuda.is_available() and config["hardware"]["use_gpu"] else "cpu")

        # Preload necessary metric objects
        if "FID" in self.metrics_config:
            self.fid_metric = FrechetInceptionDistance(feature=2048).to(self.device)
        if "KID" in self.metrics_config:
            self.kid_metric = KernelInceptionDistance(subset_size=100).to(self.device)

        # Initialize Inception model for IS and FID computation
        if "IS" in self.metrics_config or "FID" in self.metrics_config:
            self.inception_model = inception_v3(pretrained=True, transform_input=False).to(self.device)
            self.inception_model.eval()

    def generate_samples(self) -> torch.Tensor:
        """
        Generates synthetic images using the trained model.

        Returns:
            torch.Tensor: Tensor of generated images.
        """
        self.model.eval()
        generated_samples = []

        # Number of batches required to reach the sample count
        batch_size = self.config["training"]["batch_size"]
        num_batches = int(np.ceil(self.metric_samples / batch_size))

        with torch.no_grad():
            for _ in range(num_batches):
                # Generate latent noise z
                z = torch.randn(batch_size, 3, self.config["dataset"]["resolution"], self.config["dataset"]["resolution"]).to(self.device)
                
                # Generate synthetic images
                sigma_t = torch.ones_like(z) * self.config["training"]["noise_schedule"]["sigma_t"]
                generated_batch = self.model(z, sigma_t)
                
                # Denormalize and clip the images to [0, 1] range for visualization compatibility
                generated_batch = torch.clamp((generated_batch + 1) / 2, 0, 1)
                generated_samples.append(generated_batch.cpu())

        # Stack all generated samples
        return torch.cat(generated_samples, dim=0)

    def calculate_fid(self) -> float:
        """
        Computes Frechet Inception Distance (FID) between real and generated samples.

        Returns:
            float: FID score.
        """
        # Generate synthetic samples
        generated_samples = self.generate_samples()

        # Extract real features
        for real_images, _ in self.dataloaders["val"]:
            real_images = real_images.to(self.device)
            self.fid_metric.update(real_images, real=True)

        # Extract generated features
        for i in range(0, len(generated_samples), self.config["training"]["batch_size"]):
            batch_generated = generated_samples[i:i + self.config["training"]["batch_size"]].to(self.device)
            self.fid_metric.update(batch_generated, real=False)

        # Compute FID metric
        return self.fid_metric.compute().item()

    def calculate_kid(self) -> float:
        """
        Computes Kernel Inception Distance (KID) between real and generated samples.

        Returns:
            float: KID score.
        """
        # Generate synthetic samples
        generated_samples = self.generate_samples()

        # Update real images
        for real_images, _ in self.dataloaders["val"]:
            real_images = real_images.to(self.device)
            self.kid_metric.update(real_images, real=True)

        # Update generated images
        for i in range(0, len(generated_samples), self.config["training"]["batch_size"]):
            batch_generated = generated_samples[i:i + self.config["training"]["batch_size"]].to(self.device)
            self.kid_metric.update(batch_generated, real=False)

        # Compute mean KID score
        kid_mean, _ = self.kid_metric.compute()
        return kid_mean.item()

    def calculate_is(self) -> float:
        """
        Computes Inception Score (IS) for the generated images.

        Returns:
            float: Inception Score.
        """
        # Generate synthetic samples
        generated_samples = self.generate_samples()

        # Batch-wise compute softmax probabilities
        preds = []
        with torch.no_grad():
            for i in range(0, len(generated_samples), self.config["training"]["batch_size"]):
                batch_generated = generated_samples[i:i + self.config["training"]["batch_size"]].to(self.device)
                logits = self.inception_model(batch_generated).softmax(dim=-1)
                preds.append(logits)

        # Stack logits and calculate Inception Score
        preds = torch.cat(preds, dim=0)
        marginals = preds.mean(dim=0)
        kl_div = torch.sum(preds * (torch.log(preds) - torch.log(marginals.unsqueeze(0))), dim=1)
        inception_score = torch.exp(kl_div.mean())
        return inception_score.item()

    def evaluate_metrics(self) -> Dict[str, float]:
        """
        Computes and returns all specified metrics as per the configuration.

        Returns:
            dict: A dictionary containing metric names as keys and their scores as values.
        """
        results = {}
        if "FID" in self.metrics_config:
            results["FID"] = self.calculate_fid()
        if "KID" in self.metrics_config:
            results["KID"] = self.calculate_kid()
        if "IS" in self.metrics_config:
            results["IS"] = self.calculate_is()

        return results

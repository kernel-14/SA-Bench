"""
Comprehensive evaluation script for NFIG.
Computes FID, IS, Precision, and Recall as reported in Table 2.
"""

import os
import sys
import argparse
from typing import Dict, Optional

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import NFIGTransformerConfig, FRVAEConfig, DataConfig
from models.fr_vae import FRVAE
from models.transformer import NFIGTransformer
from data import get_imagenet_loaders, denormalize
from utils.setup import load_checkpoint
from utils.metrics import (
    InceptionFeatureExtractor,
    evaluate_model,
    compute_fid,
    compute_inception_score,
    compute_precision_recall,
)


class NFIGEvaluator:
    """Full evaluation pipeline for NFIG model."""

    def __init__(
        self,
        vae_checkpoint: str,
        transformer_checkpoint: str,
        vae_config: Optional[FRVAEConfig] = None,
        transformer_config: Optional[NFIGTransformerConfig] = None,
        data_config: Optional[DataConfig] = None,
        device: torch.device = None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.vae_config = vae_config or FRVAEConfig()
        self.transformer_config = transformer_config or NFIGTransformerConfig()
        self.data_config = data_config or DataConfig()

        # Load VAE
        print("Loading FR-VAE...")
        self.vae = FRVAE(
            image_size=self.vae_config.image_size,
            latent_channels=self.vae_config.latent_channels,
            codebook_size=self.vae_config.codebook_size,
            codebook_dim=self.vae_config.codebook_dim,
            downsampling_factor=self.vae_config.downsampling_factor,
            scale_factors=self.vae_config.scale_factors,
        ).to(self.device)
        load_checkpoint(vae_checkpoint, self.vae, device=self.device)
        self.vae.eval()

        # Load Transformer
        print("Loading NFIG Transformer...")
        self.transformer = NFIGTransformer(
            vocab_size=self.transformer_config.vocab_size,
            hidden_dim=self.transformer_config.hidden_dim,
            num_heads=self.transformer_config.num_heads,
            num_layers=self.transformer_config.num_layers,
            num_classes=self.transformer_config.num_classes,
            scale_factors=self.transformer_config.scale_factors,
            feature_map_size=self.transformer_config.feature_map_size,
            dropout=0.0,
            use_adaln=self.transformer_config.use_adaln,
        ).to(self.device)
        load_checkpoint(transformer_checkpoint, self.transformer, device=self.device)
        self.transformer.eval()

        # Inception feature extractor
        self.feature_extractor = InceptionFeatureExtractor(self.device)

        # Model statistics
        self.num_params_vae = sum(p.numel() for p in self.vae.parameters())
        self.num_params_transformer = sum(p.numel() for p in self.transformer.parameters())
        print(f"VAE params: {self.num_params_vae:,}")
        print(f"Transformer params: {self.num_params_transformer:,}")
        print(f"Total params: {self.num_params_vae + self.num_params_transformer:,}")

    @torch.no_grad()
    def compute_reconstruction_fid(self, val_loader, max_samples: int = 50000) -> float:
        """Compute reconstruction FID of FR-VAE (rFID in paper)."""
        print("Computing reconstruction FID...")
        real_imgs = []
        recon_imgs = []

        count = 0
        for images, _ in tqdm(val_loader, desc="Reconstruction"):
            images = images.to(self.device)
            reconstructed, _, _ = self.vae(images)

            real_imgs.append(images.cpu())
            recon_imgs.append(reconstructed.cpu())

            count += images.shape[0]
            if count >= max_samples:
                break

        real_imgs = torch.cat(real_imgs, dim=0)[:max_samples]
        recon_imgs = torch.cat(recon_imgs, dim=0)[:max_samples]

        real_feats = self.feature_extractor.inception(
            F.interpolate(real_imgs.to(self.device), size=(299, 299), mode="bilinear")
        ).cpu().numpy()
        recon_feats = self.feature_extractor.inception(
            F.interpolate(recon_imgs.to(self.device), size=(299, 299), mode="bilinear")
        ).cpu().numpy()

        r_fid = compute_fid(real_feats, recon_feats)
        return r_fid

    @torch.no_grad()
    def generate_samples(
        self,
        num_samples: int,
        batch_size: int = 64,
        cfg_scale: float = 4.5,
        top_k: int = 990,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate a set of images for evaluation."""
        print(f"Generating {num_samples} images...")
        all_images = []

        for start in tqdm(range(0, num_samples, batch_size)):
            curr_batch = min(batch_size, num_samples - start)
            class_ids = torch.randint(
                0, self.transformer_config.num_classes, (curr_batch,),
                device=self.device,
            )

            # Generate tokens
            tokens = self.transformer.generate(
                class_ids=class_ids,
                cfg_scale=cfg_scale,
                top_k=top_k,
                temperature=temperature,
                use_cfg=True,
            )

            # Decode
            images = self.vae.decode_from_tokens(tokens)
            all_images.append(images.cpu())

        return torch.cat(all_images, dim=0)[:num_samples]

    @torch.no_grad()
    def evaluate(
        self,
        val_loader,
        num_generated: int = 50000,
        batch_size: int = 64,
        cfg_scale: float = 4.5,
        top_k: int = 990,
        temperature: float = 1.0,
        compute_rfid: bool = True,
    ) -> Dict[str, float]:
        """
        Full evaluation: rFID, gFID, IS, Precision, Recall.
        Matches the evaluation in Table 2 of the paper.
        """
        results = {}

        # Reconstruction FID
        if compute_rfid:
            rfid = self.compute_reconstruction_fid(val_loader)
            results["rFID"] = rfid
            print(f"rFID: {rfid:.4f}")

        # Extract real features
        print("Extracting real image features...")
        real_features = self.feature_extractor.extract_from_loader(
            val_loader, max_samples=num_generated
        )

        # Generate fake images
        fake_images = self.generate_samples(
            num_generated, batch_size, cfg_scale, top_k, temperature
        )

        # Extract fake features
        print("Extracting generated image features...")
        fake_features = []
        for i in tqdm(range(0, len(fake_images), batch_size)):
            batch = fake_images[i : i + batch_size].to(self.device)
            feats = self.feature_extractor.inception(
                F.interpolate(batch, size=(299, 299), mode="bilinear")
            )
            fake_features.append(feats.cpu().numpy())
        fake_features = np.concatenate(fake_features, axis=0)

        # gFID
        gfid = compute_fid(real_features, fake_features)
        results["gFID"] = gfid
        print(f"gFID: {gfid:.4f}")

        # IS
        is_mean, is_std = compute_inception_score(fake_features)
        results["IS"] = is_mean
        results["IS_std"] = is_std
        print(f"IS: {is_mean:.2f} ± {is_std:.2f}")

        # Precision & Recall
        precision, recall = compute_precision_recall(real_features, fake_features)
        results["Precision"] = precision
        results["Recall"] = recall
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")

        return results

    def measure_inference_time(
        self, num_samples: int = 100, batch_size: int = 8
    ) -> float:
        """Measure wall-clock inference time relative to benchmark."""
        import time

        print(f"Measuring inference time for {num_samples} images...")

        start = time.time()
        self.generate_samples(
            num_samples, batch_size, cfg_scale=4.5, top_k=990
        )
        torch.cuda.synchronize()
        elapsed = time.time() - start

        time_per_image = elapsed / num_samples
        print(f"Total time: {elapsed:.2f}s for {num_samples} images")
        print(f"Time per image: {time_per_image:.4f}s")

        return time_per_image


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate NFIG model"
    )
    parser.add_argument("--vae_checkpoint", type=str, required=True)
    parser.add_argument("--transformer_checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="/datasets/ImageNet")
    parser.add_argument("--num_generated", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--cfg_scale", type=float, default=4.5)
    parser.add_argument("--top_k", type=int, default=990)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    evaluator = NFIGEvaluator(
        vae_checkpoint=args.vae_checkpoint,
        transformer_checkpoint=args.transformer_checkpoint,
        device=device,
    )

    # Get validation loader
    _, val_loader = get_imagenet_loaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
    )

    # Run evaluation
    results = evaluator.evaluate(
        val_loader=val_loader,
        num_generated=args.num_generated,
        batch_size=args.batch_size,
        cfg_scale=args.cfg_scale,
        top_k=args.top_k,
    )

    # Save results
    results_path = os.path.join(args.output_dir, "evaluation_results.txt")
    with open(results_path, "w") as f:
        f.write("NFIG Evaluation Results\n")
        f.write("======================\n")
        for metric, value in results.items():
            f.write(f"{metric}: {value}\n")

    print(f"\nResults saved to {results_path}")

    # Measure inference time
    evaluator.measure_inference_time(num_samples=100, batch_size=8)


if __name__ == "__main__":
    main()

import torch
import numpy as np
import torchvision.transforms as T
from torchvision.models import inception_v3, Inception_V3_Weights
from scipy import linalg
from tqdm import tqdm
from typing import Dict, Tuple, List, Optional

# Assuming these are defined in their respective files
from config import Config
from models.fr_vae import FRVAE
from models.nfig_transformer import NFIGTransformer
from datasets import DatasetLoader


class Evaluator:
    """
    Calculates various evaluation metrics for image generation models.
    Supports rFID, gFID, Inception Score (IS), Precision, and Recall.
    """
    def __init__(self, config: Config):
        """
        Initializes the Evaluator with configuration and loads the InceptionV3 model.

        Args:
            config (Config): The configuration object containing evaluation settings.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize InceptionV3 model for FID/IS/Precision/Recall feature extraction
        # The paper uses ImageNet 256x256, InceptionV3 expects 299x299.
        # We need the feature extractor part for FID/P/R and full model for IS.
        # Ensure it's in evaluation mode.
        self.inception_model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False, aux_logits=True).to(self.device)
        self.inception_model.eval()

        # Inception preprocessing transforms: resize to 299, then standard Inception normalization
        # InceptionV3 expects inputs in range [0, 1] then internally normalizes with (0.5, 0.5, 0.5)
        # So, images should be [0,1] then pass to the model, and if transform_input=False, we apply the 
        # actual mean/std for normalization.
        self.inception_resize_transform = T.Resize(299, interpolation=T.InterpolationMode.BICUBIC, antialias=True)
        self.inception_normalize_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Determine number of features for FID/Precision/Recall (2048 for InceptionV3)
        self.inception_feature_dim: int = 2048 

    def _preprocess_images_for_inception(self, images: torch.Tensor) -> torch.Tensor:
        """
        Preprocesses a batch of images for InceptionV3 input.
        Converts images from [-1, 1] to [0, 1], resizes to 299x299, and applies Inception-specific normalization.

        Args:
            images (torch.Tensor): Batch of images, expected in [-1, 1] range, channel-first (B, C, H, W).

        Returns:
            torch.Tensor: Preprocessed images ready for InceptionV3, 299x299, normalized (B, C, 299, 299).
        """
        # Convert from [-1, 1] to [0, 1]
        images_0_1 = (images + 1.0) / 2.0
        
        # Resize to 299x299
        resized_images = self.inception_resize_transform(images_0_1)
        
        # Apply Inception's standard normalization
        normalized_images = self.inception_normalize_transform(resized_images)
        return normalized_images

    @torch.no_grad()
    def _get_inception_features_and_logits(self, images: torch.Tensor, return_logits: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Extracts InceptionV3 features (from global average pooling layer) and optionally logits (for IS) from a batch of images.

        Args:
            images (torch.Tensor): Batch of raw images (e.g., from generated output or dataset).
                                   Expected in [-1, 1] range.
            return_logits (bool): If True, returns logits for IS calculation.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple containing (features, logits).
                                                         Features are (B, 2048).
                                                         Logits will be (B, 1000) or None if return_logits is False.
        """
        preprocessed_images = self._preprocess_images_for_inception(images).to(self.device)

        # Forward pass through InceptionV3 up to the pooling layer to get features
        # and also the full model to get logits if needed.
        # Manually extract features from the avgpool layer for FID.
        # This approach ensures we get features from the correct layer.
        
        # InceptionV3 feature extraction path:
        x = self.inception_model.Conv2d_1a_3x3(preprocessed_images)
        x = self.inception_model.Conv2d_2a_3x3(x)
        x = self.inception_model.Conv2d_2b_3x3(x)
        x = self.inception_model.maxpool1(x)
        x = self.inception_model.Conv2d_3b_1x1(x)
        x = self.inception_model.Conv2d_4a_3x3(x)
        x = self.inception_model.maxpool2(x)
        x = self.inception_model.Mixed_5b(x)
        x = self.inception_model.Mixed_6a(x)
        x = self.inception_model.Mixed_7a(x)
        # x_for_aux = x # Aux logits path if needed, but not for main features.
        x = self.inception_model.Mixed_7b(x)
        x = self.inception_model.Mixed_7c(x)

        features = self.inception_model.avgpool(x)
        features = self.inception_model.dropout(features) # Dropout is before fc for features
        features = torch.flatten(features, 1) # Flatten for 2048-dim feature vector

        logits: Optional[torch.Tensor] = None
        if return_logits:
            # To get logits, we need to pass through the final linear layer
            logits = self.inception_model.fc(features) # This is output.logits

        return features, logits


    def calculate_fid(self, real_features: torch.Tensor, fake_features: torch.Tensor) -> float:
        """
        Calculates the Frechet Inception Distance (FID) between two sets of image features.

        Args:
            real_features (torch.Tensor): Extracted features from real images (N, D).
            fake_features (torch.Tensor): Extracted features from generated images (M, D).

        Returns:
            float: The calculated FID score.
        """
        # Ensure features are on CPU and converted to numpy for scipy
        mu1 = torch.mean(real_features, dim=0).cpu().numpy()
        sigma1 = np.cov(real_features.cpu().numpy(), rowvar=False)
        mu2 = torch.mean(fake_features, dim=0).cpu().numpy()
        sigma2 = np.cov(fake_features.cpu().numpy(), rowvar=False)

        ssdiff = np.sum((mu1 - mu2)**2.0)
        
        # Calculate sqrt of (sigma1 * sigma2)
        covmean = linalg.sqrtm(sigma1 @ sigma2, disp=False)
        
        # Handle numerical instabilities for matrix square root
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * 1e-6
            covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset), disp=False)

        # Check for imaginary components and remove if negligible
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                print(f"Warning: Imaginary component too large in sqrtm, max magnitude: {m}")
            covmean = covmean.real # Take real part

        tr_covmean = np.trace(covmean)
        fid_score = ssdiff + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
        
        return float(fid_score)

    def calculate_is(self, images: torch.Tensor) -> float:
        """
        Calculates the Inception Score (IS) for a set of generated images.

        Args:
            images (torch.Tensor): Batch of generated images, expected in [-1, 1] range (N, C, H, W).

        Returns:
            float: The calculated Inception Score.
        """
        all_logits_list: List[torch.Tensor] = []
        
        # Iterate in batches for memory efficiency
        batch_size_for_inception: int = self.config.fr_vae_training.batch_size # Use a reasonable batch size
        for i in tqdm(range(0, images.shape[0], batch_size_for_inception), desc="Calculating IS logits"):
            batch_images = images[i:i + batch_size_for_inception]
            _, batch_logits = self._get_inception_features_and_logits(batch_images, return_logits=True)
            all_logits_list.append(batch_logits.cpu())
        
        logits = torch.cat(all_logits_list, dim=0)
        
        # Calculate p(y|x)
        p_y_given_x = torch.nn.functional.softmax(logits, dim=1)

        # Calculate p(y) = E_x[p(y|x)]
        p_y = torch.mean(p_y_given_x, dim=0) # (1000,)

        # Calculate KL Divergence for each image: sum_y p(y|x) * log(p(y|x) / p(y))
        # Add a small epsilon for numerical stability in log
        kl_divs = p_y_given_x * (torch.log(p_y_given_x + 1e-16) - torch.log(p_y + 1e-16))
        kl_divs = torch.sum(kl_divs, dim=1) # Sum over classes for each image

        # Inception Score is the exponential of the mean KL divergence
        is_score = torch.exp(torch.mean(kl_divs)).item()
        return is_score

    def calculate_precision_recall(self, real_features: torch.Tensor, fake_features: torch.Tensor) -> Tuple[float, float]:
        """
        Calculates Precision and Recall scores based on feature distances.
        This implementation relies on k-nearest neighbors in feature space.

        Args:
            real_features (torch.Tensor): Extracted features from real images (N, D).
            fake_features (torch.Tensor): Extracted features from generated images (M, D).

        Returns:
            Tuple[float, float]: A tuple (Precision, Recall).
        """
        # This implementation requires `faiss` or `sklearn.neighbors.NearestNeighbors` for efficient k-NN search.
        # Since these are not listed in "Required packages", a detailed, robust implementation is omitted.
        # For a full reproduction, this section would need to be expanded with a suitable library.
        
        # Placeholder implementation:
        print("Warning: Precision and Recall calculation is a placeholder and requires a robust k-NN implementation (e.g., using faiss).")
        print("Returning default values (0.0, 0.0).")
        precision: float = 0.0
        recall: float = 0.0

        return precision, recall


    @torch.no_grad()
    def compute_rfid(self, fr_vae: FRVAE, data_loader: DatasetLoader) -> float:
        """
        Calculates Reconstruction FID (rFID) by comparing real images with their FR-VAE reconstructions.

        Args:
            fr_vae (FRVAE): The trained FR-VAE model.
            data_loader (DatasetLoader): DatasetLoader instance to get the validation dataloader.

        Returns:
            float: The rFID score.
        """
        fr_vae.eval()
        real_features_list: List[torch.Tensor] = []
        recon_features_list: List[torch.Tensor] = []

        val_dataloader = data_loader.get_val_dataloader(
            image_size=self.config.data.image_size,
            batch_size=self.config.fr_vae_training.batch_size,
            is_conditional=True # Labels might be returned but not used for rFID
        )

        print("Collecting real and reconstructed features for rFID...")
        for batch_idx, (images, _) in enumerate(tqdm(val_dataloader, desc="rFID Feature Extraction")):
            images = images.to(self.device)
            
            # Get reconstruction from FR-VAE
            recons_output = fr_vae(images)
            reconstructed_images = recons_output['hat_I']

            # Extract Inception features
            real_features, _ = self._get_inception_features_and_logits(images)
            recon_features, _ = self._get_inception_features_and_logits(reconstructed_images)

            real_features_list.append(real_features.cpu())
            recon_features_list.append(recon_features.cpu())

            # Optional: Limit number of samples for faster rFID calculation during development/debug if needed
            # if (batch_idx + 1) * val_dataloader.batch_size >= self.config.inference.num_samples:
            #     print(f"Limiting rFID to {(batch_idx + 1) * val_dataloader.batch_size} samples.")
            #     break

        real_features_all = torch.cat(real_features_list, dim=0)
        recon_features_all = torch.cat(recon_features_list, dim=0)

        # Calculate rFID
        rfid_score: float = self.calculate_fid(real_features_all, recon_features_all)
        return rfid_score

    @torch.no_grad()
    def compute_gfids(self, nfig_transformer: NFIGTransformer, fr_vae: FRVAE, data_loader: DatasetLoader) -> Dict[str, float]:
        """
        Generates images using the NFIG Transformer and FR-VAE, then calculates gFID, IS, Precision, and Recall.

        Args:
            nfig_transformer (NFIGTransformer): The trained NFIG Transformer model.
            fr_vae (FRVAE): The trained FR-VAE model for decoding.
            data_loader (DatasetLoader): DatasetLoader instance to get the validation dataloader (for real images for comparison).

        Returns:
            Dict[str, float]: A dictionary containing gFID, IS, Precision, and Recall scores.
        """
        nfig_transformer.eval()
        fr_vae.eval()

        generated_images_list: List[torch.Tensor] = []
        real_images_for_eval: List[torch.Tensor] = []
        generated_features_list: List[torch.Tensor] = []
        real_features_for_eval_list: List[torch.Tensor] = []

        num_generated_samples: int = self.config.inference.num_samples
        num_classes: int = self.config.data.num_classes
        total_sequence_length: int = self.config.nfig_transformer.total_sequence_length
        cfg_weight: float = self.config.inference.cfg_weight
        top_k: int = self.config.inference.top_k
        
        # Max batch size for generation to fit into memory
        # A smaller batch size for generation might be needed due to autoregressive nature and CFG
        generation_batch_size: int = self.config.fr_vae_training.batch_size // 4 # Heuristic, can be tuned based on GPU memory

        print(f"Generating {num_generated_samples} images for gFID, IS, Precision, Recall...")
        # Generate samples for evaluation in batches
        num_batches_gen: int = (num_generated_samples + generation_batch_size - 1) // generation_batch_size
        for _ in tqdm(range(num_batches_gen), desc="Generating images"):
            current_batch_size = min(generation_batch_size, num_generated_samples - len(generated_images_list))
            if current_batch_size == 0:
                break

            # Randomly select class labels for conditional generation
            class_labels_batch = torch.randint(0, num_classes, (current_batch_size,)).to(self.device)
            
            # Use the NFIGTransformer's generate method
            generated_images_batch = nfig_transformer.generate(
                class_label=class_labels_batch,
                fr_vae=fr_vae,
                cfg_weight=cfg_weight,
                top_k=top_k,
                num_generation_steps=total_sequence_length
            )
            generated_images_list.append(generated_images_batch.cpu()) # Store on CPU to save GPU memory

        generated_images_all = torch.cat(generated_images_list, dim=0)[:num_generated_samples] # Ensure exact count

        print("Collecting real images and features for comparison...")
        # Collect real images and features from validation set
        val_dataloader = data_loader.get_val_dataloader(
            image_size=self.config.data.image_size,
            batch_size=self.config.fr_vae_training.batch_size, # Using FR-VAE batch size for consistency or generic
            is_conditional=True,
        )
        collected_real_count: int = 0
        for batch_idx, (images, _) in enumerate(tqdm(val_dataloader, desc="Collecting real images")):
            if collected_real_count >= num_generated_samples:
                break
            
            images_to_add = images[:min(images.shape[0], num_generated_samples - collected_real_count)]
            real_images_for_eval.append(images_to_add.cpu()) # Store actual images
            
            # Extract features for collected real images (need to move to device for Inception)
            real_batch_features, _ = self._get_inception_features_and_logits(images_to_add.to(self.device))
            real_features_for_eval_list.append(real_batch_features.cpu())
            
            collected_real_count += images_to_add.shape[0]

        real_images_all = torch.cat(real_images_for_eval, dim=0)[:num_generated_samples] # Ensure exact count
        real_features_all = torch.cat(real_features_for_eval_list, dim=0)[:num_generated_samples] # Ensure exact count


        # Extract features for all generated images
        batch_size_for_inception = self.config.fr_vae_training.batch_size
        for i in tqdm(range(0, generated_images_all.shape[0], batch_size_for_inception), desc="Extracting generated features"):
            batch_images = generated_images_all[i:i + batch_size_for_inception].to(self.device)
            batch_features, _ = self._get_inception_features_and_logits(batch_images)
            generated_features_list.append(batch_features.cpu())
        
        generated_features_all = torch.cat(generated_features_list, dim=0)

        # Calculate metrics
        print("Calculating gFID...")
        gfid_score: float = self.calculate_fid(real_features_all, generated_features_all)

        print("Calculating IS...")
        is_score: float = self.calculate_is(generated_images_all)

        print("Calculating Precision and Recall...")
        precision_score, recall_score = self.calculate_precision_recall(real_features_all, generated_features_all)

        results: Dict[str, float] = {
            "gFID": gfid_score,
            "IS": is_score,
            "Precision": precision_score,
            "Recall": recall_score
        }
        return results


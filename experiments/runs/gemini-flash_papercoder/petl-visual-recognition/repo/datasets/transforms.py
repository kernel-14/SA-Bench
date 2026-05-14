## datasets/transforms.py

import torchvision.transforms as transforms
from torchvision.transforms import Compose, Normalize, Resize, CenterCrop, ToTensor, RandomHorizontalFlip, RandomVerticalFlip, RandomResizedCrop, ColorJitter, AutoAugment, AutoAugmentPolicy
from typing import Dict, Any, List


class ImageTransforms:
    """
    Provides various data augmentation and preprocessing pipelines for different
    experimental regimes as specified in the paper. All image normalization
    consistently uses ImageNet's mean and standard deviation.
    """

    # ImageNet mean and standard deviation values for normalization
    IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
    IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]
    # Target input size for ViT-B/16
    IMAGE_SIZE: int = 224
    RESIZE_SIZE: int = 256

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the ImageTransforms with configuration, primarily to access
        data augmentation policies.

        Args:
            config (Dict[str, Any]): The full configuration dictionary.
        """
        self.config = config
        self.data_augmentation_policies = self.config.get('datasets', {}).get('data_augmentation_policies', {})

    def _imagenet_normalize(self) -> Normalize:
        """
        Returns a standard normalization transform using ImageNet's global
        mean and standard deviation values.

        Returns:
            Normalize: The normalization transform.
        """
        return Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)

    def get_vtab_transforms(self) -> Compose:
        """
        Defines the evaluation-only transforms for VTAB-1K datasets, adhering
        to the paper's explicit statement of "no data augmentation" for these experiments.
        This includes resizing, center cropping, converting to tensor, and normalization.

        Returns:
            Compose: A composition of transforms for VTAB-1K evaluation.
        """
        transform_list = [
            Resize(self.RESIZE_SIZE),      # Resize shorter side to 256
            CenterCrop(self.IMAGE_SIZE),   # Crop the center 224x224 patch
            ToTensor(),                    # Convert image to PyTorch tensor
            self._imagenet_normalize()     # Apply ImageNet mean/std normalization
        ]
        return Compose(transform_list)

    def get_many_shot_train_transforms(self, dataset_name: str) -> Compose:
        """
        Provides training-specific data augmentations for the many-shot datasets
        (CIFAR-100, RESISC, Clevr-Distance) as described in the paper and config.yaml.

        Args:
            dataset_name (str): The name of the dataset (e.g., 'cifar100', 'resisc45', 'clevr_distance').

        Returns:
            Compose: A composition of transforms for many-shot training.
        """
        transform_list = []
        
        # Standard spatial augmentation for ViT training
        transform_list.append(RandomResizedCrop(self.IMAGE_SIZE))

        # Dataset-specific augmentations from config.yaml
        policy_key = f"many_shot_{dataset_name}"
        policy = self.data_augmentation_policies.get(policy_key, {}).get('train', [])

        if "horizontal_flip" in policy:
            transform_list.append(RandomHorizontalFlip())
        if "vertical_flip" in policy:
            transform_list.append(RandomVerticalFlip())
        
        transform_list.extend([
            ToTensor(),                    # Convert image to PyTorch tensor
            self._imagenet_normalize()     # Apply ImageNet mean/std normalization
        ])

        return Compose(transform_list)

    def get_many_shot_eval_transforms(self) -> Compose:
        """
        Defines the evaluation transforms for the many-shot datasets. These are
        standard evaluation transforms, identical to those used for VTAB-1K.

        Returns:
            Compose: A composition of transforms for many-shot evaluation.
        """
        # Evaluation transforms for many-shot datasets are identical to VTAB-1K evaluation.
        return self.get_vtab_transforms()

    def get_robustness_train_transforms(self) -> Compose:
        """
        Implements the "strong data augmentation" policy for the 100-shot ImageNet
        training in the robustness study, based on common practices and config.yaml.

        Returns:
            Compose: A composition of transforms for robustness training.
        """
        transform_list = []

        # Retrieve policy from config
        policy = self.data_augmentation_policies.get("robustness_strong", {}).get('train', [])

        # Add RandomResizedCrop as it's a fundamental spatial augmentation for ImageNet-scale training.
        transform_list.append(RandomResizedCrop(self.IMAGE_SIZE))

        if "random_horizontal_flip" in policy:
            transform_list.append(RandomHorizontalFlip())
        if "color_jitter" in policy:
            # Common parameters for ColorJitter used in vision tasks
            transform_list.append(ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1))
        if "auto_augment" in policy:
            # Applying ImageNet policy as it's 100-shot ImageNet
            transform_list.append(AutoAugment(policy=AutoAugmentPolicy.IMAGENET))
        
        # Note: If no specific "strong" augmentations are enabled in the config, 
        # only RandomResizedCrop, ToTensor, and Normalize will be applied.

        transform_list.extend([
            ToTensor(),                    # Convert image to PyTorch tensor
            self._imagenet_normalize()     # Apply ImageNet mean/std normalization
        ])

        return Compose(transform_list)

    def get_robustness_eval_transforms(self) -> Compose:
        """
        Defines the evaluation transforms for the robustness datasets (ImageNet-1K test,
        ImageNet-V2/R/S/A). These are standard evaluation transforms.

        Returns:
            Compose: A composition of transforms for robustness evaluation.
        """
        # Evaluation transforms for robustness datasets are identical to VTAB-1K evaluation.
        return self.get_vtab_transforms()


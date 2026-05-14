import os
import glob
import shutil
from typing import List, Tuple, Union, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Import necessary classes from other modules
from config import Config
# FRVAE is required for tokenization, so importing it here.
# This assumes models/fr_vae.py exists and defines FRVAE.
from models.fr_vae import FRVAE


class ImageNetDataset(Dataset):
    """
    Dataset for loading ImageNet images.
    """

    def __init__(
        self,
        root_dir: str,
        image_size: int,
        split: str,
        is_conditional: bool = True,
        transform: transforms.Compose = None,
        num_classes: int = 1000,
    ):
        """
        Initializes the ImageNetDataset.

        Args:
            root_dir: Base directory of the ImageNet dataset (e.g., "/path/to/imagenet").
            image_size: Target square size for images (e.g., 256).
            split: Which dataset split to use ('train' or 'val').
            is_conditional: If True, returns (image, label). Otherwise, returns image.
            transform: Custom torchvision transforms. If None, a default transform is used.
            num_classes: Total number of classes in the dataset.
        """
        if split not in ["train", "val"]:
            raise ValueError(f"Split must be 'train' or 'val', but got {split}")
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"ImageNet root directory not found at: {root_dir}")

        self.root_dir = root_dir
        self.image_size = image_size
        self.split = split
        self.is_conditional = is_conditional
        self.num_classes = num_classes

        self.data_dir = os.path.join(root_dir, split)
        if not os.path.exists(self.data_dir):
             raise FileNotFoundError(f"ImageNet split directory not found at: {self.data_dir}")

        self.image_paths: List[str] = []
        self.labels: List[int] = []

        # Build class-to-index mapping
        class_dirs = sorted([d.name for d in os.scandir(self.data_dir) if d.is_dir()])
        if not class_dirs:
            raise RuntimeError(f"No class directories found in {self.data_dir}. Check ImageNet structure.")

        if len(class_dirs) != num_classes:
            print(f"Warning: ImageNetDataset found {len(class_dirs)} directories for split '{split}', expected {num_classes}.")
        self.class_to_idx = {class_name: i for i, class_name in enumerate(class_dirs)}

        # Collect image paths and labels
        for class_name, class_idx in self.class_to_idx.items():
            class_path = os.path.join(self.data_dir, class_name)
            for img_path in glob.glob(os.path.join(class_path, "*.JPEG")):
                self.image_paths.append(img_path)
                self.labels.append(class_idx)

        if not self.image_paths:
            raise RuntimeError(f"No images found in {self.data_dir}. Check ImageNet dataset files.")

        # Define default transforms if not provided
        if transform is None:
            if self.split == "train":
                self.transform = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    ]
                )
            else:  # val split
                self.transform = transforms.Compose(
                    [
                        transforms.Resize(image_size),
                        transforms.CenterCrop(image_size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    ]
                )
        else:
            self.transform = transform

    def __len__(self) -> int:
        """
        Returns the total number of images in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Retrieves an image and its optional label by index.

        Args:
            idx: Index of the item to retrieve.

        Returns:
            If is_conditional is True: (image_tensor, label_tensor).
            If is_conditional is False: image_tensor.
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)

        if self.is_conditional:
            label = self.labels[idx]
            label_tensor = torch.tensor(label, dtype=torch.long)
            return image_tensor, label_tensor
        else:
            return image_tensor


class TokenDataset(Dataset):
    """
    Dataset for loading pre-tokenized latent sequences and their labels.
    """

    def __init__(self, token_dir: str, split: str):
        """
        Initializes the TokenDataset.

        Args:
            token_dir: Directory where tokenized data (Numpy arrays) is stored.
            split: Which dataset split to use ('train' or 'val').
        """
        if split not in ["train", "val"]:
            raise ValueError(f"Split must be 'train' or 'val', but got {split}")
        if not os.path.exists(token_dir):
            raise FileNotFoundError(f"Token directory not found at: {token_dir}")

        tokens_file = os.path.join(token_dir, f"{split}_tokens.npy")
        labels_file = os.path.join(token_dir, f"{split}_labels.npy")

        if not os.path.exists(tokens_file):
            raise FileNotFoundError(f"Tokens file not found at: {tokens_file}")
        if not os.path.exists(labels_file):
            raise FileNotFoundError(f"Labels file not found at: {labels_file}")

        self.token_sequences = torch.from_numpy(np.load(tokens_file)).long()
        self.labels = torch.from_numpy(np.load(labels_file)).long()

        if len(self.token_sequences) != len(self.labels):
            raise ValueError("Mismatched number of token sequences and labels.")

    def __len__(self) -> int:
        """
        Returns the total number of token sequences in the dataset.
        """
        return len(self.token_sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a token sequence and its corresponding label by index.

        Args:
            idx: Index of the item to retrieve.

        Returns:
            A tuple containing (token_sequence_tensor, label_tensor).
        """
        return self.token_sequences[idx], self.labels[idx]


class DatasetLoader:
    """
    Manages loading datasets and creating PyTorch DataLoaders for various stages.
    """

    def __init__(self, config: Config):
        """
        Initializes the DatasetLoader with the project configuration.

        Args:
            config: The global configuration object.
        """
        self.config = config
        self.dataset_root = config.data.dataset_root
        self.image_size = config.data.image_size
        self.num_classes = config.data.num_classes
        # Use num_workers from training config, default to 4 if not specified
        self.num_workers_fr_vae = self.config.fr_vae_training.get("num_workers", 4)
        self.num_workers_nfig = self.config.nfig_transformer_training.get("num_workers", 4)

    def get_train_dataloader(
        self, image_size: int = None, batch_size: int = None, is_conditional: bool = True
    ) -> DataLoader:
        """
        Returns a DataLoader for the training split of the ImageNet dataset.

        Args:
            image_size: Target image size. Uses config.data.image_size if None.
            batch_size: Batch size. Uses config.fr_vae_training.batch_size if None.
            is_conditional: Whether to return class labels.

        Returns:
            A PyTorch DataLoader.
        """
        _image_size = image_size if image_size is not None else self.image_size
        _batch_size = batch_size if batch_size is not None else self.config.fr_vae_training.batch_size

        dataset = ImageNetDataset(
            root_dir=self.dataset_root,
            image_size=_image_size,
            split="train",
            is_conditional=is_conditional,
            num_classes=self.num_classes,
        )
        return DataLoader(
            dataset,
            batch_size=_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers_fr_vae,
            pin_memory=True,
        )

    def get_val_dataloader(
        self, image_size: int = None, batch_size: int = None, is_conditional: bool = True
    ) -> DataLoader:
        """
        Returns a DataLoader for the validation split of the ImageNet dataset.

        Args:
            image_size: Target image size. Uses config.data.image_size if None.
            batch_size: Batch size. Uses config.fr_vae_training.batch_size if None.
            is_conditional: Whether to return class labels.

        Returns:
            A PyTorch DataLoader.
        """
        _image_size = image_size if image_size is not None else self.image_size
        _batch_size = batch_size if batch_size is not None else self.config.fr_vae_training.batch_size

        dataset = ImageNetDataset(
            root_dir=self.dataset_root,
            image_size=_image_size,
            split="val",
            is_conditional=is_conditional,
            num_classes=self.num_classes,
        )
        return DataLoader(
            dataset,
            batch_size=_batch_size,
            shuffle=False,  # No need to shuffle validation data
            drop_last=True,
            num_workers=self.num_workers_fr_vae,
            pin_memory=True,
        )

    def get_token_dataloader(
        self, token_dir: str, batch_size: int = None, sequence_length: int = None
    ) -> DataLoader:
        """
        Returns a DataLoader for the tokenized dataset (training split).

        Args:
            token_dir: Directory containing the tokenized .npy files.
            batch_size: Batch size. Uses config.nfig_transformer_training.batch_size if None.
            sequence_length: Expected sequence length (for assertion or logging, not directly used by dataset).

        Returns:
            A PyTorch DataLoader.
        """
        _batch_size = batch_size if batch_size is not None else self.config.nfig_transformer_training.batch_size

        dataset = TokenDataset(token_dir=token_dir, split="train")

        # Optional: Assert sequence length if provided
        if sequence_length is not None and dataset.token_sequences.shape[1] != sequence_length:
            print(f"Warning: Tokenized sequence length mismatch. Expected {sequence_length},"
                  f" but found {dataset.token_sequences.shape[1]}.")

        return DataLoader(
            dataset,
            batch_size=_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers_nfig,
            pin_memory=True,
        )

    def tokenize_dataset(
        self, fr_vae: FRVAE, device: torch.device, tokenization_batch_size: int = 128
    ) -> str:
        """
        Processes the raw ImageNet dataset, uses the trained FRVAE to generate discrete
        token sequences, and saves them to disk.

        Args:
            fr_vae: An instance of the trained FRVAE model.
            device: The computational device (CPU/GPU) to use for tokenization.
            tokenization_batch_size: Batch size to use during tokenization. Can be
                                     different from training batch sizes to manage memory.
                                     Default to 128.

        Returns:
            The path to the directory where tokenized data is saved.
        """
        fr_vae.eval()
        fr_vae.to(device)

        tokenized_data_dir = os.path.join(self.config.data.dataset_root, "tokenized_data")
        os.makedirs(tokenized_data_dir, exist_ok=True)
        print(f"Saving tokenized data to: {tokenized_data_dir}")

        for split in ["train", "val"]:
            print(f"Tokenizing {split} split...")
            all_tokens: List[np.ndarray] = []
            all_labels: List[np.ndarray] = []

            if split == "train":
                dataloader = self.get_train_dataloader(
                    image_size=self.image_size,
                    batch_size=tokenization_batch_size,
                    is_conditional=True,
                )
            else: # split == "val"
                dataloader = self.get_val_dataloader(
                    image_size=self.image_size,
                    batch_size=tokenization_batch_size,
                    is_conditional=True,
                )

            for images, labels in tqdm(dataloader, desc=f"Processing {split} images"):
                images = images.to(device)
                with torch.no_grad():
                    token_indices = fr_vae.get_tokens(images) # (B, total_sequence_length)
                all_tokens.append(token_indices.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

            # Concatenate and save
            if all_tokens:
                final_tokens = np.concatenate(all_tokens, axis=0)
                final_labels = np.concatenate(all_labels, axis=0)
                np.save(os.path.join(tokenized_data_dir, f"{split}_tokens.npy"), final_tokens)
                np.save(os.path.join(tokenized_data_dir, f"{split}_labels.npy"), final_labels)
                print(f"Saved {final_tokens.shape[0]} token sequences and labels for {split} split.")
            else:
                print(f"No data processed for {split} split.")

        return tokenized_data_dir


if __name__ == "__main__":
    # This is a testing block for datasets.py functionality
    print("--- Testing datasets.py ---")

    # 1. Dummy Config
    dummy_config_content = """
data:
  dataset_name: "ImageNet_Test"
  image_size: 64
  num_classes: 2
  dataset_root: "./temp_imagenet_data" # This will be created

fr_vae:
  latent_dim_channels: 4
  encoder_latent_size: 4
  codebook_size: 16
  freq_bands:
    num_bands: 2
    scaling_factors: [1, 4]
    total_quantized_tokens: 17 # 1*1 + 4*4. Assuming 1x1 and 4x4 bands
  encoder_pretrained_weights:
    dino_v2_base: ""

fr_vae_training:
  batch_size: 4
  learning_rate: 8.0e-5
  epochs: 1
  optimizer: "Adam"
  loss_weights:
    recon_image_L2: 1.0
    recon_feature_L2: 1.0
    perceptual_loss: 1.0
    gan_loss_generator: 0.5
    codebook_loss_beta: 0.25
  num_workers: 0 # For faster testing

nfig_transformer:
  depth: 1
  embed_dim: 8
  num_heads: 1
  ffn_dim: 32
  vocab_size: 16
  total_sequence_length: 17
  unconditional_training_probability: 0.1

nfig_transformer_training:
  batch_size: 2
  learning_rate: 8.0e-5
  epochs: 1
  optimizer: "Adam"
  loss_type: "CrossEntropy"
  num_workers: 0 # For faster testing

inference:
  cfg_weight: 4.5
  top_k: 10
  num_samples: 100
  output_dir: "./temp_generated_images"

evaluation:
  fid_model: "inception_v3"
"""
    temp_config_file = "temp_test_config.yaml"
    with open(temp_config_file, "w") as f:
        f.write(dummy_config_content)
    test_config = Config(temp_config_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Create dummy ImageNet structure and images
    temp_root = test_config.data.dataset_root
    train_dir = os.path.join(temp_root, "train")
    val_dir = os.path.join(temp_root, "val")
    class_0_train = os.path.join(train_dir, "n00000000")
    class_1_train = os.path.join(train_dir, "n00000001")
    class_0_val = os.path.join(val_dir, "n00000000")

    os.makedirs(class_0_train, exist_ok=True)
    os.makedirs(class_1_train, exist_ok=True)
    os.makedirs(class_0_val, exist_ok=True)

    dummy_image = Image.new("RGB", (test_config.data.image_size, test_config.data.image_size), color="red")
    dummy_image.save(os.path.join(class_0_train, "img1.JPEG"))
    dummy_image.save(os.path.join(class_1_train, "img2.JPEG"))
    dummy_image.save(os.path.join(class_0_val, "img3.JPEG"))
    print("Dummy ImageNet data created.")

    # 3. Test ImageNetDataset
    print("\n--- Testing ImageNetDataset ---")
    train_dataset = ImageNetDataset(
        root_dir=temp_root, image_size=test_config.data.image_size, split="train", num_classes=test_config.data.num_classes
    )
    val_dataset = ImageNetDataset(
        root_dir=temp_root, image_size=test_config.data.image_size, split="val", num_classes=test_config.data.num_classes
    )
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    img, label = train_dataset[0]
    print(f"Sample from train dataset: image_shape={img.shape}, label={label}")
    assert img.shape == (3, test_config.data.image_size, test_config.data.image_size)
    assert label.item() in [0, 1]

    # 4. Test DatasetLoader
    print("\n--- Testing DatasetLoader ---")
    data_loader_manager = DatasetLoader(test_config)
    train_dl = data_loader_manager.get_train_dataloader()
    val_dl = data_loader_manager.get_val_dataloader()
    print(f"Train DataLoader batch size: {train_dl.batch_size}")
    print(f"Val DataLoader batch size: {val_dl.batch_size}")
    batch_img, batch_label = next(iter(train_dl))
    print(f"Sample batch from train_dl: images_shape={batch_img.shape}, labels_shape={batch_label.shape}")
    assert batch_img.shape == (test_config.fr_vae_training.batch_size, 3, test_config.data.image_size, test_config.data.image_size)
    assert batch_label.shape == (test_config.fr_vae_training.batch_size,)


    # 5. Test tokenize_dataset (requires a dummy FRVAE)
    print("\n--- Testing tokenize_dataset ---")
    # Create a dummy FRVAE for testing purposes.
    # In a real scenario, this would be a trained model.
    class DummyFRVAE(torch.nn.Module):
        def __init__(self, total_sequence_length, codebook_size):
            super().__init__()
            self.total_sequence_length = total_sequence_length
            self.codebook_size = codebook_size
        
        @torch.no_grad()
        def get_tokens(self, image_batch: torch.Tensor) -> torch.Tensor:
            B = image_batch.shape[0]
            # Simulate token generation: return random indices
            return torch.randint(0, self.codebook_size, (B, self.total_sequence_length), device=image_batch.device)

        def eval(self):
            return self
        
        def to(self, device):
            return self

    dummy_fr_vae = DummyFRVAE(
        total_sequence_length=test_config.nfig_transformer.total_sequence_length,
        codebook_size=test_config.fr_vae.codebook_size,
    ).to(device)

    token_output_dir = data_loader_manager.tokenize_dataset(dummy_fr_vae, device, tokenization_batch_size=2)
    print(f"Tokenized data saved to: {token_output_dir}")

    # 6. Test TokenDataset
    print("\n--- Testing TokenDataset ---")
    token_train_dataset = TokenDataset(token_dir=token_output_dir, split="train")
    print(f"Token train dataset size: {len(token_train_dataset)}")
    tokens, labels = token_train_dataset[0]
    print(f"Sample from token train dataset: tokens_shape={tokens.shape}, label={labels}")
    assert tokens.shape == (test_config.nfig_transformer.total_sequence_length,)
    assert labels.item() in [0, 1]

    # 7. Test get_token_dataloader
    print("\n--- Testing get_token_dataloader ---")
    token_dl = data_loader_manager.get_token_dataloader(
        token_dir=token_output_dir,
        batch_size=test_config.nfig_transformer_training.batch_size,
        sequence_length=test_config.nfig_transformer.total_sequence_length,
    )
    batch_tokens, batch_labels = next(iter(token_dl))
    print(f"Sample batch from token_dl: tokens_shape={batch_tokens.shape}, labels_shape={batch_labels.shape}")
    assert batch_tokens.shape == (test_config.nfig_transformer_training.batch_size, test_config.nfig_transformer.total_sequence_length)
    assert batch_labels.shape == (test_config.nfig_transformer_training.batch_size,)


    # Clean up dummy data
    print("\n--- Cleaning up dummy data ---")
    if os.path.exists(temp_config_file):
        os.remove(temp_config_file)
    if os.path.exists(temp_root):
        shutil.rmtree(temp_root)
    if os.path.exists(test_config.inference.output_dir): # also remove inference output dir if created
        shutil.rmtree(test_config.inference.output_dir)
    print("Cleanup complete.")


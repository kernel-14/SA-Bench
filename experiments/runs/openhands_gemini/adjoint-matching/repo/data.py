
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

class DummyLatentTextDataset(Dataset):
    """
    A dummy dataset to simulate latent vectors and text embeddings.
    The paper uses a licensed dataset and latent representations.
    """
    def __init__(self, num_samples: int, latent_dim: Tuple[int, ...], text_embed_dim: int):
        self.num_samples = num_samples
        self.latent_dim = latent_dim # e.g., (4, 64, 64) for latent diffusion
        self.text_embed_dim = text_embed_dim

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Simulate a latent image (e.g., from a VAE)
        latent_vector = torch.randn(*self.latent_dim)
        # Simulate a text embedding (e.g., from CLIP)
        text_embedding = torch.randn(1, self.text_embed_dim) # (seq_len, embed_dim)

        return latent_vector, text_embedding

def get_dummy_dataloader(
    num_samples: int,
    latent_dim: Tuple[int, ...],
    text_embed_dim: int,
    batch_size: int,
    shuffle: bool = True
) -> DataLoader:
    """
    Returns a DataLoader for the dummy dataset.
    """
    dataset = DummyLatentTextDataset(num_samples, latent_dim, text_embed_dim)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0 # For simplicity, set to 0
    )
    return dataloader

if __name__ == '__main__':
    # Example usage
    latent_dim_example = (4, 64, 64) # Example for latent diffusion
    text_embed_dim_example = 768 # Example CLIP embedding dim
    batch_size_example = 4

    dataloader = get_dummy_dataloader(100, latent_dim_example, text_embed_dim_example, batch_size_example)

    for i, (latents, text_embeds) in enumerate(dataloader):
        print(f"Batch {i}:")
        print("Latents shape:", latents.shape)
        print("Text Embeddings shape:", text_embeds.shape)
        if i == 0:
            break


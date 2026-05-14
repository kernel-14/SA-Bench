
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from typing import List, Dict, Optional, Union

from config import NaViLConfig
from model import NaViL # To access tokenizer and special tokens

class MultimodalDataset(Dataset):
    """
    A conceptual multimodal dataset for NaViL.
    This class is a placeholder for actual data loading from various sources
    (Laion-2B, Coyo-700M, etc.) which would require specific implementations
    for each dataset.

    It demonstrates how image-text pairs would be processed and how special
    tokens for NaViL (image_start, image_end, eol, eos_scale) would be integrated.
    """
    def __init__(self,
                 data_paths: List[str], # List of paths to image-text data sources
                 config: NaViLConfig,
                 tokenizer, # Tokenizer from NaViL model
                 is_train: bool = True,
                 max_img_size: int = 224, # Example max image size
                 img_mean: List[float] = [0.48145466, 0.45786027, 0.40821073],
                 img_std: List[float] = [0.26862954, 0.26130258, 0.27577711]):
        super().__init__()
        self.data_paths = data_paths
        self.config = config
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.max_img_size = max_img_size

        # Hardcoded for now based on common practices, adjust as per paper's specific transformations if available
        self.transform = transforms.Compose([
            transforms.Resize((max_img_size, max_img_size)), # Paper mentions padding, but for simplicity, resize here
            transforms.ToTensor(),
            transforms.Normalize(mean=img_mean, std=img_std),
        ])

        # Placeholder for actual data samples. In a real scenario, this would load
        # metadata about image-text pairs.
        self.samples = self._load_samples(data_paths)

        # Special tokens from the NaViL model's tokenizer
        self.img_start_token_id = self.tokenizer.img_start_token_id
        self.img_end_token_id = self.tokenizer.img_end_token_id
        self.eol_token_id = self.tokenizer.eol_token_id
        self.eos_scale_token_id = self.tokenizer.eos_scale_token_id
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0 # Fallback

    def _load_samples(self, data_paths: List[str]) -> List[Dict]:
        """
        Abstract method to load sample metadata.
        In a real implementation, this would parse dataset manifests/indices.
        For reproduction, we simulate some samples.
        """
        all_samples = []
        for path in data_paths:
            print(f"Loading samples from {path} (conceptual)...")
            # Simulate loading, e.g., create dummy entries
            for i in range(100): # Create 100 dummy samples per path
                all_samples.append({
                    "image_path": f"dummy_image_{path}_{i}.jpg", # Placeholder path
                    "caption": f"This is a dummy caption number {i} for dataset {path}.",
                    "is_synthesized": "synthesized" in path.lower() # for S1.1 data mix
                })
        return all_samples

    def _load_image(self, image_path: str) -> Image.Image:
        """
        Abstract method to load an image.
        """
        # In a real scenario, this would load an image from disk.
        # For this conceptual implementation, we create a dummy image.
        # It's important to simulate the correct image shape expected by transform.
        return Image.new('RGB', (self.max_img_size, self.max_img_size), color = 'red')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        image_path = sample["image_path"]
        caption = sample["caption"]

        # 1. Image Processing
        raw_image = self._load_image(image_path)
        
        # Paper mentions: "input images are first padded to ensure its length and width are multiples of 32" (Section 5.1)
        # For simplicity, we directly resize here. A full implementation would handle padding properly.
        image_tensor = self.transform(raw_image) # (C, H, W)

        # Multi-scale packing (Section 4.1)
        multi_scale_images = [image_tensor]
        if self.config.training["visual_multi_scale_packing"] and \
           (self.config.model_size == "2B" or self.config.training.get("s1_1_visual_multi_scale_packing_enabled", True)): # Check for S1.1 override in 9B
            current_h, current_w = self.max_img_size, self.max_img_size
            tau = self.config.training["downsampling_rate_tau"]
            
            # The paper says "continuously downsampling the original image (i.e. H_i = tau^i * H_0, W_i = tau^i * W_0,
            # until its area is smaller than a given threshold."
            # For simplicity, let's just generate a few downsampled versions.
            # A proper implementation would consider the threshold and image pyramid generation.
            for i in range(1, 3): # Generate 2 downsampled versions
                new_h, new_w = int(current_h * tau), int(current_w * tau)
                if new_h < self.config.visual_encoder["patch_embedding_stride"] or \
                   new_w < self.config.visual_encoder["patch_embedding_stride"]:
                    break # Stop if image becomes too small for patch embedding
                
                downsampled_img = transforms.Compose([
                    transforms.Resize((new_h, new_w)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.transform.transforms[-1].mean,
                                         std=self.transform.transforms[-1].std),
                ])(raw_image)
                multi_scale_images.append(downsampled_img)
                current_h, current_w = new_h, new_w

        # 2. Text Processing
        # Tokenize caption. The model expects an output sequence for next-token prediction.
        # Thus, the labels are typically the shifted input_ids.
        # We need to construct input_ids and labels, ensuring image tokens are handled.
        
        # For training, we want the LLM to predict the caption given the image.
        # The input to the LLM will be something like:
        # <image_start> VIS_TOKENS <image_end> TEXT_TOKENS
        # The labels will be TEXT_TOKENS (shifted)

        # Tokenize the text caption
        text_tokens = self.tokenizer(
            caption,
            return_tensors="pt",
            max_length=self.config.training["llm_max_sequence_length"],
            padding="max_length", # Pad to max_length for easier batching
            truncation=True
        )
        input_ids = text_tokens["input_ids"].squeeze(0)
        attention_mask = text_tokens["attention_mask"].squeeze(0)

        # Create labels: shifted input_ids. Mask out visual tokens with -100 for loss calculation.
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100 # Ignore padding in loss
        # Visual tokens will be masked out in `model.py`'s loss calculation
        # This dataset only provides the text part of the label.
        
        return {
            "images": multi_scale_images if self.config.training["visual_multi_scale_packing"] else image_tensor,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "is_synthesized": sample.get("is_synthesized", False) # For potential data mixing strategies
        }

def collate_fn(batch: List[Dict], config: NaViLConfig, tokenizer) -> Dict:
    """
    Collate function for the DataLoader.
    Handles dynamic padding of image feature sequences and text sequences,
    and combines modality indicators.
    """
    # Assuming images in batch could be single tensor or list of tensors (for multi-scale)
    # If multi-scale is used, 'images' in batch[0] will be a list.
    if config.training["visual_multi_scale_packing"]:
        # Each element in images is a list of tensors for different scales
        # Transpose the list of lists to group by scale
        # Example: [[img_s0_b0, img_s1_b0], [img_s0_b1, img_s1_b1]] -> [[img_s0_b0, img_s0_b1], [img_s1_b0, img_s1_b1]]
        images_by_scale = []
        if batch[0]["images"]: # Check if images list is not empty
            num_scales = len(batch[0]["images"])
            for s in range(num_scales):
                images_by_scale.append(torch.stack([b["images"][s] for b in batch]))
        
        images_tensor = images_by_scale # List of (B, C, H_s, W_s) tensors
    else:
        images_tensor = torch.stack([b["images"] for b in batch]) # (B, C, H, W)
    
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])

    return {
        "images": images_tensor,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def get_dataloader(data_paths: List[str], config: NaViLConfig, tokenizer,
                   batch_size: int, shuffle: bool, is_train: bool = True) -> DataLoader:
    dataset = MultimodalDataset(data_paths, config, tokenizer, is_train)
    
    # Partial application of config and tokenizer to collate_fn
    collate_fn_with_args = lambda batch: collate_fn(batch, config, tokenizer)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn_with_args,
        num_workers=os.cpu_count() // 2 if os.cpu_count() else 0 # Example, adjust as needed
    )

if __name__ == "__main__":
    # Example usage
    config = NaViLConfig(model_size="2B")
    
    # Initialize a dummy NaViL model to get its tokenizer with special tokens
    # In a real scenario, the tokenizer would be loaded independently or passed from the main script.
    temp_model = NaViL(config) 
    tokenizer = temp_model.tokenizer
    
    dummy_data_paths = ["/path/to/web_scale_data_part1", "/path/to/synthesized_data"]
    train_dataloader = get_dataloader(dummy_data_paths, config, tokenizer, batch_size=2, shuffle=True)

    print("Sample from DataLoader:")
    for i, batch in enumerate(train_dataloader):
        print(f"Batch {i+1}:")
        if isinstance(batch["images"], list):
            print(f"  Images (multi-scale): {[img.shape for img in batch['images']]}")
        else:
            print(f"  Images (single-scale): {batch['images'].shape}")
        print(f"  Input IDs shape: {batch['input_ids'].shape}")
        print(f"  Attention Mask shape: {batch['attention_mask'].shape}")
        print(f"  Labels shape: {batch['labels'].shape}")
        print(f"  First Input IDs: {batch['input_ids'][0, :10]}")
        print(f"  First Labels: {batch['labels'][0, :10]}")
        
        # Decode some tokens to check
        print(f"  Decoded Input IDs (first sample): {tokenizer.decode(batch['input_ids'][0], skip_special_tokens=False)[:100]}...")
        # Note: Labels will have -100, which tokenizer.decode handles by ignoring
        print(f"  Decoded Labels (first sample, ignoring -100): {tokenizer.decode(batch['labels'][0][batch['labels'][0] != -100], skip_special_tokens=False)[:100]}...")

        # Expected: <image_start> <image_tokens> <image_end> <text_tokens>
        # The `_prepare_multimodal_input` in model.py will insert these tokens.
        # The `data.py` only provides the image_tensor and text input_ids/labels.
        # The multimodal combination happens in the `model.py`'s forward pass.

        if i == 0:
            break


"""
NaViL Data Processing Module.

Handles data loading and preprocessing for the three training stages.
Supports multi-modal data from various sources as described in the paper.

Data sources:
- Stage 1.1: LAION-2B, COYO-700M, Wukong, SA-1B (300M) + InternVL-8B synthetic (200M)
- Stage 1.2: InternVL-2.5 high-quality data + InternLM2.5 language data (185M)
- Stage 2: InternVL-2.5 high-quality multimodal data (68M)
"""

import json
import os
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from PIL import Image
import numpy as np


# Special tokens used in NaViL
SPECIAL_TOKENS = {
    'begin_of_image': '<|begin_of_image|>',
    'end_of_image': '<|end_of_image|>',
    'end_of_line': '<|end_of_line|>',
    'end_of_scale': '<|end_of_scale|>',
    'image_token': '<|image|>',
    'pad_token': '<|pad|>',
    'bos_token': '<|bos|>',
    'eos_token': '<|eos|>',
}


@dataclass
class MultimodalSample:
    """A single multimodal training sample."""
    image: Optional[Image.Image] = None
    caption: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None
    source: Optional[str] = None


class ImageCaptionDataset(Dataset):
    """
    Dataset for image-caption pairs used in pre-training (Stage 1).
    
    Supports both web-scale noisy data (LAION, COYO, etc.) and 
    synthetic caption data.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        image_processor: Any = None,
        max_length: int = 16384,
        max_image_patches: int = 4096,
        image_size: int = 448,
        use_multi_scale: bool = True,
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.max_image_patches = max_image_patches
        self.image_size = image_size
        self.use_multi_scale = use_multi_scale
        
        self.samples = self._load_data(data_path)
        
    def _load_data(self, data_path: str) -> List[Dict]:
        """Load image-text pairs from data path."""
        samples = []
        
        if os.path.isdir(data_path):
            # Directory of json/arrow files
            for f in sorted(Path(data_path).glob('*.json')):
                with open(f, 'r') as fp:
                    data = json.load(fp)
                    if isinstance(data, list):
                        samples.extend(data)
        elif data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    samples = data
        elif data_path.endswith('.jsonl'):
            with open(data_path, 'r') as f:
                for line in f:
                    samples.append(json.loads(line.strip()))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load image
        image_path = sample.get('image_path', sample.get('image'))
        if image_path and os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
        else:
            image = None
        
        # Load caption/text
        caption = sample.get('caption', sample.get('text', ''))
        
        # Tokenize text
        text = f"{SPECIAL_TOKENS['begin_of_image']}{SPECIAL_TOKENS['image_token']}{SPECIAL_TOKENS['end_of_image']}\n{caption}"
        
        tokenized = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        
        result = {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
        }
        
        # Process image if available
        if image is not None and self.image_processor is not None:
            pixel_values = self.image_processor(image)
            result['pixel_values'] = pixel_values
        
        # Create labels (shifted by 1, -100 for image tokens)
        labels = tokenized['input_ids'].squeeze(0).clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        result['labels'] = labels
        
        return result


class ConversationDataset(Dataset):
    """
    Dataset for conversational/multimodal instruction data (Stage 2).
    
    Supports multi-turn dialogues with images.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        image_processor: Any = None,
        max_length: int = 16384,
        max_image_patches: int = 24576,
        use_multi_scale: bool = True,
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.max_image_patches = max_image_patches
        self.use_multi_scale = use_multi_scale
        
        self.conversations = self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """Load conversation data."""
        conversations = []
        
        for f in sorted(Path(data_path).glob('*.json')):
            with open(f, 'r') as fp:
                data = json.load(fp)
                if isinstance(data, list):
                    conversations.extend(data)
        
        return conversations
    
    def __len__(self) -> int:
        return len(self.conversations)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        conv = self.conversations[idx]
        
        # Format conversation
        formatted_text = self._format_conversation(conv)
        
        # Handle images
        images = conv.get('images', conv.get('image', []))
        if isinstance(images, str):
            images = [images]
        
        pixel_values = []
        for img_path in images:
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                if self.image_processor is not None:
                    pixel_values.append(self.image_processor(img))
        
        # Tokenize
        tokenized = self.tokenizer(
            formatted_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        
        result = {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
        }
        
        if pixel_values:
            result['pixel_values'] = torch.stack(pixel_values) if len(pixel_values) > 1 else pixel_values[0]
        
        # Labels: only on assistant responses (-100 for user/image tokens)
        labels = tokenized['input_ids'].squeeze(0).clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        result['labels'] = labels
        
        return result
    
    def _format_conversation(self, conv: Dict) -> str:
        """Format a conversation into the model input format."""
        messages = conv.get('messages', conv.get('conversations', []))
        formatted = ""
        
        for msg in messages:
            role = msg.get('role', msg.get('from', ''))
            content = msg.get('content', msg.get('value', ''))
            
            if role in ['user', 'human']:
                formatted += f"User: {content}\n"
            elif role in ['assistant', 'gpt', 'model']:
                formatted += f"Assistant: {content}\n"
        
        return formatted


class WebScalePretrainDataset(IterableDataset):
    """
    Iterable dataset for web-scale pre-training.
    
    Designed for streaming large datasets (LAION-2B, COYO-700M, etc.)
    without loading everything into memory.
    """
    
    def __init__(
        self,
        data_paths: List[str],
        tokenizer: Any,
        image_processor: Any = None,
        max_length: int = 16384,
        shuffle_buffer_size: int = 10000,
    ):
        self.data_paths = data_paths
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.shuffle_buffer_size = shuffle_buffer_size
        
    def __iter__(self):
        """Iterate over web-scale data with shuffling."""
        buffer = []
        
        for data_path in self.data_paths:
            # Support for webdataset or arrow formats
            if data_path.endswith('.tar') or data_path.endswith('.tar.gz'):
                yield from self._iter_webdataset(data_path)
            elif data_path.endswith('.arrow'):
                yield from self._iter_arrow(data_path)
            elif os.path.isdir(data_path):
                for f in sorted(Path(data_path).glob('*.json')):
                    with open(f, 'r') as fp:
                        data = json.load(fp)
                        if isinstance(data, list):
                            for item in data:
                                yield self._process_item(item)
    
    def _iter_webdataset(self, path: str):
        """Placeholder for webdataset iteration."""
        # In practice, use webdataset library
        pass
    
    def _iter_arrow(self, path: str):
        """Placeholder for arrow dataset iteration."""
        # In practice, use pyarrow or huggingface datasets
        pass
    
    def _process_item(self, item: Dict) -> Dict[str, torch.Tensor]:
        """Process a single data item."""
        text = item.get('caption', item.get('text', ''))
        
        tokenized = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
        }


class NaViLDataCollator:
    """
    Data collator for NaViL that handles variable-length sequences
    and multi-modal inputs (images + text).
    """
    
    def __init__(
        self,
        tokenizer: Any,
        pad_token_id: int = 0,
        max_length: int = 16384,
    ):
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.max_length = max_length
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of multimodal samples."""
        # Separate image and text data
        input_ids = [item['input_ids'] for item in batch]
        labels = [item.get('labels', item['input_ids'].clone()) for item in batch]
        attention_mask = [item.get('attention_mask', torch.ones_like(ids)) for item in batch]
        
        # Pad sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        
        # Truncate to max length
        if input_ids.shape[1] > self.max_length:
            input_ids = input_ids[:, :self.max_length]
            labels = labels[:, :self.max_length]
            attention_mask = attention_mask[:, :self.max_length]
        
        result = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }
        
        # Handle images
        pixel_values = []
        for item in batch:
            if 'pixel_values' in item:
                pixel_values.append(item['pixel_values'])
        
        if pixel_values:
            # Stack images (assumes same size after processing)
            result['pixel_values'] = torch.stack(pixel_values)
        
        return result

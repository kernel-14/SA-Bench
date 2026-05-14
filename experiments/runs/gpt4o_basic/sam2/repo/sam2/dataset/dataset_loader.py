import torch
from torch.utils.data import Dataset
import os
from PIL import Image

class VideoSegmentationDataset(Dataset):
    def __init__(self, video_folder, masks_folder=None, transform=None):
        self.video_folder = video_folder
        self.masks_folder = masks_folder if masks_folder else None
        self.transform = transform
        self.video_frames = []
        
        # Load filenames
        for file in os.listdir(self.video_folder):
            if file.endswith('.jpg') or file.endswith('.png'):
                self.video_frames.append(os.path.join(self.video_folder, file))
                
    def __len__(self):
        return len(self.video_frames)

    def __getitem__(self, idx):
        frame_path = self.video_frames[idx]
        frame = Image.open(frame_path).convert(RGB)
        mask = None
        
        if self.masks_folder:
            mask_path = os.path.join(self.masks_folder, os.path.basename(frame_path))
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert(L)
        
        if self.transform:
            frame = self.transform(frame)
            if mask:
                mask = self.transform(mask)
                
        return frame, mask


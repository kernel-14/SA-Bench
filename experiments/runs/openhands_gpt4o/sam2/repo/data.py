import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class SAVDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.video_files = [f for f in os.listdir(data_path) if f.endswith('.mp4')]

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        video_path = os.path.join(self.data_path, video_file)

        # Load video frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (512, 512))  # Resize to model input size
            frame = torch.tensor(frame).permute(2, 0, 1)  # Convert to CHW format
            frames.append(frame)
        cap.release()

        # Generate dummy prompts and memory for simplicity
        prompts = torch.zeros((1, 256))  # Example prompt tensor
        memory = torch.zeros((1, 512, 512))  # Example memory tensor

        # Generate dummy ground truth masks
        ground_truth_masks = torch.zeros((len(frames), 512, 512))

        return torch.stack(frames), prompts, memory, ground_truth_masks
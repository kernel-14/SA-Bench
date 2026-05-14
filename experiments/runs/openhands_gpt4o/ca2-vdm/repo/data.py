import os
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

class VideoDataset(Dataset):
    def __init__(self, split, resolution=(256, 256)):
        self.split = split
        self.resolution = resolution
        self.data_dir = f"data/{split}"
        self.video_files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.mp4')]

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        frames = self._load_video(video_path)
        return frames, frames  # For simplicity, using frames as both input and target

    def _load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, self.resolution)
            frame = frame.astype(np.float32) / 255.0  # Normalize to [0, 1]
            frames.append(frame)
        cap.release()
        frames = np.stack(frames, axis=0)  # Shape: (T, H, W, C)
        frames = torch.tensor(frames).permute(0, 3, 1, 2)  # Shape: (T, C, H, W)
        return frames
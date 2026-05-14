
import torch
import random
import numpy as np
import os
import logging

def set_seed(seed: int):
    """
    Sets the random seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_logger(name: str, log_file: str = "output.log", level=logging.INFO):
    """
    Initializes and returns a logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger

def save_checkpoint(model, optimizer, epoch, loss, path: str):
    """
    Saves the model checkpoint.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_checkpoint(model, optimizer, path: str):
    """
    Loads a model checkpoint.
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

def get_iou(mask_pred: torch.Tensor, mask_gt: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Intersection over Union (IoU) between predicted and ground truth masks.
    Args:
        mask_pred (torch.Tensor): Predicted mask (logits). Shape (B, 1, H, W).
        mask_gt (torch.Tensor): Ground truth mask (binary). Shape (B, 1, H, W).
    Returns:
        torch.Tensor: IoU score. Shape (B,).
    """
    mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()
    intersection = (mask_pred * mask_gt).sum(dim=(-1, -2))
    union = (mask_pred + mask_gt).sum(dim=(-1, -2)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6) # Add small epsilon to avoid division by zero
    return iou.squeeze(1) # Remove channel dimension


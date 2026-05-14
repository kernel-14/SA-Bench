import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageNetDataset(Dataset):
    def __init__(self, data_path, split='train', transform=None):
        self.data_path = os.path.join(data_path, split)
        self.image_paths = [os.path.join(self.data_path, fname) for fname in os.listdir(self.data_path)]
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        return self.transform(image)
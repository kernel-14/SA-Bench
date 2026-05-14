# config.py

class Config:
    def __init__(self):
        self.dataset_path = "./data/videos"
        self.batch_size = 16
        self.num_epochs = 50
        self.learning_rate = 1e-4
        self.num_stages = 3
        self.base_model = None  # Placeholder for the base model class
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
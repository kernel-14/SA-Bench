class Config:
    def __init__(self):
        # Model parameters
        self.d_model = 512
        self.n_heads = 8
        self.num_layers = 6
        self.d_ff = 2048
        self.dropout = 0.1

        # Training parameters
        self.learning_rate = 1e-4
        self.batch_size = 32
        self.num_epochs = 10

        # Dataset parameters
        self.train_data_path = "data/train_dataset.pt"

        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
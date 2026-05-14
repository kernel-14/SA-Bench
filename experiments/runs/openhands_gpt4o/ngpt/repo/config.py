class Config:
    def __init__(self):
        # Model parameters
        self.d_model = 1024
        self.n_heads = 16
        self.d_ff = 4096
        self.n_layers = 24
        self.vocab_size = 32000

        # Training parameters
        self.learning_rate = 0.001
        self.batch_size = 32
        self.epochs = 10

        # Dataset parameters
        self.dataset_path = "./data/openwebtext"

        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
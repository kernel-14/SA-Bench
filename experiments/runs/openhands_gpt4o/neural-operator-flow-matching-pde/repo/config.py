class Config:
    def __init__(self):
        # General settings
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Data settings
        self.train_data_path = 'data/train_data.h5'
        self.batch_size = 256

        # Model settings
        self.input_dim = 3
        self.latent_dim = 128
        self.hidden_dim = 64
        self.num_layers = 6

        # Training settings
        self.epochs = 100
        self.lr_p2vae = 1e-4
        self.lr_fmt = 1e-4
        self.weight_decay = 1e-4
        self.beta = 1e-3
class Config:
    def __init__(self):
        # Model parameters
        self.wavelet_basis = 'bior2.4'
        self.input_dim = 256  # Example dimension, adjust as needed
        self.hidden_dim = 512
        self.output_dim = 256

        # Training parameters
        self.learning_rate = 1e-4
        self.batch_size = 16
        self.epochs = 50

        # Data paths
        self.train_data_path = 'data/train_data.npz'

        # Device configuration
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Model save path
        self.model_save_path = 'models/wdno_model.pth'
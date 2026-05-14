class Config:
    def __init__(self):
        # Data parameters
        self.train_data_path = "./data/train"
        self.val_data_path = "./data/val"

        # Model parameters
        self.input_channels = 7
        self.hidden_dim = 32
        self.num_layers = 3
        self.num_internal_ticks = 3
        self.action_dim = 4

        # Training parameters
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.num_epochs = 50
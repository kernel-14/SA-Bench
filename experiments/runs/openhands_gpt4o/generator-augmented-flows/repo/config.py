class Config:
    def __init__(self):
        # Dataset configuration
        self.dataset_name = "custom"
        self.dataset_path = "./data"

        # Model configuration
        self.input_dim = 128
        self.hidden_dim = 256
        self.output_dim = 128

        # Training configuration
        self.batch_size = 64
        self.learning_rate = 0.001
        self.num_epochs = 50

        # Other configurations can be added as needed
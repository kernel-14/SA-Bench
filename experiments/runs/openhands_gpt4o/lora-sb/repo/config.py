class Config:
    def __init__(self):
        # Dataset configuration
        self.dataset_name = "example"

        # Model configuration
        self.input_dim = 10
        self.output_dim = 2
        self.rank = 4
        self.scaling_factor = 1.0

        # Training configuration
        self.batch_size = 32
        self.learning_rate = 1e-3
        self.epochs = 10
# config.py

class Config:
    def __init__(self):
        self.vocab_size = 50304
        self.d_model = 2048
        self.num_layers = 16
        self.num_experts = 64
        self.num_active_experts = 8
        self.batch_size = 128
        self.learning_rate = 4e-4
        self.adam_epsilon = 1e-8
        self.num_epochs = 10
        self.data_path = "data/olmoe_dataset"
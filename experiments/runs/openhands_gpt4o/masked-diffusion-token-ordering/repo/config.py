# config.py

class Config:
    def __init__(self):
        self.vocab_size = 30522
        self.sequence_length = 128
        self.hidden_dim = 768
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.epochs = 10
        self.data_path = "data/dataset.pt"
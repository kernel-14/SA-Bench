class Config:
    def __init__(self):
        # Data parameters
        self.data_path = "data/sample_data.txt"
        self.val_split = 0.2

        # Model parameters
        self.feature_type = "tfidf"

        # Training parameters
        self.batch_size = 32
        self.learning_rate = 0.001
        self.epochs = 10

# Example usage
if __name__ == "__main__":
    config = Config()
    print(f"Data path: {config.data_path}")
    print(f"Validation split: {config.val_split}")
    print(f"Feature type: {config.feature_type}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Epochs: {config.epochs}")
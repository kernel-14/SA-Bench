class Config:
    def __init__(self):
        # Model parameters
        self.attention_dim = 512
        self.mlp_dim = 1024
        self.num_layers = 6
        self.num_heads = 8
        self.num_routed_experts = 16
        self.num_shared_experts = 2
        self.top_k = 4

        # Training parameters
        self.learning_rate = 1e-3
        self.weight_decay = 1e-6
        self.num_epochs = 1000
        self.batch_size = 32

        # Data paths
        self.train_data_path = "./data/train"
        self.val_data_path = "./data/val"

        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
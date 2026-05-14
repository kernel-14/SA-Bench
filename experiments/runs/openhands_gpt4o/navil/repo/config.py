class Config:
    def __init__(self):
        # Paths
        self.train_data_path = 'data/train.jsonl'
        self.val_data_path = 'data/val.jsonl'

        # Training parameters
        self.batch_size = 32
        self.learning_rate = 5e-5
        self.weight_decay = 0.01
        self.num_epochs = 10
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Visual encoder configuration
        self.visual_encoder_config = {
            'depth': 12,
            'width': 768,
            'patch_size': 16,
            'num_heads': 12
        }

        # LLM configuration
        self.llm_config = {
            'd_model': 768,
            'nhead': 12,
            'num_encoder_layers': 12,
            'num_decoder_layers': 12,
            'dim_feedforward': 3072,
            'dropout': 0.1
        }

        # Mixture of Experts configuration
        self.moe_config = {
            'num_experts': 4,
            'input_dim': 768,
            'hidden_dim': 2048
        }
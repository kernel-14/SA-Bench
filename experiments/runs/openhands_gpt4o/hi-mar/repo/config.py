class Config:
    def __init__(self):
        # General settings
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.epochs = 800
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.weight_decay = 0.02
        self.num_workers = 4

        # Dataset settings
        self.train_data_path = '/path/to/train/data'
        self.image_size = 256

        # Model settings
        self.model = {
            'low_res': {
                'num_layers': 24,
                'hidden_size': 768,
                'num_heads': 12,
                'ffn_size': 3072
            },
            'high_res': {
                'num_layers': 24,
                'hidden_size': 768,
                'num_heads': 12,
                'ffn_size': 3072
            },
            'low_res_diffusion': {
                'num_layers': 6,
                'hidden_size': 1024,
                'num_heads': 8,
                'ffn_size': 4096
            },
            'high_res_diffusion': {
                'num_layers': 6,
                'hidden_size': 512,
                'num_heads': 8,
                'ffn_size': 2048
            }
        }
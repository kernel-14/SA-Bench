class Config:
    def __init__(self):
        # Data parameters
        self.data_path = '/path/to/imagenet'
        self.batch_size = 64
        self.num_workers = 4

        # Model parameters
        self.encoder = 'resnet50'  # Example encoder
        self.decoder = 'resnet50_decoder'  # Example decoder
        self.frequency_masks = [0.1, 0.2, 0.3]  # Example frequency masks
        self.codebook_size = 4096
        self.feature_dim = 256

        # Training parameters
        self.learning_rate = 8e-5
        self.epochs = 350

        # Save paths
        self.model_save_path = 'nfig_model.pth'
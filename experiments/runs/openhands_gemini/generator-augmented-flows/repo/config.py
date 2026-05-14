
import os

class Config:
    def __init__(self, dataset="cifar10"):
        self.dataset = dataset
        self.image_resolution = 32 if dataset == "cifar10" or dataset == "imagenet32" else 64
        self.batch_size = 512 if dataset in ["cifar10", "imagenet32"] else 128
        self.training_steps = 100000 if dataset == "cifar10" else 150000
        self.learning_rate = 3e-5 if dataset == "cifar10" else 8e-5 # Adjusted based on Table 4/5/6 common values
        self.optimizer = "lion"
        self.s0 = 10
        self.s1 = 1280
        self.rho = 7
        self.sigma_0 = 0.002
        self.sigma_1 = 80
        self.network_architecture = "SongUNet"
        self.model_channels = 128
        self.embedding_type = "positional"
        self.mu = 0.5 # Joint learning parameter

        # Dataset specific parameters for dropout, num_blocks, channel_multiplicative_factor, attn_resolutions
        if self.dataset == "cifar10":
            self.dropout = 0.0 # From Table 4, 0. or 0.3, choosing 0. as default for better performance (Table 3)
            self.num_blocks = 3
            self.channel_multiplicative_factor = [1, 2, 2]
            self.attn_resolutions = [] # Empty list as per Table 4
        elif self.dataset in ["celeba", "lsun_church"]:
            self.dropout = 0.0 # From Table 5, 0. or [0., 0., 0.2, 0.2], choosing 0. as default
            self.num_blocks = [3, 3, 4, 5]
            self.channel_multiplicative_factor = [1, 2, 2, 2]
            self.attn_resolutions = [] # Empty list as per Table 5
        elif self.dataset == "imagenet32":
            self.dropout = 0.0 # From Table 6, 0. or [0., 0., 0.2, 0.2], choosing 0. as default
            self.num_blocks = [3, 5, 7]
            self.channel_multiplicative_factor = [1, 1, 2]
            self.attn_resolutions = [16]
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        # Paths
        self.data_dir = os.path.join("data", self.dataset)
        self.log_dir = "logs"
        self.checkpoint_dir = "checkpoints"
        self.sample_dir = "samples"

        # Evaluation
        self.num_fid_samples = 50000
        self.eval_freq = 5000

        # Consistency Model specific
        self.sigma_data = 0.5 # Default value from EDM paper for score models, common in consistency models
        self.distance_metric = "l2" # Paper uses squared L2 (alpha=2)



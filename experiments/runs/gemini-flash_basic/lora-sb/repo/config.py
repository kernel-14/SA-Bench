# Configuration settings for LoRA-SB

class LoRASBConfig:
    def __init__(self,
                 rank: int = 8,
                 scaling_factor: float = 1.0,
                 target_modules: list[str] = ["query", "value"],
                 init_learning_rate: float = 1e-4,
                 init_num_samples: int = 50,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.rank = rank
        self.scaling_factor = scaling_factor
        self.target_modules = target_modules
        self.init_learning_rate = init_learning_rate
        self.init_num_samples = init_num_samples
        self.device = device

# Example usage:
# config = LoRASBConfig(rank=32, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"])


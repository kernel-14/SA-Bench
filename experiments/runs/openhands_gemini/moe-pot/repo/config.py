
import os
from easydict import EasyDict as edict

# Configuration for MoE-POT
config = edict()

# ---------------------------- Basic parameters ----------------------------
config.project_name = "MoE-POT"
config.run_name = "moe_pot_pretrain"
config.seed = 42
config.device = "cuda"
config.num_workers = 4 # for data loading

# ---------------------------- Model parameters ----------------------------
config.model = edict()
config.model.attention_dim = 512
config.model.mlp_dim = 512
config.model.num_layers = 4
config.model.num_heads = 4
config.model.num_routed_experts = 16
config.model.num_shared_experts = 2
config.model.top_k_experts = 4
config.model.patch_size = 8
config.model.input_channels = 1 # C (placeholder, will be determined by dataset)
config.model.output_channels = 1 # C (placeholder, will be determined by dataset)
config.model.pos_encoding_dim = None # Will be set to attention_dim

# ---------------------------- Data parameters ----------------------------
config.data = edict()
config.data.dataset_names = ["FNO-1e5", "FNO-1e3", "CNS-01-001", "SWE", "DR", "CFDBench"]
config.data.base_path = "./data"
config.data.h_resolution = 128 # HxW resolution
config.data.time_steps = 10 # T
config.data.noise_epsilon = 0.05 # Epsilon for noise injection, placeholder
config.data.padding_value = 1.0 # Value to pad unused channels
config.data.train_split_ratio = 0.8
config.data.batch_size = 20
config.data.balance_weights = { # Weights for balanced data sampling, placeholder
    "FNO-1e5": 1.0, "FNO-1e3": 1.0, "CNS-01-001": 1.0,
    "SWE": 1.0, "DR": 1.0, "CFDBench": 1.0
}


# ---------------------------- Training parameters ----------------------------
config.train = edict()
config.train.epochs = 1000
config.train.learning_rate = 1e-3
config.train.weight_decay = 1e-6
config.train.adam_betas = (0.9, 0.9)
config.train.warmup_epochs = 200
config.train.balance_loss_weight = 0.1 # w_bal

# ---------------------------- Fine-tuning parameters ----------------------------
config.finetune = edict()
config.finetune.epochs = 200
config.finetune.learning_rate = 1e-3
config.finetune.warmup_epochs = 40

# ---------------------------- Downstream task parameters ----------------------------
config.downstream = edict()
config.downstream.epochs = 500
config.downstream.learning_rate = 1e-3
config.downstream.warmup_epochs = 100

# ---------------------------- Checkpoint parameters ----------------------------
config.checkpoint = edict()
config.checkpoint.save_dir = "./checkpoints"
config.checkpoint.save_interval = 50 # epochs
config.checkpoint.resume = False
config.checkpoint.resume_path = ""

# ---------------------------- Evaluation parameters ----------------------------
config.eval = edict()
config.eval.eval_interval = 10 # epochs


# Ensure directories exist
os.makedirs(config.checkpoint.save_dir, exist_ok=True)
os.makedirs(config.data.base_path, exist_ok=True)

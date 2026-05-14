
import math

class Config:
    def __init__(self, model_size: str = "0.5B", is_ngpt: bool = False):
        self.model_size = model_size
        self.is_ngpt = is_ngpt

        # Model Parameters (from Table 2)
        if model_size == "0.5B":
            self.n_layers = 24
            self.d_model = 1024
            self.n_heads = 16
        elif model_size == "1.0B":
            self.n_layers = 36
            self.d_model = 1280
            self.n_heads = 20
        else:
            raise ValueError(f"Unsupported model size: {model_size}. Choose '0.5B' or '1.0B'.")

        self.d_k = self.d_model // self.n_heads  # Key Dimension (dmodel/nheads)
        self.d_mlp = 4 * self.d_model # MLP Dimension (4dmodel)

        self.vocab_size = 32000 # LLaMA-2 tokenizer with 32k tokens
        self.dropout = 0.1 # Common dropout value in Transformers
        self.rope_base = 10000 # Base for RoPE, mentioned in A.6

        # Optimization Parameters (from Table 3)
        self.optimizer = "Adam" if is_ngpt else "AdamW"
        self.weight_decay = 0.0 if is_ngpt else 0.1
        self.num_warmup_steps = 0 if is_ngpt else 2000
        self.lr_schedule = "Cosine Annealing"
        self.initial_lr = 2e-3 # Example initial LR, paper states "problem-specific" and "best initial learning rate settings"
        self.final_lr = 0.0

        # Training Parameters
        self.batch_size = 512 # Global batch size is 512, A.6
        self.context_length = 4096 # 4k context length for main experiments
        self.max_iters = 200000 # Max iterations for GPT to reach nGPT's performance, but nGPT is faster
        self.eval_interval = 1000
        self.eval_iters = 200
        self.log_interval = 10

        # nGPT specific initializations (from Section 2.6)
        # These are handled within the model's __init__ and _init_weights methods for now.
        # s_qk, s_u, s_v, s_z, alpha_A, alpha_M are initialized within the modules.

        # For the case of initial_lr = 0.1 for 1B model with 8k context, it implies a modification
        # in alpha_A_init and alpha_M_init. We'll stick to the default of 0.05 unless specified.
        # alpha_A_init = 0.05 (order of 1/n_layers)
        # alpha_A_scale = 1/sqrt(d_model)
        # s_qk_init = 1
        # s_qk_scale = 1/sqrt(d_model)
        # s_u_init = 1
        # s_u_scale = 1
        # s_v_init = 1
        # s_v_scale = 1
        # s_z_init = 1
        # s_z_scale = 1/sqrt(d_model)

        # Dataset (A.6)
        self.dataset = "OpenWebText"

        # Hardware (A.6)
        self.num_gpus = 64
        self.gpus_per_node = 8
        self.num_nodes = self.num_gpus // self.gpus_per_node

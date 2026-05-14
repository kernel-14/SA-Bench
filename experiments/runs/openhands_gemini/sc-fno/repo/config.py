
import torch

class Config:
    def __init__(self):
        # General training parameters
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_epochs = 500
        self.learning_rate = 0.001
        self.batch_size = 16 # Default, overridden per case
        self.num_train_samples = 2000 # Default, overridden per case
        self.train_test_split = [0.7, 0.15, 0.15] # Train, Validation, Test

        # FNO Model Hyperparameters (Table C.7)
        self.fno_modes_t = 8
        self.fno_modes_x = 8
        self.fno_modes_y = 8 # Only for 2D spatial PDEs
        self.fno_width = 20
        self.fno_num_fourier_layers = 4
        self.fno_num_learnable_params = None # Will be set dynamically

        # Loss function weights (Algorithm 2 & 3)
        self.lambda_u = 1.0 # Weight for L_u
        self.lambda_s = 1.0 # Weight for L_s
        self.lambda_eq = 1.0 # Weight for L_eq (alpha in paper)

        # Differential Equation Specific Parameters (Table B.6)
        self.equation_configs = {
            "ODE1": {
                "params": {"alpha": [1, 3], "beta": [1, 3], "gamma": [0, 1]},
                "time_steps": 100, "M": 10,
                "fno_modes_t": 8, "fno_modes_x": 8, "fno_modes_y": None,
                "fno_width": 20, "fno_num_fourier_layers": 4,
                "batch_size": 16, "num_train_samples": 2000,
                "num_learnable_params": 17921
            },
            "ODE2": {
                "params": {"delta": [0.02, 0.06], "alpha": [0.01, 0.03], "beta": [20, 60],
                           "gamma": [0.5, 1.5], "omega": [0.2, 0.6],
                           "epsilon": [0.0, 0.2], "zeta": [0.0, 0.2]},
                "time_steps": 100, "M": 10,
                "fno_modes_t": 8, "fno_modes_x": 8, "fno_modes_y": None,
                "fno_width": 20, "fno_num_fourier_layers": 4,
                "batch_size": 16, "num_train_samples": 2000,
                "num_learnable_params": 17921
            },
            "PDE1": {
                "params": {"c": [0.0, 0.25], "alpha": [0.0, 0.1], "beta": [0.0, 0.25],
                           "gamma": [0.0, 0.25], "omega": [0.0, 0.25]},
                "time_steps": 30, "spatial_x": 20, "M": 5,
                "fno_modes_t": 8, "fno_modes_x": 8, "fno_modes_y": None,
                "fno_width": 20, "fno_num_fourier_layers": 4,
                "batch_size": 4, "num_train_samples": 2000,
                "num_learnable_params": 107897
            },
            "PDE2": {
                "params": {"alpha": [0.1, 1.0], "gamma": [0.025, 0.25],
                           "delta": [0.1, 0.5], "omega": [0.01, 0.1]},
                "time_steps": 30, "spatial_x": 40, "M": 5,
                "fno_modes_t": 8, "fno_modes_x": 8, "fno_modes_y": None,
                "fno_width": 20, "fno_num_fourier_layers": 4,
                "batch_size": 4, "num_train_samples": 2000,
                "num_learnable_params": 107897
            },
            "PDE3": { # Navier-Stokes
                "params": {"alpha": [torch.pi, 5*torch.pi], "beta": [torch.pi, 5*torch.pi]},
                "time_steps": 30, "spatial_x": 64, "spatial_y": 64, "M": 1,
                "fno_modes_t": None, "fno_modes_x": 8, "fno_modes_y": 8, # Table C.7 has '-' for mode t
                "fno_width": 20, "fno_num_fourier_layers": 4,
                "batch_size": 4, "num_train_samples": 1000,
                "num_learnable_params": 209397
            },
            "PDE4": { # Allen-Cahn
                "params": {"epsilon": [0.01, 1.0], "alpha": [0.01, 1.0], "beta": [0.01, 1.0],
                           "c": [0.1, 0.9], "omega": [5.0, 10.0]},
                "time_steps": 30, "spatial_x": 40, "M": 5,
                "fno_modes_t": 8, "fno_modes_x": 8, "fno_modes_y": None,
                "fno_width": 20, "fno_num_fourier_layers": 4,
                "batch_size": 1, "num_train_samples": 100,
                "num_learnable_params": 107897
            },
            "PDE2_ZONED": { # High-dimensional parameter space (section 3.4)
                "params": {}, # This will be dynamically generated, 2S+2 = 82 parameters
                "time_steps": 30, "spatial_x": 40, "M": 5,
                "fno_modes_t": 8, "fno_modes_x": 8, "fno_modes_y": None,
                "fno_width": 20, "fno_num_fourier_layers": 4,
                "batch_size": 1, "num_train_samples": 100, # Or 500
                "num_learnable_params": 107897
            }
        }

    def update_for_equation(self, eq_name):
        if eq_name not in self.equation_configs:
            raise ValueError(f"Equation {eq_name} not found in configs.")
        
        eq_config = self.equation_configs[eq_name]
        self.fno_modes_t = eq_config.get("fno_modes_t", self.fno_modes_t)
        self.fno_modes_x = eq_config.get("fno_modes_x", self.fno_modes_x)
        self.fno_modes_y = eq_config.get("fno_modes_y", self.fno_modes_y)
        self.fno_width = eq_config.get("fno_width", self.fno_width)
        self.fno_num_fourier_layers = eq_config.get("fno_num_fourier_layers", self.fno_num_fourier_layers)
        self.batch_size = eq_config.get("batch_size", self.batch_size)
        self.num_train_samples = eq_config.get("num_train_samples", self.num_train_samples)
        self.fno_num_learnable_params = eq_config.get("num_learnable_params", self.fno_num_learnable_params)
        self.current_equation_params = eq_config["params"]
        self.current_equation_time_steps = eq_config["time_steps"]
        self.current_equation_spatial_x = eq_config.get("spatial_x")
        self.current_equation_spatial_y = eq_config.get("spatial_y")
        self.current_equation_M = eq_config["M"]


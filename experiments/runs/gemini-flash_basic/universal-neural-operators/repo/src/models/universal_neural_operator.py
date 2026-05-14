import torch
import torch.nn as nn
import torch.nn.functional as F

from .fno import FNO1d, FNO2d
from .mamba_ssm import MambaFNO1d, MambaFNO2d
from .perceiver_io import PerceiverFNO1d, PerceiverFNO2d
from src.adapters.adapters import LiftingAdapter, ProjectionAdapter

class UniversalNeuralOperator(nn.Module):
    def __init__(self, 
                 core_model_type: str, 
                 shared_model_params: dict, 
                 lifting_params: dict, 
                 projection_params: dict,
                 data_dim: int = 1 # 1 for 1D, 2 for 2D
                ):
        super().__init__()
        self.core_model_type = core_model_type
        self.data_dim = data_dim

        # Initialize adapters
        # lifting_params: {'in_channels': original_in, 'out_channels': width}
        # projection_params: {'in_channels': width, 'out_channels': original_out}
        self.lifting_adapter = LiftingAdapter(**lifting_params)
        self.projection_adapter = ProjectionAdapter(**projection_params)
        
        # The 'width' of the FNO/MambaFNO/PerceiverFNO is the hidden dimension
        # that connects the lifting and projection adapters.
        core_width = lifting_params['out_channels'] # This should be the 'width' for the core models
        
        # Initialize the core neural operator model
        if data_dim == 1:
            if core_model_type == "FNO":
                self.core_model = FNO1d(
                    in_channels=core_width,
                    out_channels=core_width,
                    **shared_model_params
                )
            elif core_model_type == "MambaFNO":
                self.core_model = MambaFNO1d(
                    in_channels=core_width,
                    out_channels=core_width,
                    **shared_model_params
                )
            elif core_model_type == "PerceiverFNO":
                self.core_model = PerceiverFNO1d(
                    in_channels=core_width,
                    out_channels=core_width,
                    **shared_model_params
                )
            else:
                raise ValueError(f"Unknown core_model_type for 1D: {core_model_type}")
        elif data_dim == 2:
            if core_model_type == "FNO":
                self.core_model = FNO2d(
                    in_channels=core_width,
                    out_channels=core_width,
                    **shared_model_params
                )
            elif core_model_type == "MambaFNO":
                self.core_model = MambaFNO2d(
                    in_channels=core_width,
                    out_channels=core_width,
                    **shared_model_params
                )
            elif core_model_type == "PerceiverFNO":
                self.core_model = PerceiverFNO2d(
                    in_channels=core_width,
                    out_channels=core_width,
                    **shared_model_params
                )
            else:
                raise ValueError(f"Unknown core_model_type for 2D: {core_model_type}")
        else:
            raise ValueError(f"Unsupported data_dim: {data_dim}. Only 1D and 2D are supported.")


    def forward(self, x):
        # x: (batchsize, ..., original_in_channels)

        # Apply lifting adapter
        # Lifting adapter expects (batchsize, ..., in_channels)
        # It will transform to (batchsize, ..., latent_dim)
        lifted_x = self.lifting_adapter(x)
        
        # Apply core model
        # Core model expects (batchsize, ..., latent_dim) and outputs (batchsize, ..., latent_dim)
        core_output = self.core_model(lifted_x)
        
        # Apply projection adapter
        # Projection adapter expects (batchsize, ..., latent_dim) and outputs (batchsize, ..., original_out_channels)
        output = self.projection_adapter(core_output)
        
        return output

    def freeze_core(self):
        for param in self.core_model.parameters():
            param.requires_grad = False

    def unfreeze_core(self):
        for param in self.core_model.parameters():
            param.requires_grad = True

    def get_trainable_parameters_for_finetuning(self):
        # During fine-tuning, only adapters are trained
        return list(self.lifting_adapter.parameters()) + \
               list(self.projection_adapter.parameters())

    def get_trainable_parameters_for_pretraining(self):
        # During pre-training, all parameters are trained
        return list(self.parameters())


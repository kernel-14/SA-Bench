
import os
import torch

class Config:
    def __init__(self, experiment_name="wdno_experiment"):
        self.experiment_name = experiment_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_dir = os.path.join("./results", experiment_name)
        os.makedirs(self.output_dir, exist_ok=True)

        # Training parameters
        self.train_batch_size = 16
        self.learning_rate = 1e-4
        self.training_steps = 190000
        self.learning_rate_scheduler = "cosine_annealing" # or "StepLR" for WNO, MWT
        self.optimizer = "Adam"

        # Diffusion Model Parameters (DDPM)
        self.num_diffusion_steps = 1000 # K in algorithm 1
        self.beta_schedule = "linear" # Common for DDPM
        self.beta_start = 0.0001
        self.beta_end = 0.02
        self.ddim_sampling_iterations = 50 # M in paper, number of denoising steps during inference
        self.ddim_eta = 1.0 # eta in paper, controls stochasticity of DDIM
        self.guidance_weight = 120000 # lambda in paper (for control tasks)
        self.guidance_scheduler = "cosine" # For control tasks

        # UNet Architecture Parameters (for both BRM and SRM)
        self.unet_initial_dim = 128
        self.unet_dim_mults = [1, 2, 4, 8]
        self.unet_resnet_block_groups = 8
        self.unet_attention_heads = 4
        self.unet_attention_hidden_dim = 32
        self.unet_conv_kernel_size = 3
        self.unet_downsampling_layers = 4 # Number of downsampling/upsampling layers

        # Wavelet Transform Parameters
        self.wavelet_type_1d = 'bior2.4' # for 1D Burgers, 1D Navier-Stokes, 1D Advection
        self.wavelet_mode_1d = 'periodization'
        self.wavelet_type_2d = 'bior1.3' # for 2D Fluid, ERA5
        self.wavelet_mode_2d = 'zero'
        self.wavelet_level = 5 # for WNO 1D, example value from Table 27

        # Data Parameters (Examples, actual values depend on specific dataset)
        self.data_res_h = 81 # 1D Burgers: 81 time steps
        self.data_res_w = 120 # 1D Burgers: 120 spatial points
        self.data_channels = 1 # Base number of channels for a single variable (e.g., u, f)

        # Number of channels for raw input/target data before wavelet transform
        self.raw_input_channels_x = self.data_channels
        self.raw_input_channels_cond = self.data_channels

        # Specifics for 2D Incompressible Fluid (Table 20)
        self.fluid_attention_heads = 4
        self.fluid_kernel_size_conv3d = (3, 3, 3)
        self.fluid_padding_conv3d = (1, 1, 1)
        self.fluid_stride_conv3d = (1, 1, 1)
        self.fluid_kernel_size_downsampling = (1, 4, 4)
        self.fluid_padding_downsampling = (0, 1, 1)
        self.fluid_stride_downsampling = (1, 2, 2)
        self.fluid_kernel_size_upsampling = (1, 4, 4) # This is repeated in paper, likely a typo
        # self.fluid_padding_upsampling = (0, 1, 1) # Not explicit, assuming based on kernel/stride
        # self.fluid_stride_upsampling = (1, 2, 2) # Not explicit

        # Super-Resolution Training
        self.enable_super_resolution_training = False # Flag to enable/disable SRM training
        self.super_res_factors = [2, 4, 8] # Factors for downsampling to create multi-resolution dataset

        # Experiment Specific Parameters (to be set when running specific experiments)
        self.pde_type = "1d_burgers" # "1d_advection", "1d_navier_stokes", "2d_fluid", "era5"
        self.task_type = "simulation" # "control"

        # Baseline parameters (from tables in Appendix I, J, K)
        # FNO
        self.fno_modes = 16 # for 1D, 16 for 2D
        self.fno_width = 64
        self.fno_input_channels = 3 # 1D Burgers
        self.fno_output_channels = 1 # 1D Burgers
        self.fno_lifting_hidden_channels = 256
        self.fno_projection_hidden_channels = 256
        self.fno_fourier_layers = 4
        self.fno_mlp_expansion_ratio = 0.5
        self.fno_non_linearity = "Gelu"
        self.fno_rank_tensor_factorization = 1.0
        self.fno_domain_padding_mode = "one-sided"

        # WNO
        self.wno_level_wavelet_decomposition = 5
        self.wno_uplifting_dimension = 40
        self.wno_num_wavelet_layers = 4
        self.wno_type_wavelet = "sym4" # for 1D, "db4" for 2D (table 33 says db4, text says bior1.3)

        # CNN (1D Burgers)
        self.cnn_conv_kernel_size = 5
        self.cnn_conv_padding = 2
        self.cnn_activation_function = "ELU"
        self.cnn_latent_vector_size = 256

        # MWT
        self.mwt_wavelet_basis = "legendre"
        self.mwt_num_fourier_modes = 10 # 10 for 1D, 12 for 2D
        self.mwt_kernel_size = 4 # 4 for 1D, 3 for 2D

        # OFormer
        self.oformer_encoder_type = "SpatialEncoder2D" # "SpatialTemporalEncoder2D" for 2D
        self.oformer_encoder_input_channels = 3 # 1D
        self.oformer_encoder_embedding_dim_token = 96
        self.oformer_encoder_embedding_dim_encoded_sequence = 256
        self.oformer_encoder_heads = 4
        self.oformer_encoder_depth = 6
        self.oformer_encoder_resolution = 120 # 1D
        self.oformer_encoder_dropout_embedding = 0.05
        self.oformer_decoder_type = "PointWiseDecoder2DSimple"
        self.oformer_decoder_latent_channels = 256
        self.oformer_decoder_out_channels = 1
        self.oformer_decoder_scale = 0.5
        self.oformer_decoder_res = 120

        # OFormer 2D Specifics (Table 35)
        self.oformer_2d_encoder_input_channels = 3 # This contradicts paper '3496'
        self.oformer_2d_encoder_embedding_dim_token = 96
        self.oformer_2d_encoder_embedding_dim_encoded_sequence = 192
        self.oformer_2d_encoder_heads = 1
        self.oformer_2d_encoder_depth = 5
        self.oformer_2d_decoder_out_channels = 1
        self.oformer_2d_decoder_propagate_forward = 3
        self.oformer_2d_decoder_length_output_sequence = 2
        self.oformer_2d_decoder_propagator_depth = 10
        self.oformer_2d_decoder_curriculum_ratio = 0.1
        self.oformer_2d_decoder_curriculum_steps = 610

        # MS-L-NODE
        self.mslnode_encoder_cnn_channels = 128
        self.mslnode_encoder_latent_dim = 8
        self.mslnode_decoder_cnn_channels = 128
        self.mslnode_decoder_latent_dim = 8
        self.mslnode_agg_heads = 1
        self.mslnode_agg_static_layers = 64
        self.mslnode_agg_dynamical_layers = 8

        # PID, SAC, BC, BPPO specific configs are more complex and depend on environment interaction,
        # so they will be integrated when those specific baselines are implemented.

    def update_for_pde(self, pde_type):
        self.pde_type = pde_type
        if pde_type == "1d_burgers":
            self.data_res_h = 81
            self.data_res_w = 120
            self.data_channels = 1 # u(t,x) is 1 channel
            self.raw_input_channels_x = self.data_channels # u_data has 1 channel
            # For 1D Burgers simulation, condition is u0 (1ch) and f (1ch), so 2 channels
            # For 1D Burgers control, condition is u0 (1ch) and u_T (1ch), so 2 channels
            self.raw_input_channels_cond = self.data_channels * 2 
            
            self.fno_input_channels = 3 # u0 and f
            self.fno_output_channels = 1 # u_0_T
            self.wno_type_wavelet = "sym4" # Table 27
            self.mwt_num_fourier_modes = 10
            self.mwt_kernel_size = 4
            self.oformer_encoder_input_channels = 3
            self.oformer_encoder_resolution = 120
            self.oformer_decoder_res = 120
        elif pde_type == "1d_advection":
            self.data_res_h = 80 # 80 timesteps of evolution
            self.data_res_w = 120 # Assuming same spatial res as burgers
            self.data_channels = 1
            self.raw_input_channels_x = self.data_channels # u_data has 1 channel
            # For Advection simulation, condition is u0 (1ch)
            self.raw_input_channels_cond = self.data_channels
            self.wno_type_wavelet = "bior2.4" # Assuming similar to 1D Burgers for consistency based on paper text
        elif pde_type == "1d_navier_stokes":
            self.data_res_h = 81
            self.data_res_w = 120
            self.data_channels = 1
            self.raw_input_channels_x = self.data_channels
            # For 1D Navier-Stokes simulation, condition is u0 (1ch)
            self.raw_input_channels_cond = self.data_channels
            self.wno_type_wavelet = "bior2.4"
            self.mwt_num_fourier_modes = 10
            self.mwt_kernel_size = 4
        elif pde_type == "2d_fluid":
            self.data_res_h = 32 # Time steps
            self.data_res_w = 64 # Spatial res x
            self.data_res_d = 64 # Spatial res y
            self.data_channels = 3 # Density, velocity_x, velocity_y
            self.raw_input_channels_x = self.data_channels # u_data has 3 channels (density, vx, vy)
            # For 2D fluid simulation: condition is initial density (1ch) and force (1ch), so 2 channels
            # For 2D fluid control: condition is initial density (1ch) and target smoke percentage (1ch), so 2 channels
            self.raw_input_channels_cond = 1 + 1 # Initial density (1) + Force (1) for simulation, or Initial density (1) + Target Smoke (1) for control

            self.wavelet_type_2d = 'bior1.3'
            self.fno_input_channels = 6 # previous density, velocity, control, percentage
            self.fno_output_channels = 3 # next density, velocity, percentage
            self.mwt_num_fourier_modes = 12
            self.mwt_kernel_size = 3
            self.oformer_encoder_type = "SpatialTemporalEncoder2D"
            # Note: paper's 2D OFormer input_channels (Table 35) '3496' seems like a concatenation or flattened value,
            # using '3' as a placeholder for actual channels (e.g. density, vx, vy)
            self.oformer_encoder_input_channels = 3
            self.oformer_encoder_resolution = (64, 64) # Assuming spatial resolution
            self.oformer_decoder_res = (64, 64)
        elif pde_type == "era5":
            self.data_res_h = 20 # 20 hours prediction
            self.data_res_w = None # Not specified, depends on resolution
            self.data_res_d = None
            self.data_channels = 1 # Temperature
            self.raw_input_channels_x = self.data_channels # u_data has 1 channel
            self.raw_input_channels_cond = self.data_channels # a_data (previous 12 hours state) has 1 channel
            self.wavelet_type_2d = 'bior1.3' # Assuming similar to 2D Fluid

config = Config()

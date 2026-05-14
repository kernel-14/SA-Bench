
import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    # General
    config.seed = 42
    config.log_dir = './logs'
    config.save_dir = './checkpoints'

    # Data
    config.data = data = ml_collections.ConfigDict()
    data.pde_name = 'Burgers' # 'Burgers', 'Hyper Diffusion', 'Kuramoto-Sivashinsky', 'Advection-Diffusion'
    data.low_data_regime = True
    data.n_train_low_data = 25 # Number of training trajectories for low data regime
    data.n_train_ood = 1000 # Number of training trajectories for OOD experiment
    data.n_val = 250
    data.n_test = 250
    data.spatial_resolution = 256 # For 1D PDEs
    data.temporal_resolution = 59 # Number of sub-sampled time steps
    data.num_initial_steps = 10 # For autoregressive prediction
    data.advection_diffusion_params = ml_collections.ConfigDict()
    data.advection_diffusion_params.alpha = 0.026 # Diffusion coefficient
    data.advection_diffusion_params.dt_sim = 5e-10 # Simulation delta t
    data.advection_diffusion_params.total_time_steps_sim = 200 # Total simulation steps

    # Model (FNO)
    config.model = model = ml_collections.ConfigDict()
    model.architecture = 'FNO'
    model.modes = 12 # per spatial dimension
    model.hidden_dim = 18
    model.num_fourier_blocks = 4
    model.output_dim = 1 # For 1D PDEs, scalar field
    model.add_pos_encoding = True # If positional encoding is added to input

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.epochs = 100 # For low data regime
    training.epochs_ood = 1000 # For OOD experiment
    training.learning_rate = 1e-3
    training.batch_size = 32
    training.optimizer = 'AdamW'
    training.weight_decay = 1e-4 # AdamW default, adjust as needed based on paper's impl.
    training.cosine_decay_steps = None # Calculated based on epochs and batch_size
    training.warmup_steps = None # Typically 10% of total steps, or fixed small number

    # Uncertainty Quantification
    config.uq = uq = ml_collections.ConfigDict()
    uq.method = 'LUNO-LA' # 'LUNO-LA', 'LUNO-Iso', 'Sample-LA', 'Sample-Iso', 'Ensemble', 'InputPerturbations'
    uq.num_samples = 200 # For sample-based UQ
    uq.sigma_iso = None # Calibrated for isotropic Gaussian
    uq.laplace_rank = 500 # Rank for low-rank GGN approximation
    uq.laplace_minibatch_size = 1000 # For OOD experiment GGN approximation
    uq.last_layer_la = True # Restrict uncertainty to last Fourier block

    # Evaluation
    config.evaluation = evaluation = ml_collections.ConfigDict()
    evaluation.metrics = ['RMSE', 'NLL', 'Chi2']
    evaluation.n_test_samples = 250
    evaluation.calibration_grid_points = 500

    return config


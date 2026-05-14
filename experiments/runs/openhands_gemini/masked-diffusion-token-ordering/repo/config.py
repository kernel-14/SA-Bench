
import torch

class Config:
    # General
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model Architecture (based on transformer in Section 3.2, "Experimental setup")
    # Paper mentions "transformer with causal attention" for pi-learner, and "170M MDM" or "8B LLaDA"
    # We will define a general transformer block here, specific sizes can be adjusted for experiments.
    model_type = "transformer"
    vocab_size = 50257 # Example for common text models, adjust based on dataset
    max_sequence_length = 2048 # Section 3.2
    hidden_size = 768 # Base size, can scale up
    num_attention_heads = 12
    num_layers = 12
    intermediate_size = hidden_size * 4
    hidden_act = "gelu"
    hidden_dropout_prob = 0.1
    attention_probs_dropout_prob = 0.1
    initializer_range = 0.02
    layer_norm_eps = 1e-12
    use_learnable_pos_embeddings = True # Section 3.2: "learnable positional embedding layer for all experiments to correct this."

    # MDM Specific (Section 2)
    mask_token_id = 0 # As defined in Section 2
    
    # Noise Schedule (Section 2)
    # The paper mentions alpha_t is a predefined noise schedule satisfying alpha_0 approx 1, alpha_1 approx 0.
    # It doesn't specify the exact schedule, so we'll use a common linear schedule for alpha_t, and derive 1-alpha_t.
    # We will discretize t from 0 to 1 into 'num_diffusion_steps'
    num_diffusion_steps = 1000 # Common for diffusion models

    # Training (Section 2.1.1, Section 3.2)
    batch_size = 128 # Section D.2 for Sudoku/Zebra
    learning_rate = 4e-4 # Section C.1
    min_learning_rate = 4e-5 # Section C.1
    weight_decay = 0.1 # Section C.1
    beta1 = 0.9 # Section C.1
    beta2 = 0.95 # Section C.1
    num_train_epochs = 300 # Section D.2 for Sudoku/Zebra (for logic puzzles)
    # For text data, "total number of tokens observed by the model during training equals C / 6N"
    # This implies variable iterations based on FLOPs. We'll use epochs for now.
    
    # Optimizer
    optimizer_type = "AdamW" # Section C.1

    # Dataset (Section 3.2)
    dataset_name = "Slimpajama" # Section 3.2, for text data
    
    # Inference (Section 4)
    sampling_steps = 50 # Section D.2
    
    # Adaptive Inference specific (Section 4.1)
    gumbel_noise_coeff = 0.5 # Section D.2
    
    # Evaluation (Section 4.2)
    # L&O-NAE-SAT distributions: (25, 275), (30, 270), (40, 260), (50, 250), (100, 200) - Section D.1.1
    # Sudoku: 6M GPT-2 model (Section D.2)
    # Zebra: 19M model (Section D.2)
    # LLaDA-8B for natural language tasks (Section 4.4)
    

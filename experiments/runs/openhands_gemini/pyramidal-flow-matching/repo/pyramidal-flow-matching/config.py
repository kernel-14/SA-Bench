
import argparse

def get_config():
    parser = argparse.ArgumentParser(description="Pyramidal Flow Matching for Efficient Video Generative Modeling")

    # Model Architecture
    parser.add_argument('--model_name', type=str, default='MM-DiT',
                        help='Base model architecture. Paper uses MM-DiT based on SD3 Medium.')
    parser.add_argument('--num_transformer_layers', type=int, default=24,
                        help='Number of transformer layers in the DiT model.')
    parser.add_argument('--model_params', type=int, default=2_000_000_000,
                        help='Total number of parameters in the model (2B for MM-DiT).')
    parser.add_argument('--text_encoder_t5', type=str, default='google/t5-v1_1-large',
                        help='T5 encoder for prompt embedding.')
    parser.add_argument('--text_encoder_clip', type=str, default='openai/clip-vit-large-patch14',
                        help='CLIP encoder for prompt embedding.')
    parser.add_argument('--vae_compression_factor', type=int, default=8,
                        help='Spatial and temporal downsampling ratio for 3D VAE (8x8x8).')
    parser.add_argument('--num_pyramid_stages', type=int, default=3,
                        help='Number of pyramid stages for spatial and temporal pyramids.')

    # Training Procedure
    parser.add_argument('--stage1_epochs', type=int, default=50000,
                        help='Training steps for stage 1 (image training).')
    parser.add_argument('--stage2_epochs_2s', type=int, default=80000,
                        help='Training steps for stage 2 (low-res video, 2s).')
    parser.add_argument('--stage2_epochs_5s', type=int, default=120000,
                        help='Training steps for stage 2 (low-res video, 5s).')
    parser.add_argument('--stage3_epochs', type=int, default=50000,
                        help='Training steps for stage 3 (high-res video).')
    parser.add_argument('--image_data_proportion_stage2', type=float, default=0.125,
                        help='Proportion of image data in each batch during Stage 2.')

    # Optimizer Hyperparameters (Table 4)
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        help='Optimizer to use.')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Beta1 for AdamW optimizer.')
    parser.add_argument('--beta2_stage1', type=float, default=0.999,
                        help='Beta2 for AdamW optimizer in Stage 1.')
    parser.add_argument('--beta2_stages23', type=float, default=0.95,
                        help='Beta2 for AdamW optimizer in Stage 2 and 3.')
    parser.add_argument('--eps', type=float, default=1e-6,
                        help='Epsilon for AdamW optimizer.')
    parser.add_argument('--global_batch_size_stage1', type=int, default=1536,
                        help='Global batch size for Stage 1.')
    parser.add_argument('--global_batch_size_stage2', type=int, default=768,
                        help='Global batch size for Stage 2.')
    parser.add_argument('--global_batch_size_stage3', type=int, default=384,
                        help='Global batch size for Stage 3.')
    parser.add_argument('--learning_rate_stage12', type=float, default=1e-4,
                        help='Learning rate for Stage 1 and 2.')
    parser.add_argument('--learning_rate_stage3', type=float, default=5e-5,
                        help='Learning rate for Stage 3.')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='Number of warmup steps for learning rate schedule.')
    parser.add_argument('--lr_schedule', type=str, default='Constant with warmup',
                        help='Learning rate schedule type.')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer.')
    parser.add_argument('--gradient_clipping', type=float, default=1.0,
                        help='Gradient clipping value.')
    parser.add_argument('--numerical_precision', type=str, default='bfloat16',
                        help='Numerical precision for training (bfloat16).')
    parser.add_argument('--num_gpus', type=int, default=128,
                        help='Number of NVIDIA A100 GPUs used.')

    # Data
    parser.add_argument('--image_datasets', type=list,
                        default=['LAION-5B', 'CC-12M', 'SA-1B', 'JourneyDB'],
                        help='Image datasets used for training.')
    parser.add_argument('--video_datasets', type=list,
                        default=['WebVid-10M', 'OpenVid-1M', 'Open-Sora Plan 1M'],
                        help='Video datasets used for training.')
    parser.add_argument('--resolution', type=int, default=768,
                        help='Generated video resolution (e.g., 768p).')
    parser.add_argument('--fps', type=int, default=24,
                        help='Frames per second for generated videos.')
    parser.add_argument('--max_video_duration_s', type=int, default=10,
                        help='Maximum video duration in seconds.')
    parser.add_argument('--max_video_frames', type=int, default=241,
                        help='Maximum number of video frames (for 10s at 24fps + 1 for conditioning).')
    parser.add_argument('--history_noise_strength_min', type=float, default=0.0,
                        help='Minimum strength of corruptive noise added to history pyramid conditions.')
    parser.add_argument('--history_noise_strength_max', type=float, default=1/3,
                        help='Maximum strength of corruptive noise added to history pyramid conditions.')


    # Inference
    parser.add_argument('--guidance_scale', type=float, default=4.0,
                        help='Classifier-free guidance scale.')

    args = parser.parse_args([]) # Passing an empty list to avoid parsing command line args here
    return args


#!/bin/bash

# scripts/evaluate.sh
#
# This script launches the evaluation phase for the trained P2VAE and FMT models.
# It uses distributed execution if multiple GPUs are configured in config.yaml.
# It requires pre-trained P2VAE and FMT model checkpoints to be available.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration Variables ---
# Path to the main configuration file
CONFIG_FILE="./config.yaml"

# Default number of GPUs if not specified or parsing fails
DEFAULT_NUM_GPUS=1

# --- Extract Number of GPUs from config.yaml ---
# We use 'yq' to parse the YAML file and extract the 'global.num_gpus' value.
# This assumes 'yq' is installed on the system (e.g., `sudo snap install yq` or `brew install yq`).
# If 'yq' is not found or the key is missing/invalid, it defaults to DEFAULT_NUM_GPUS.
if command -v yq &> /dev/null; then
    NUM_GPUS=$(yq e '.global.num_gpus' "$CONFIG_FILE")
    if [ -z "$NUM_GPUS" ] || ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]]; then
        echo "Warning: 'global.num_gpus' not found or is invalid in $CONFIG_FILE. Using default: $DEFAULT_NUM_GPUS GPU(s)."
        NUM_GPUS=$DEFAULT_NUM_GPUS
    else
        echo "Detected $NUM_GPUS GPU(s) from $CONFIG_FILE."
    fi
else
    echo "Warning: 'yq' command not found. Falling back to default: $DEFAULT_NUM_GPUS GPU(s)."
    echo "Please install 'yq' (e.g., 'sudo snap install yq' or 'brew install yq') for robust config parsing."
    NUM_GPUS=$DEFAULT_NUM_GPUS
fi

# --- Validate Number of GPUs ---
if [ "$NUM_GPUS" -le 0 ]; then
    echo "Error: Number of GPUs must be a positive integer. Got $NUM_GPUS. Exiting."
    exit 1
fi

# --- Extract Checkpoint Directory from config.yaml ---
# This directory is where the P2VAE and FMT model checkpoints are expected to be found.
if command -v yq &> /dev/null; then
    CHECKPOINT_ROOT=$(yq e '.logging.checkpoint_dir' "$CONFIG_FILE")
    if [ -z "$CHECKPOINT_ROOT" ]; then
        echo "Warning: 'logging.checkpoint_dir' not found in $CONFIG_FILE. Using default: './checkpoints'."
        CHECKPOINT_ROOT="./checkpoints"
    else
        echo "Using checkpoint root: $CHECKPOINT_ROOT."
    fi
else
    echo "Warning: 'yq' command not found. Cannot extract checkpoint_dir. Using default: './checkpoints'."
    CHECKPOINT_ROOT="./checkpoints"
fi

# --- Define P2VAE and FMT Checkpoint Paths ---
# This script assumes the P2VAETrainer and FMTTrainer saved their best models
# as 'p2vae_best_model.pth' and 'fmt_best_model.pth' respectively.
P2VAE_CHECKPOINT="$CHECKPOINT_ROOT/p2vae_best_model.pth"
FMT_CHECKPOINT="$CHECKPOINT_ROOT/fmt_best_model.pth"

# --- Validate Checkpoint Existence ---
if [ ! -f "$P2VAE_CHECKPOINT" ]; then
    echo "Error: Pre-trained P2VAE checkpoint not found at $P2VAE_CHECKPOINT."
    echo "Please ensure the P2VAE model has been trained successfully (e.g., using scripts/train_p2vae.sh) "
    echo "and that 'p2vae_best_model.pth' is present in the specified checkpoint directory."
    exit 1
fi

if [ ! -f "$FMT_CHECKPOINT" ]; then
    echo "Error: Pre-trained FMT checkpoint not found at $FMT_CHECKPOINT."
    echo "Please ensure the FMT model has been trained successfully (e.g., using scripts/train_fmt.sh) "
    echo "and that 'fmt_best_model.pth' is present in the specified checkpoint directory."
    exit 1
fi

# --- Set PYTHONPATH ---
# Add the current directory to PYTHONPATH so Python can find local modules
# (e.g., main.py, data/, models/, training/, utils/).
export PYTHONPATH=$PYTHONPATH:$(pwd)

# --- Launch Evaluation ---
echo "Launching evaluation on $NUM_GPUS GPU(s)..."
echo "Config file: $CONFIG_FILE"
echo "P2VAE Checkpoint: $P2VAE_CHECKPOINT"
echo "FMT Checkpoint: $FMT_CHECKPOINT"

if [ "$NUM_GPUS" -gt 1 ]; then
    # Multi-GPU (Distributed) Evaluation using torch.distributed.launch
    # The --master_port argument should ideally be unique for each distributed job.
    # main.py internally handles the local_rank argument provided by torch.distributed.launch.
    python -m torch.distributed.launch \
        --nproc_per_node="$NUM_GPUS" \
        --master_port=29502 \
        main.py \
        --config_path="$CONFIG_FILE" \
        --stage="evaluate" \
        --p2vae_checkpoint="$P2VAE_CHECKPOINT" \
        --fmt_checkpoint="$FMT_CHECKPOINT"
else
    # Single-GPU Evaluation
    python main.py \
        --config_path="$CONFIG_FILE" \
        --stage="evaluate" \
        --p2vae_checkpoint="$P2VAE_CHECKPOINT" \
        --fmt_checkpoint="$FMT_CHECKPOINT"
fi

echo "Evaluation script finished. Check logs directory for detailed results."

#!/bin/bash

# scripts/train_fmt.sh
#
# This script launches the Flow Marching Transformer (FMT) training process
# using distributed training via torch.distributed.launch, if multiple GPUs
# are available and configured. It requires a pre-trained P2VAE model.

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
# This directory is where the P2VAE model checkpoint is expected to be found.
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

# --- Define P2VAE Checkpoint Path ---
# FMT training requires a pre-trained P2VAE model.
# This assumes the P2VAETrainer saved its best model as 'p2vae_best_model.pth' in the checkpoint_dir.
P2VAE_CHECKPOINT="$CHECKPOINT_ROOT/p2vae_best_model.pth"

# Ensure the P2VAE checkpoint exists before attempting to train FMT.
if [ ! -f "$P2VAE_CHECKPOINT" ]; then
    echo "Error: Pre-trained P2VAE checkpoint not found at $P2VAE_CHECKPOINT."
    echo "Please ensure the P2VAE model has been trained successfully (e.g., using scripts/train_p2vae.sh) "
    echo "and that 'p2vae_best_model.pth' is present in the specified checkpoint directory."
    exit 1
fi

# --- Set PYTHONPATH ---
# Add the current directory to PYTHONPATH so Python can find local modules
# (e.g., main.py, data/, models/, training/, utils/).
export PYTHONPATH=$PYTHONPATH:$(pwd)

# --- Launch Training ---
echo "Launching FMT training on $NUM_GPUS GPU(s)..."

if [ "$NUM_GPUS" -gt 1 ]; then
    # Multi-GPU (Distributed) Training
    # torch.distributed.launch handles setting up environment variables like MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE.
    # --master_port is typically chosen to be a free port.
    python -m torch.distributed.launch \
        --nproc_per_node="$NUM_GPUS" \
        --master_port=29501 \
        main.py \
        --config_path="$CONFIG_FILE" \
        --stage="train_fmt" \
        --p2vae_checkpoint="$P2VAE_CHECKPOINT"
else
    # Single-GPU Training
    python main.py \
        --config_path="$CONFIG_FILE" \
        --stage="train_fmt" \
        --p2vae_checkpoint="$P2VAE_CHECKPOINT"
fi

echo "FMT training script finished."

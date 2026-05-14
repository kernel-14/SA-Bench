#!/bin/bash

# scripts/train_p2vae.sh
#
# This script launches the P2VAE training process using distributed training
# via torch.distributed.launch, based on the configuration defined in config.yaml.

# --- Configuration Variables ---
# Path to the main configuration file
CONFIG_FILE="./config.yaml"

# Default number of GPUs if not specified in config or yq is not available
DEFAULT_NUM_GPUS=1

# --- Extract Number of GPUs from config.yaml ---
# We use 'yq' to parse the YAML file and extract the 'global.num_gpus' value.
# If 'yq' is not installed or the key is not found, it defaults to DEFAULT_NUM_GPUS.
# To install yq: sudo apt-get update && sudo apt-get install yq (Linux) or brew install yq (macOS)
if command -v yq &> /dev/null
then
    NUM_GPUS=$(yq e '.global.num_gpus' "$CONFIG_FILE")
    if [ -z "$NUM_GPUS" ] || ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]]
    then
        echo "Warning: 'global.num_gpus' not found or is invalid in $CONFIG_FILE. Using default: $DEFAULT_NUM_GPUS GPU(s)."
        NUM_GPUS=$DEFAULT_NUM_GPUS
    else
        echo "Detected $NUM_GPUS GPU(s) from $CONFIG_FILE."
    fi
else
    echo "Warning: 'yq' command not found. Falling back to default: $DEFAULT_NUM_GPUS GPU(s)."
    echo "Please install 'yq' (e.g., 'brew install yq' or 'snap install yq') for robust config parsing."
    NUM_GPUS=$DEFAULT_NUM_GPUS
fi

# --- Validate Number of GPUs ---
if [ "$NUM_GPUS" -le 0 ]; then
    echo "Error: Number of GPUs must be a positive integer. Got $NUM_GPUS. Exiting."
    exit 1
fi

# --- Launch Distributed Training ---
echo "Launching P2VAE training on $NUM_GPUS GPU(s)..."

# The torch.distributed.launch utility (or torchrun in newer PyTorch versions)
# handles setting up environment variables like MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE.
# We pass the --config_path and --stage arguments to main.py.
# The `find_unused_parameters=False` is often a good default for performance
# in DDP unless there are specific parts of the model that are intentionally not used
# in the forward pass. For P2VAE, typically all parameters are used.
# It is important that main.py is prepared to handle the DDP setup using `local_rank`.
python -m torch.distributed.launch \
    --nproc_per_node="$NUM_GPUS" \
    main.py \
    --config_path="$CONFIG_FILE" \
    --stage="train_p2vae"

echo "P2VAE training script finished."


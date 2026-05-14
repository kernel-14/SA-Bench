#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Argument Parsing ---

# Mandatory argument: MODEL_VARIANT (e.g., navil_2b, navil_9b)
MODEL_VARIANT="${1}"
if [ -z "${MODEL_VARIANT}" ]; then
    echo "Error: MODEL_VARIANT is a mandatory argument (e.g., navil_2b, navil_9b)."
    echo "Usage: ./scripts/train_navil.sh <MODEL_VARIANT> [NUM_GPUS] [--master_addr <IP>] [--master_port <PORT>] [--checkpoint_path <PATH>]"
    exit 1
fi

# Optional argument: NUM_GPUS (number of processes/GPUs to use per node)
# If not provided, accelerate launch will default to using all available GPUs.
NUM_GPUS="${2}"

# Initialize optional arguments with default values
MASTER_ADDR="127.0.0.1"
MASTER_PORT="29500"
CHECKPOINT_PATH="" # Empty if not provided, main.py will handle absence

# Shift past mandatory and optional positional arguments
shift 2 # Shift past MODEL_VARIANT and NUM_GPUS if they exist

# Parse remaining optional named arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --master_addr)
      if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
        MASTER_ADDR="$2"
        shift 2
      else
        echo "Error: Argument for --master_addr is missing." >&2
        exit 1
      fi
      ;;
    --master_port)
      if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
        MASTER_PORT="$2"
        shift 2
      else
        echo "Error: Argument for --master_port is missing." >&2
        exit 1
      fi
      ;;
    --checkpoint_path)
      if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
        CHECKPOINT_PATH="$2"
        shift 2
      else
        echo "Error: Argument for --checkpoint_path is missing." >&2
        exit 1
      fi
      ;;
    -*|--*=) # Unrecognized options
      echo "Error: Unrecognized option $1" >&2
      exit 1
      ;;
    *) # Unrecognized positional arguments
      echo "Error: Unrecognized positional argument $1" >&2
      exit 1
      ;;
  esac
done

# --- Validate NUM_GPUS and Prepare accelerate launch argument ---
NPROC_PER_NODE_ARG=""
if [ -n "${NUM_GPUS}" ]; then
    if ! [[ "${NUM_GPUS}" =~ ^[0-9]+$ ]] || [ "${NUM_GPUS}" -lt 1 ]; then
        echo "Error: NUM_GPUS must be a positive integer if specified."
        exit 1
    fi
    NPROC_PER_NODE_ARG="--num_processes ${NUM_GPUS}"
    echo "Requested number of GPUs per node: ${NUM_GPUS}"
else
    echo "NUM_GPUS not specified. accelerate launch will automatically use all available GPUs."
fi

# --- Environment Variables Setup ---
echo "Setting up distributed training environment variables..."
export MASTER_ADDR="${MASTER_ADDR}"
export MASTER_PORT="${MASTER_PORT}"

echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "MODEL_VARIANT: ${MODEL_VARIANT}"
if [ -n "${CHECKPOINT_PATH}" ]; then
    echo "CHECKPOINT_PATH: ${CHECKPOINT_PATH}"
fi

# --- Define Paths ---
# Get the directory where the script is located, then go up one level to find the project root.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(dirname "${SCRIPT_DIR}")

CONFIG_PATH="${PROJECT_ROOT}/config.yaml"
MAIN_SCRIPT="${PROJECT_ROOT}/main.py"
# Output directory for logs and checkpoints, specific to the model variant and mode
OUTPUT_DIR="${PROJECT_ROOT}/output/${MODEL_VARIANT}_train"

echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "CONFIG_PATH: ${CONFIG_PATH}"
echo "MAIN_SCRIPT: ${MAIN_SCRIPT}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"

# --- Construct and Launch Training Process using accelerate launch ---
echo "Launching training for ${MODEL_VARIANT}..."

# Base accelerate launch command
ACCELERATE_COMMAND="accelerate launch \
    --master_port ${MASTER_PORT} \
    --main_process_ip ${MASTER_ADDR} \
    ${NPROC_PER_NODE_ARG} \
    ${MAIN_SCRIPT} \
    --mode train \
    --config_path ${CONFIG_PATH} \
    --model_variant ${MODEL_VARIANT} \
    --output_dir ${OUTPUT_DIR}"

# Add checkpoint path if provided
if [ -n "${CHECKPOINT_PATH}" ]; then
    ACCELERATE_COMMAND="${ACCELERATE_COMMAND} --checkpoint_path ${CHECKPOINT_PATH}"
fi

echo "Executing command: ${ACCELERATE_COMMAND}"
eval "${ACCELERATE_COMMAND}"

echo "Training process finished for ${MODEL_VARIANT}."

#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Argument Parsing ---

# Initialize variables with empty strings or default values
CONFIG_PATH=""
MODEL_VARIANT=""
CHECKPOINT_PATH=""
BENCHMARKS="" # Optional: comma-separated list of benchmarks
NUM_GPUS=1   # Default to 1 GPU
OUTPUT_DIR=""  # Optional: output directory for evaluation results

# Use getopt for robust argument parsing
# -o "" specifies no short options
# --long defines the long options with a colon indicating they require an argument
TEMP_ARGS=$(getopt -o "" --long config-path:,model-variant:,checkpoint-path:,benchmarks::,num-gpus::,output-dir: -- "$@")
if [ $? -ne 0 ]; then
  echo "Error: Failed to parse arguments. Please check your command line syntax." >&2
  exit 1
fi

# Eval the parsed arguments to reset positional parameters
eval set -- "$TEMP_ARGS"

# Loop through the parsed arguments
while true; do
  case "$1" in
    --config-path)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --model-variant)
      MODEL_VARIANT="$2"
      shift 2
      ;;
    --checkpoint-path)
      CHECKPOINT_PATH="$2"
      shift 2
      ;;
    --benchmarks)
      # Optional argument, might be empty if just --benchmarks is passed without value
      if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
        BENCHMARKS="$2"
        shift 2
      else
        BENCHMARKS="" # No value provided, keep it empty for main.py to use config default
        shift 1
      fi
      ;;
    --num-gpus)
      # Optional argument, might be empty if just --num-gpus is passed without value
      if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
        NUM_GPUS="$2"
        shift 2
      else
        NUM_GPUS=1 # Default if flag is present but no value
        shift 1
      fi
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --) # End of options
      shift
      break
      ;;
    *)
      echo "Internal error in argument parsing!" >&2
      exit 1
      ;;
  esac
done

# --- Argument Validation ---

# Check if mandatory arguments are provided
if [[ -z "$CONFIG_PATH" ]]; then
  echo "Error: --config-path is a mandatory argument." >&2
  exit 1
fi
if [[ -z "$MODEL_VARIANT" ]]; then
  echo "Error: --model-variant is a mandatory argument." >&2
  exit 1
fi
if [[ -z "$CHECKPOINT_PATH" ]]; then
  echo "Error: --checkpoint-path is a mandatory argument for evaluation." >&2
  exit 1
fi

# Validate NUM_GPUS
if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] || [ "$NUM_GPUS" -lt 1 ]; then
  echo "Error: --num-gpus must be a positive integer." >&2
  exit 1
fi

# --- Define Paths ---
# Get the directory where the script is located, then go up one level to find the project root.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(dirname "${SCRIPT_DIR}")

MAIN_SCRIPT="${PROJECT_ROOT}/main.py"
# Output directory for evaluation results, if not specified via CLI, main.py will use config's default.
# If provided via CLI, it takes precedence.
CLI_OUTPUT_DIR_ARG=""
if [[ -n "$OUTPUT_DIR" ]]; then
  CLI_OUTPUT_DIR_ARG="--output_dir ${OUTPUT_DIR}"
  echo "CLI specified output directory: ${OUTPUT_DIR}"
else
  # Use a default structure if not specified, based on model variant and mode
  OUTPUT_DIR_DEFAULT="${PROJECT_ROOT}/output/${MODEL_VARIANT}_eval"
  CLI_OUTPUT_DIR_ARG="--output_dir ${OUTPUT_DIR_DEFAULT}"
  echo "Using default output directory: ${OUTPUT_DIR_DEFAULT}"
fi


echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "CONFIG_PATH: ${CONFIG_PATH}"
echo "MAIN_SCRIPT: ${MAIN_SCRIPT}"
echo "MODEL_VARIANT: ${MODEL_VARIANT}"
echo "CHECKPOINT_PATH: ${CHECKPOINT_PATH}"
echo "NUM_GPUS: ${NUM_GPUS}"
if [[ -n "$BENCHMARKS" ]]; then
  echo "EVAL BENCHMARKS: ${BENCHMARKS}"
else
  echo "EVAL BENCHMARKS: All from config.yaml"
fi


# --- Construct Python Command for main.py ---
PYTHON_COMMAND="python ${MAIN_SCRIPT} \
  --mode eval \
  --config_path ${CONFIG_PATH} \
  --model_variant ${MODEL_VARIANT} \
  --checkpoint_path ${CHECKPOINT_PATH} \
  ${CLI_OUTPUT_DIR_ARG}"

# Add optional benchmarks argument if provided
if [[ -n "$BENCHMARKS" ]]; then
  PYTHON_COMMAND="${PYTHON_COMMAND} --eval_benchmarks ${BENCHMARKS}"
fi

# --- Execute Evaluation Process ---
echo "Executing evaluation command:"
echo "  ${PYTHON_COMMAND}"

if [[ "$NUM_GPUS" -gt 1 ]]; then
  echo "Launching distributed evaluation using accelerate launch with ${NUM_GPUS} processes..."
  # For distributed evaluation, accelerate launch will handle setting up environment variables
  # and passing the local_rank to main.py automatically.
  accelerate launch --num_processes "${NUM_GPUS}" ${PYTHON_COMMAND}
else
  echo "Launching single-GPU/CPU evaluation..."
  # Directly execute the python command
  eval "${PYTHON_COMMAND}"
fi

echo "Evaluation process finished for ${MODEL_VARIANT}."


#!/bin/bash

# This script orchestrates the execution of all experiments defined in config.yaml.
# It ensures the environment is correctly set up and delegates the main logic
# to the Python script 'main.py'.

# --- 1. Error Handling ---
# Exit immediately if a command exits with a non-zero status.
set -e

# --- 2. Navigate to Project Root ---
# Assuming this script is located in 'scripts/' relative to the project root.
# This command changes the current directory to the parent directory of the script's location.
SCRIPT_DIR="$(dirname "$0")"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
echo "Navigating to project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# --- 3. Virtual Environment Activation (Optional) ---
# It's highly recommended to use a virtual environment for dependency management.
# Uncomment and adjust the following lines if you have a virtual environment.

# Example for 'venv' (Python's built-in virtual environment):
# VENV_PATH="./venv"
# if [ -d "$VENV_PATH" ]; then
#     echo "Activating virtual environment: $VENV_PATH"
#     source "$VENV_PATH/bin/activate"
# else
#     echo "Virtual environment not found at $VENV_PATH. Please create and install dependencies, or adjust path."
#     # Optionally, you might want to exit here if the venv is critical.
#     # exit 1
# fi

# Example for 'conda' (if you manage environments with Anaconda/Miniconda):
# CONDA_ENV_NAME="my_peft_env" # Replace with your conda environment name
# if command -v conda &>/dev/null; then
#     echo "Activating conda environment: $CONDA_ENV_NAME"
#     conda activate "$CONDA_ENV_NAME"
# else
#     echo "Conda not found. Please ensure conda is installed and the environment exists, or adjust path."
#     # Optionally, you might want to exit here if the conda env is critical.
#     # exit 1
# fi

# --- 4. Define Configuration File Path ---
# The main configuration file for all experiments.
CONFIG_FILE="config.yaml"

# Check if the config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found in $PROJECT_ROOT."
    exit 1
fi

# --- 5. Execute Main Python Script ---
echo "Starting the PEFT visual recognition study as defined in $CONFIG_FILE..."
echo "Logs will be generated in the directory specified by 'logging.log_dir' in $CONFIG_FILE."

# Run the Python script. The '--config' argument is used by main.py to load the YAML file.
python main.py --config "$CONFIG_FILE"

# --- 6. Completion Message ---
echo "All experiments launched successfully. Please check the 'logs/' directory for detailed outputs and results."

# --- 7. Deactivate Virtual Environment (Optional) ---
# If a virtual environment was activated, it's good practice to deactivate it.
# Uncomment if you used activation in step 3.

# Example for 'venv':
# if [ -d "$VENV_PATH" ]; then
#     echo "Deactivating virtual environment."
#     deactivate
# fi

# Example for 'conda':
# if command -v conda &>/dev/null; then
#     echo "Deactivating conda environment."
#     conda deactivate
# fi


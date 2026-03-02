#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="hab-prediction"                        # conda environment name
REQUIREMENTS="requirements.txt"              # placeholder — swap with your actual yaml

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_SCRIPT="$SCRIPT_DIR/forecast.py"   # placeholder — swap with your script path
PNG_OUTPUT_DIR="$SCRIPT_DIR/PNGOUTPUT"
DEST_DIR="~"

source ~/miniconda3/etc/profile.d/conda.sh

if conda env list | grep -q "^$ENV_NAME "; then
    echo ">>> Environment '$ENV_NAME' already exists, updating..."
    conda activate "$ENV_NAME"
    pip install -r "$REQUIREMENTS"
else
    echo ">>> Creating conda environment: $ENV_NAME"
    # 1. Create the env (just python, no packages)
    conda create --name "$ENV_NAME" python=3.11.7 --yes
    # 2. Activate it
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
    # 3. Install packages from requirements.txt
    pip install -r requirements.txt
fi

# Activate inside the script via conda's shell hook
# eval "$(conda shell.bash hook)"
# conda activate "$ENV_NAME"
# echo ">>> Activated: $CONDA_DEFAULT_ENV"

# echo ">>> Running Python script: $PYTHON_SCRIPT"
# python "$PYTHON_SCRIPT"


echo ">>> Deactivating conda environment"
conda deactivate

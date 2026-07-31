#!/bin/bash
#SBATCH --job-name=setup_env312
#SBATCH --cpus-per-task=4
#SBATCH -p debug
#SBATCH --gres=gpu:1
# #SBATCH --nodelist=hpc[1-3]
# ─────────────────────────────────────────────────────────────────────────────
# Creates a fresh Python 3.12 virtualenv at ~/prism_env312 with all packages
# required by LORE.
# ─────────────────────────────────────────────────────────────────────────────


# Resolve repo root (directory of this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

set -e
ENV_DIR="${LORE_ENV:-$HOME/prism_env312}"

echo "=== Python version ==="
python3 --version
python3.12 --version

echo "=== Creating virtualenv at $ENV_DIR ==="
python3.12 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
python --version

echo "=== Upgrading pip ==="
pip install --upgrade pip setuptools wheel

echo "=== Installing PyTorch (CUDA 11.8) ==="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "=== Installing core scientific packages ==="
pip install \
    numpy \
    scipy \
    scikit-image \
    matplotlib \
    tqdm \
    nibabel \
    SimpleITK \
    connected-components-3d \
    einops \
    monai

echo "=== Installing medical image packages ==="
pip install medpy

echo "=== Installing other utilities ==="
pip install \
    tensorboard \
    pandas \
    Pillow \
    opencv-python-headless \
    pyyaml \
    requests \
    timm

echo "=== Verifying key imports ==="
python -c "
import torch
import numpy as np
import nibabel
import scipy
import skimage
import medpy
import monai
print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('numpy:', np.__version__)
print('All imports OK')
"

echo "=== DONE. New env is at: $ENV_DIR ==="
echo "To use: source $ENV_DIR/bin/activate"

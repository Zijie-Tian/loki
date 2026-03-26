#!/bin/bash
# Initialize conda properly for use in bash scripts
eval "$(conda shell.bash hook)"

echo "Creating the 'compass' environment with python=3.10.19..."
conda create -y -n compass python=3.10.19

echo "Activating 'compass' environment..."
conda activate compass

echo "Installing torch==2.9.1+cu128..."
pip install torch==2.9.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128

echo "Installing transformers, accelerate, datasets, wandb..."
pip install transformers==4.56.0 accelerate==1.12.0 datasets==4.4.2 wandb==0.23.1

echo "Verifying environment contents:"
python -c "import torch, transformers, accelerate, datasets, wandb; print(f'PyTorch: {torch.__version__}'); print(f'Transformers: {transformers.__version__}'); print(f'Accelerate: {accelerate.__version__}')"

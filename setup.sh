#!/bin/bash

set -e

if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed."
    exit 1
fi

echo "Creating conda environment..."
conda create -n gcr python=3.10 -y

conda activate gcr
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
pip install numpy einops pillow
#!/bin/bash
# DR-Saintvision Linux/Mac Setup Script

echo "========================================"
echo "DR-Saintvision Environment Setup"
echo "========================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create conda environment
echo "Creating conda environment..."
conda create -n dr-saintvision python=3.10 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate dr-saintvision

# Install PyTorch with CUDA
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install requirements
echo "Installing required packages..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data logs cache

# Install Ollama (optional)
read -p "Do you want to install Ollama for easier model management? (y/n): " install_ollama
if [ "$install_ollama" = "y" ]; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh

    echo "Downloading models via Ollama..."
    ollama pull mistral:7b-instruct-v0.2-q4_0
    ollama pull llama3.2:latest
    ollama pull qwen2.5:7b-instruct-q4_0
fi

echo "========================================"
echo "Setup completed!"
echo "========================================"
echo ""
echo "To start the system:"
echo "1. Activate environment: conda activate dr-saintvision"
echo "2. Run: python main.py"

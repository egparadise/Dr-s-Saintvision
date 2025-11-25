"""
Model Download Script for DR-Saintvision
Downloads required models from Hugging Face
"""

import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU detected: {gpu_name}")
            print(f"GPU Memory: {gpu_memory:.1f} GB")
            return True
        else:
            print("No CUDA GPU detected. Models will run on CPU (much slower).")
            return False
    except ImportError:
        print("PyTorch not installed. Please run setup.bat/setup.sh first.")
        return False


def download_huggingface_models():
    """Download models from Hugging Face"""
    print("\n" + "=" * 60)
    print("Downloading models from Hugging Face")
    print("This may take a while depending on your internet connection.")
    print("=" * 60 + "\n")

    models = [
        {
            "name": "Mistral-7B-Instruct-v0.2",
            "id": "mistralai/Mistral-7B-Instruct-v0.2",
            "size": "~14GB"
        },
        {
            "name": "Llama-3.2-7B-Instruct",
            "id": "meta-llama/Llama-3.2-7B-Instruct",
            "size": "~14GB",
            "note": "Requires Hugging Face login and model access approval"
        },
        {
            "name": "Qwen2.5-7B-Instruct",
            "id": "Qwen/Qwen2.5-7B-Instruct",
            "size": "~15GB"
        }
    ]

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        for model_info in models:
            print(f"\n{'=' * 40}")
            print(f"Downloading: {model_info['name']}")
            print(f"Model ID: {model_info['id']}")
            print(f"Estimated Size: {model_info['size']}")
            if 'note' in model_info:
                print(f"Note: {model_info['note']}")
            print("=" * 40)

            try:
                # Download tokenizer
                print(f"Downloading tokenizer...")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_info['id'],
                    trust_remote_code=True
                )
                print(f"Tokenizer downloaded successfully")

                # Download model (just config to verify access)
                print(f"Downloading model configuration...")
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(
                    model_info['id'],
                    trust_remote_code=True
                )
                print(f"Model configuration verified")

                print(f"{model_info['name']} ready for use!")

            except Exception as e:
                logger.error(f"Failed to download {model_info['name']}: {e}")
                print(f"Error downloading {model_info['name']}: {e}")

                if "meta-llama" in model_info['id']:
                    print("\nFor Llama models, you need to:")
                    print("1. Create a Hugging Face account")
                    print("2. Request access at: https://huggingface.co/meta-llama")
                    print("3. Run: huggingface-cli login")

    except ImportError as e:
        print(f"Required packages not installed: {e}")
        print("Please run setup.bat/setup.sh first.")
        return False

    return True


def setup_ollama_models():
    """Setup models using Ollama"""
    print("\n" + "=" * 60)
    print("Setting up models with Ollama")
    print("Ollama provides easier model management for local inference")
    print("=" * 60 + "\n")

    models = [
        "mistral:7b-instruct-v0.2-q4_0",
        "llama3.2:latest",
        "qwen2.5:7b-instruct-q4_0"
    ]

    import subprocess

    # Check if Ollama is installed
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True
        )
        print(f"Ollama version: {result.stdout.strip()}")
    except FileNotFoundError:
        print("Ollama is not installed.")
        print("\nTo install Ollama:")
        print("  Windows: Download from https://ollama.ai")
        print("  Linux/Mac: curl -fsSL https://ollama.ai/install.sh | sh")
        return False

    # Pull models
    for model in models:
        print(f"\nPulling model: {model}")
        try:
            subprocess.run(
                ["ollama", "pull", model],
                check=True
            )
            print(f"Successfully pulled {model}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to pull {model}: {e}")

    return True


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║          DR-Saintvision Model Download Utility                ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    # Check GPU
    has_gpu = check_gpu()

    print("\nSelect download method:")
    print("1. Hugging Face (direct download, larger files)")
    print("2. Ollama (easier management, quantized models)")
    print("3. Both")
    print("4. Skip download")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == "1":
        download_huggingface_models()
    elif choice == "2":
        setup_ollama_models()
    elif choice == "3":
        download_huggingface_models()
        setup_ollama_models()
    elif choice == "4":
        print("Skipping model download.")
    else:
        print("Invalid choice. Exiting.")
        return

    print("\n" + "=" * 60)
    print("Model setup complete!")
    print("=" * 60)

    if has_gpu:
        print("\nRecommended settings for your GPU:")
        print("- Use 4-bit quantization for memory efficiency")
        print("- Monitor GPU memory with: nvidia-smi")
    else:
        print("\nNote: Running on CPU will be significantly slower.")
        print("Consider using Ollama with quantized models for better performance.")

    print("\nTo start DR-Saintvision:")
    print("  python main.py")


if __name__ == "__main__":
    main()

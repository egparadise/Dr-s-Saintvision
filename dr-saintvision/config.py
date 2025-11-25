"""
Configuration for DR-Saintvision
Central configuration management
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
CACHE_DIR = PROJECT_ROOT / "cache"

# Create directories if they don't exist
for dir_path in [DATA_DIR, LOGS_DIR, CACHE_DIR]:
    dir_path.mkdir(exist_ok=True)


@dataclass
class ModelConfig:
    """Configuration for AI models"""
    # Model identifiers
    search_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    reasoning_model: str = "meta-llama/Llama-3.2-7B-Instruct"
    synthesis_model: str = "Qwen/Qwen2.5-7B-Instruct"

    # Ollama model names (if using Ollama)
    search_ollama: str = "mistral:7b-instruct-v0.2-q4_0"
    reasoning_ollama: str = "llama3.2:latest"
    synthesis_ollama: str = "qwen2.5:7b-instruct-q4_0"

    # Model settings
    use_quantization: bool = True
    quantization_bits: int = 4
    max_memory: Optional[Dict] = None
    device_map: str = "auto"

    # Generation settings
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.95


@dataclass
class ServerConfig:
    """Configuration for servers"""
    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Gradio server
    gradio_host: str = "0.0.0.0"
    gradio_port: int = 7860
    gradio_share: bool = False

    # Ollama server
    ollama_host: str = "localhost"
    ollama_port: int = 11434


@dataclass
class DatabaseConfig:
    """Configuration for database"""
    db_path: str = str(PROJECT_ROOT / "dr_saintvision.db")
    pool_size: int = 5
    max_overflow: int = 10


@dataclass
class SearchConfig:
    """Configuration for web search"""
    max_results: int = 5
    timeout: float = 30.0
    region: str = "wt-wt"  # Worldwide


@dataclass
class DebateConfig:
    """Configuration for debate system"""
    parallel_analysis: bool = True
    timeout_seconds: float = 300.0
    save_history: bool = True
    max_history: int = 100


@dataclass
class Config:
    """Main configuration class"""
    models: ModelConfig = field(default_factory=ModelConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    debate: DebateConfig = field(default_factory=DebateConfig)

    # Global settings
    use_ollama: bool = False
    debug: bool = False
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables"""
        config = cls()

        # Override with environment variables
        if os.getenv("USE_OLLAMA"):
            config.use_ollama = os.getenv("USE_OLLAMA").lower() == "true"

        if os.getenv("DEBUG"):
            config.debug = os.getenv("DEBUG").lower() == "true"

        if os.getenv("LOG_LEVEL"):
            config.log_level = os.getenv("LOG_LEVEL")

        if os.getenv("API_PORT"):
            config.server.api_port = int(os.getenv("API_PORT"))

        if os.getenv("GRADIO_PORT"):
            config.server.gradio_port = int(os.getenv("GRADIO_PORT"))

        return config


# Default configuration instance
default_config = Config.from_env()


def get_config() -> Config:
    """Get the default configuration"""
    return default_config


# Example .env file content
ENV_TEMPLATE = """
# DR-Saintvision Configuration
# Copy this to .env and modify as needed

# Model Settings
USE_OLLAMA=false

# Server Settings
API_PORT=8000
GRADIO_PORT=7860

# Debug Settings
DEBUG=false
LOG_LEVEL=INFO

# Hugging Face (for model access)
# HF_TOKEN=your_token_here
"""


def create_env_template():
    """Create a template .env file"""
    env_path = PROJECT_ROOT / ".env.template"
    with open(env_path, "w") as f:
        f.write(ENV_TEMPLATE)
    print(f"Created .env template at {env_path}")


if __name__ == "__main__":
    # Print current configuration
    config = get_config()
    print("Current Configuration:")
    print(f"  Use Ollama: {config.use_ollama}")
    print(f"  API Port: {config.server.api_port}")
    print(f"  Gradio Port: {config.server.gradio_port}")
    print(f"  Debug: {config.debug}")
    print(f"\n  Search Model: {config.models.search_model}")
    print(f"  Reasoning Model: {config.models.reasoning_model}")
    print(f"  Synthesis Model: {config.models.synthesis_model}")

    # Create env template
    create_env_template()

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DR-Saintvision is a multi-agent AI debate system where three specialized LLM agents collaborate to answer complex questions:

- **Search Agent (Mistral-7B)**: Web search via DuckDuckGo + RAG-based analysis
- **Reasoning Agent (Llama-3.2-7B)**: Chain-of-thought deep logical reasoning
- **Synthesis Agent (Qwen2.5-7B)**: Combines both analyses into final answer with confidence scoring

The debate flow: Search and Reasoning run in parallel → Synthesis combines their outputs → Final answer with confidence scores.

## Common Commands

```bash
# Run the full system (API + Gradio UI)
python dr-saintvision/main.py

# Run API server only (port 8000)
python dr-saintvision/main.py --api-only

# Run Gradio UI only (port 7860)
python dr-saintvision/main.py --gradio-only

# Run in CLI mode
python dr-saintvision/main.py --cli

# Single query (add --ollama for local inference)
python dr-saintvision/main.py --query "your question here"
python dr-saintvision/main.py --query "your question" --ollama

# Run tests
pytest dr-saintvision/tests/ -v

# Run single test file
pytest dr-saintvision/tests/test_agents.py -v

# Run with coverage
pytest dr-saintvision/tests/ --cov=. --cov-report=html

# Download HuggingFace models (alternative to Ollama)
python dr-saintvision/download_models.py
```

## Architecture

```
dr-saintvision/
├── models/           # AI agent implementations
│   ├── base_agent.py       # Abstract base class (HuggingFace + 4-bit quantization)
│   ├── search_agent.py     # DuckDuckGo search + analysis
│   ├── reasoning_agent.py  # Chain-of-thought reasoning
│   ├── synthesis_agent.py  # Multi-perspective synthesis
│   └── debate_manager.py   # Orchestrates parallel agent execution
├── backend/
│   ├── api.py              # FastAPI REST endpoints
│   ├── database.py         # SQLite via SQLAlchemy
│   └── evaluation.py       # Answer quality metrics
├── frontend/
│   └── app.py              # Gradio web interface
├── utils/
│   ├── web_search.py       # DuckDuckGo search wrapper
│   ├── metrics.py          # Evaluation metrics
│   └── prompts.py          # Prompt templates
├── research/               # Benchmarking and experiments
│   ├── algorithms.py       # Algorithm implementations
│   ├── benchmarks.py       # Performance benchmarks
│   ├── experiments.py      # Experiment runners
│   └── comparison.py       # Single vs debate comparison
├── config.py               # Centralized configuration (dataclasses + env vars)
└── main.py                 # Entry point with CLI argument parsing
```

## Key Patterns

### Agent System
- All agents inherit from `BaseAgent` (`models/base_agent.py:15`) which handles model loading, 4-bit quantization via bitsandbytes, and generation
- Agents support both HuggingFace Transformers and Ollama backends (`use_ollama` flag)
- Agent `process()` methods are async and return dictionaries with results and confidence scores
- Use `unload_model()` to free GPU memory when switching models

### Debate Flow
`DebateManager.conduct_debate()` (`models/debate_manager.py:96`) orchestrates:
1. Parallel execution via `asyncio.gather()` for Search + Reasoning agents
2. Sequential Synthesis phase that receives both outputs
3. Weighted confidence calculation (Search: 25%, Reasoning: 30%, Synthesis: 45%)
4. Status callbacks via `on_status_change` for UI updates

### Configuration
- `config.py` uses dataclasses (`ModelConfig`, `ServerConfig`, `DatabaseConfig`, `DebateConfig`)
- Environment variables override defaults via `.env` file
- Key env vars: `USE_OLLAMA`, `API_PORT`, `GRADIO_PORT`, `DEBUG`, `LOG_LEVEL`
- Use `Config.from_env()` to load configuration

### Ollama Integration
Agents use `httpx.AsyncClient` to communicate with Ollama API at `localhost:11434`. The `_generate_with_ollama()` method in each agent handles this.

## Model Configuration

Ollama models (recommended for local use, ~4GB VRAM each):
- `mistral:7b-instruct-v0.2-q4_0`
- `llama3.2:latest`
- `qwen2.5:7b-instruct-q4_0`

HuggingFace models:
- `mistralai/Mistral-7B-Instruct-v0.2`
- `meta-llama/Llama-3.2-7B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze` | POST | Full multi-agent debate |
| `/quick` | POST | Quick single-pass analysis |
| `/debate/{query_id}` | GET | Retrieve debate details |
| `/compare` | POST | Compare single-agent vs debate |
| `/stats` | GET | System statistics |
| `/history/{user_id}` | GET | User debate history |
| `/search` | GET | Search past debates |
| `/health` | GET | Health check |

API docs available at `http://localhost:8000/docs` (Swagger) and `/redoc`.

## Testing

Tests use pytest with mocking to avoid loading actual models:
```python
with patch.object(SearchAgent, 'load_model'):
    agent = SearchAgent(use_ollama=True)

# For DebateManager, mock all three agents:
with patch('models.debate_manager.SearchAgent'), \
     patch('models.debate_manager.ReasoningAgent'), \
     patch('models.debate_manager.SynthesisAgent'):
    manager = DebateManager(config=DebateConfig(use_ollama=True))
```

## System Requirements

- **Minimum**: NVIDIA RTX 3060 (12GB VRAM), 16GB RAM, Python 3.10+
- **Recommended**: NVIDIA RTX 4070+ (12GB+ VRAM), 32GB RAM, 50GB storage for models

## Troubleshooting

- **CUDA OOM**: Enable quantization in `.env` with `USE_QUANTIZATION=true`
- **Ollama connection**: Ensure `ollama serve` is running and models are pulled
- **HuggingFace access**: Run `huggingface-cli login` for gated models

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DR-Saintvision is a multi-agent AI debate system where three specialized LLM agents collaborate to answer complex questions:

- **Search Agent (Mistral-7B)**: Performs web searches and RAG-based analysis
- **Reasoning Agent (Llama-3.2-7B)**: Conducts deep logical reasoning with chain-of-thought
- **Synthesis Agent (Qwen2.5-7B)**: Combines analyses from both agents into a final answer with confidence scoring

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

# Single query
python dr-saintvision/main.py --query "your question here"

# Run tests
pytest dr-saintvision/tests/ -v

# Run single test file
pytest dr-saintvision/tests/test_agents.py -v

# Run with coverage
pytest dr-saintvision/tests/ --cov=. --cov-report=html
```

## Architecture

```
dr-saintvision/
├── models/           # AI agent implementations
│   ├── base_agent.py       # Abstract base class for all agents (HuggingFace + quantization)
│   ├── search_agent.py     # Web search + DuckDuckGo + RAG
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
└── config.py               # Centralized configuration (dataclasses + env vars)
```

## Key Patterns

### Agent System
- All agents inherit from `BaseAgent` which handles model loading, quantization (4-bit via bitsandbytes), and generation
- Agents support both HuggingFace Transformers and Ollama backends (`use_ollama` flag)
- Agent `process()` methods are async and return dictionaries with results and confidence scores

### Debate Flow
`DebateManager.conduct_debate()` orchestrates:
1. Parallel execution via `asyncio.gather()` for Search + Reasoning agents
2. Sequential Synthesis phase that receives both outputs
3. Weighted confidence calculation (Search: 25%, Reasoning: 30%, Synthesis: 45%)

### Configuration
- `config.py` uses dataclasses (`ModelConfig`, `ServerConfig`, `DatabaseConfig`, etc.)
- Environment variables override defaults via `.env` file
- Key env vars: `USE_OLLAMA`, `API_PORT`, `GRADIO_PORT`, `DEBUG`

## Model Configuration

Ollama models (recommended for local use):
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

## Testing

Tests use pytest with mocking to avoid loading actual models:
```python
with patch.object(SearchAgent, 'load_model'):
    agent = SearchAgent(use_ollama=True)
```

The `tests/` directory mirrors the main structure with `test_agents.py`, `test_api.py`, and `test_utils.py`.

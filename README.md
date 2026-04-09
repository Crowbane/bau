# BAU

A next-generation autonomous agent that plans, executes, remembers, and forges its own tools — all running locally, model-agnostic, in a polished terminal UI.

## Install

```bash
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Run

```bash
export ANTHROPIC_API_KEY=sk-...   # or OPENAI_API_KEY, etc.
python main.py
```

Edit `config.yaml` to switch providers (Anthropic, OpenAI, Ollama, vLLM, LM Studio).

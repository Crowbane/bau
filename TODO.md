# TODO

## Recommendations

- [ ] **Prompt caching** — Wire `cache_control` parameter through `LLMClient.complete()` in `llm.py`. LiteLLM already supports it; the wrapper just doesn't expose it yet.

- [ ] **Streaming token counter** — Add a live token usage indicator to the UI in `ui.py`. Event hooks are already in place; needs a running counter rendered in the display.

## Optional packages (from ARCHITECTURE.md)

- [ ] **flashrank** (v0.2.10, reranking) — Already used in `agent.py:579` with a graceful `ImportError` fallback, but not listed in `requirements.txt`. Add to requirements to make reranking active by default.

- [ ] **instructor** (v1.15.1, structured output) — Not used anywhere. Could replace raw JSON-mode LLM calls (e.g. planning, tool creation) with Pydantic-validated structured output for more reliable parsing.

"""LLM client — LiteLLM wrapper with retries, streaming, cost tracking."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, AsyncIterator


@dataclass(slots=True)
class CompletionResult:
    """Structured result from an LLM completion call."""

    text: str
    tool_calls: list[dict] | None
    usage: dict[str, int]
    cost: float
    raw: Any


@dataclass(slots=True)
class StreamChunk:
    """A single chunk from a streaming LLM response."""

    text: str
    tool_calls: list[dict] | None
    done: bool
    raw: Any


class LLMClient:
    """Model-agnostic LLM client wrapping LiteLLM.

    Provides a unified interface for OpenAI, Anthropic, Ollama, vLLM,
    llama.cpp, and LM Studio via LiteLLM's completion API.

    Args:
        config: Parsed config dict (from config.yaml).
    """

    def __init__(self, config: dict) -> None:
        # Lazy import — LiteLLM is slow to load at module level
        import litellm

        self._litellm = litellm
        litellm.drop_params = True  # silently drop unsupported params

        model_cfg = config["model"]
        self._provider = model_cfg["provider"]
        self._model_name = model_cfg["name"]
        self._api_base = model_cfg.get("api_base")
        self._api_key_env = model_cfg.get("api_key_env")
        self._max_tokens = config.get("limits", {}).get("max_tokens_per_call", 4096)

        # Resolve model string for LiteLLM
        self._model = self._build_model_string(self._provider, self._model_name)

        # Optional planner model override
        planner_cfg = config.get("planner_model")
        self._planner_model = (
            self._build_model_string(planner_cfg["provider"], planner_cfg["name"])
            if planner_cfg
            else self._model
        )

        # Cumulative cost / usage tracking
        self._total_cost: float = 0.0
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0

        # Resolve API key from env
        self._api_key: str | None = None
        if self._api_key_env:
            self._api_key = os.environ.get(self._api_key_env)

        # Fallback router (optional)
        self._router = self._build_router(config) if config.get("fallback_models") else None

    # ------------------------------------------------------------------
    # Model string helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_model_string(provider: str, name: str) -> str:
        """Build a LiteLLM model identifier from provider + name.

        Args:
            provider: Provider key from config (e.g. 'anthropic', 'ollama').
            name: Model name (e.g. 'claude-sonnet-4-20250514').

        Returns:
            LiteLLM-compatible model string.
        """
        match provider:
            case "anthropic":
                return f"anthropic/{name}"
            case "openai":
                return f"openai/{name}"
            case "ollama":
                return f"ollama_chat/{name}"
            case "openai_compat" | "vllm" | "lmstudio" | "llamacpp":
                return f"openai/{name}"
            case _:
                return f"{provider}/{name}"

    def _build_router(self, config: dict):
        """Build a LiteLLM Router with fallback chain.

        Args:
            config: Full config dict containing 'fallback_models' list.

        Returns:
            Configured litellm.Router instance.
        """
        primary = {
            "model_name": "primary",
            "litellm_params": {
                "model": self._model,
                **({"api_base": self._api_base} if self._api_base else {}),
                **({"api_key": self._api_key} if self._api_key else {}),
            },
        }
        model_list = [primary]
        fallbacks_list = []

        for i, fb in enumerate(config["fallback_models"]):
            fb_name = f"fallback_{i}"
            model_list.append({
                "model_name": fb_name,
                "litellm_params": {
                    "model": self._build_model_string(fb["provider"], fb["name"]),
                    **({"api_base": fb.get("api_base")} if fb.get("api_base") else {}),
                },
            })
            fallbacks_list.append({"primary": [fb_name]})

        return self._litellm.Router(model_list=model_list, fallbacks=fallbacks_list)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_kwargs(self, model: str | None = None, **extra: Any) -> dict[str, Any]:
        """Assemble keyword arguments for a LiteLLM call.

        Args:
            model: Override model string.
            **extra: Additional params forwarded to LiteLLM.

        Returns:
            Dict of kwargs ready for acompletion().
        """
        kwargs: dict[str, Any] = {
            "model": model or self._model,
            "max_tokens": extra.pop("max_tokens", self._max_tokens),
        }
        if self._api_base and "api_base" not in extra:
            kwargs["api_base"] = self._api_base
        if self._api_key and "api_key" not in extra:
            kwargs["api_key"] = self._api_key
        kwargs.update(extra)
        return kwargs

    def _parse_response(self, response: Any) -> CompletionResult:
        """Extract a structured CompletionResult from a LiteLLM response.

        Args:
            response: Raw LiteLLM ModelResponse.

        Returns:
            Parsed CompletionResult.
        """
        choice = response.choices[0]
        message = choice.message

        # Extract tool calls, normalising arguments to dicts
        tool_calls = None
        if message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    args = json.loads(args)
                tool_calls.append({
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": args,
                    },
                })

        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        try:
            cost = self._litellm.completion_cost(response)
        except Exception:
            cost = 0.0

        self._total_cost += cost
        self._total_prompt_tokens += usage["prompt_tokens"]
        self._total_completion_tokens += usage["completion_tokens"]

        return CompletionResult(
            text=message.content or "",
            tool_calls=tool_calls,
            usage=usage,
            cost=cost,
            raw=response,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> CompletionResult:
        """Send a completion request and return a structured result.

        Args:
            messages: Chat messages in OpenAI format.
            tools: Tool definitions in OpenAI JSON Schema format.
            model: Override model string (e.g. for planner calls).
            **kwargs: Extra params passed through to LiteLLM.

        Returns:
            CompletionResult with text, tool_calls, usage, cost, raw.

        Raises:
            litellm.RateLimitError: After 3 retries on rate limits.
            litellm.APIConnectionError: After 3 retries on connection failures.
        """
        from tenacity import (
            retry,
            retry_if_exception_type,
            stop_after_attempt,
            wait_exponential,
        )

        call_kwargs = self._call_kwargs(model=model, **kwargs)
        call_kwargs["messages"] = messages
        if tools:
            call_kwargs["tools"] = tools

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=16),
            retry=retry_if_exception_type((
                self._litellm.RateLimitError,
                self._litellm.APIConnectionError,
            )),
            reraise=True,
        )
        async def _call() -> Any:
            if self._router:
                routed = {**call_kwargs, "model": "primary"}
                return await self._router.acompletion(**routed)
            return await self._litellm.acompletion(**call_kwargs)

        response = await _call()
        return self._parse_response(response)

    async def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a completion response chunk by chunk.

        Args:
            messages: Chat messages in OpenAI format.
            tools: Tool definitions in OpenAI JSON Schema format.
            model: Override model string.
            **kwargs: Extra params passed through to LiteLLM.

        Yields:
            StreamChunk for each piece of the response.
        """
        call_kwargs = self._call_kwargs(model=model, **kwargs)
        call_kwargs["messages"] = messages
        call_kwargs["stream"] = True
        if tools:
            call_kwargs["tools"] = tools

        response = await self._litellm.acompletion(**call_kwargs)

        collected_text = ""
        collected_tool_calls: list[dict] = []

        async for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue

            chunk_text = delta.content or ""
            collected_text += chunk_text

            # Accumulate streamed tool-call deltas
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    while len(collected_tool_calls) <= idx:
                        collected_tool_calls.append({
                            "id": "",
                            "function": {"name": "", "arguments": ""},
                        })
                    entry = collected_tool_calls[idx]
                    if tc_delta.id:
                        entry["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            entry["function"]["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            entry["function"]["arguments"] += tc_delta.function.arguments

            done = chunk.choices[0].finish_reason is not None
            yield StreamChunk(
                text=chunk_text,
                tool_calls=collected_tool_calls if done and collected_tool_calls else None,
                done=done,
                raw=chunk,
            )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text. Uses tiktoken for OpenAI-family models, else len//4.

        Args:
            text: The text to count tokens for.

        Returns:
            Estimated token count.
        """
        try:
            import tiktoken

            enc = tiktoken.encoding_for_model(self._model_name)
            return len(enc.encode(text))
        except Exception:
            return len(text) // 4

    def get_cost(self) -> dict[str, float | int]:
        """Return cumulative cost and token usage across all calls.

        Returns:
            Dict with total_cost, prompt_tokens, completion_tokens.
        """
        return {
            "total_cost": self._total_cost,
            "prompt_tokens": self._total_prompt_tokens,
            "completion_tokens": self._total_completion_tokens,
        }

    def supports_tools(self, model: str | None = None) -> bool:
        """Check whether the configured model supports tool/function calling.

        Args:
            model: Model string to check. Defaults to the configured model.

        Returns:
            True if the model supports tool calling.
        """
        target = model or self._model
        try:
            return self._litellm.supports_function_calling(model=target)
        except Exception:
            return False

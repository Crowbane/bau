"""Phase 1 smoke tests — config, prompts, LLMClient (no real API calls)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import load_config, load_prompts, render_prompt


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def sample_config() -> dict:
    return {
        "model": {
            "provider": "anthropic",
            "name": "claude-sonnet-4-20250514",
            "api_key_env": "ANTHROPIC_API_KEY",
        },
        "limits": {
            "max_tokens_per_call": 4096,
        },
    }


def _mock_response(text: str = "BAU online.") -> MagicMock:
    """Build a fake LiteLLM ModelResponse."""
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = text
    choice.message.tool_calls = None
    response.choices = [choice]
    response.usage.prompt_tokens = 100
    response.usage.completion_tokens = 10
    response.usage.total_tokens = 110
    return response


# ------------------------------------------------------------------
# LLMClient tests
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_complete_returns_expected_text(sample_config: dict) -> None:
    mock_resp = _mock_response("BAU online.")

    with (
        patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp),
        patch("litellm.completion_cost", return_value=0.001),
    ):
        from llm import LLMClient

        client = LLMClient(sample_config)
        result = await client.complete([{"role": "user", "content": "Hello"}])

    assert result.text == "BAU online."
    assert result.cost == pytest.approx(0.001)
    assert result.tool_calls is None


@pytest.mark.asyncio
async def test_cost_tracking_accumulates(sample_config: dict) -> None:
    mock_resp = _mock_response("ok")

    with (
        patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp),
        patch("litellm.completion_cost", return_value=0.002),
    ):
        from llm import LLMClient

        client = LLMClient(sample_config)
        await client.complete([{"role": "user", "content": "1"}])
        await client.complete([{"role": "user", "content": "2"}])

    cost = client.get_cost()
    assert cost["total_cost"] == pytest.approx(0.004)
    assert cost["prompt_tokens"] == 200
    assert cost["completion_tokens"] == 20


@pytest.mark.asyncio
async def test_tool_calls_parsed(sample_config: dict) -> None:
    mock_resp = _mock_response("")
    tc = MagicMock()
    tc.id = "call_123"
    tc.function.name = "web_search"
    tc.function.arguments = '{"query": "hello"}'
    mock_resp.choices[0].message.tool_calls = [tc]
    mock_resp.choices[0].message.content = None

    with (
        patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp),
        patch("litellm.completion_cost", return_value=0.0),
    ):
        from llm import LLMClient

        client = LLMClient(sample_config)
        result = await client.complete([{"role": "user", "content": "search"}])

    assert result.tool_calls is not None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["function"]["name"] == "web_search"
    assert result.tool_calls[0]["function"]["arguments"] == {"query": "hello"}


@pytest.mark.asyncio
async def test_supports_tools(sample_config: dict) -> None:
    with patch("litellm.supports_function_calling", return_value=True):
        from llm import LLMClient

        client = LLMClient(sample_config)
        assert client.supports_tools() is True


@pytest.mark.asyncio
async def test_count_tokens_fallback(sample_config: dict) -> None:
    """Non-OpenAI models fall back to len//4."""
    with patch("litellm.supports_function_calling", return_value=False):
        from llm import LLMClient

        client = LLMClient(sample_config)
        count = client.count_tokens("a" * 400)
        assert count == 100


# ------------------------------------------------------------------
# Config loading tests
# ------------------------------------------------------------------

def test_config_loads_from_file() -> None:
    config = load_config("config.yaml")
    assert "model" in config
    assert config["model"]["provider"] in ("anthropic", "openai", "ollama", "openai_compat")
    assert "name" in config["model"]


def test_config_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent.yaml")


def test_config_missing_model_key(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("limits:\n  max_iterations: 30\n")
    with pytest.raises(ValueError, match="missing required 'model'"):
        load_config(str(bad))


def test_config_missing_provider(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("model:\n  name: gpt-4o\n")
    with pytest.raises(ValueError, match="'provider'"):
        load_config(str(bad))


def test_config_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_config(str(empty))


# ------------------------------------------------------------------
# Prompt loading tests
# ------------------------------------------------------------------

def test_prompts_load_all_four() -> None:
    prompts = load_prompts("prompts/")
    for name in ("system", "planner", "critic", "tool_creator"):
        assert name in prompts, f"Missing prompt: {name}"
        assert len(prompts[name]) > 0


def test_prompts_missing_directory() -> None:
    with pytest.raises(FileNotFoundError):
        load_prompts("nonexistent_dir/")


def test_render_prompt_substitutes_placeholders() -> None:
    prompts = load_prompts("prompts/")
    rendered = render_prompt(
        prompts["system"],
        tools="web_search, file_read",
        date="2026-04-09",
        memory_block="User prefers concise answers.",
    )
    assert "{{tools}}" not in rendered
    assert "{{date}}" not in rendered
    assert "{{memory_block}}" not in rendered
    assert "web_search, file_read" in rendered
    assert "2026-04-09" in rendered
    assert "User prefers concise answers." in rendered

"""Phase 3 smoke tests — Agent loop with stub LLM and tools."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import Agent, AgentState, _cosine_similarity
from llm import CompletionResult
from tools import ToolRegistry


# ------------------------------------------------------------------
# Stubs
# ------------------------------------------------------------------

def _resp(text: str = "", tool_calls: list[dict] | None = None) -> CompletionResult:
    """Build a canned CompletionResult."""
    return CompletionResult(
        text=text,
        tool_calls=tool_calls,
        usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        cost=0.001,
        raw=None,
    )


def _tc(name: str, args: dict, call_id: str = "call_1") -> list[dict]:
    """Build a single-element tool_calls list."""
    return [{"id": call_id, "function": {"name": name, "arguments": args}}]


class StubLLM:
    """Returns canned responses in order."""

    def __init__(self, responses: list[CompletionResult]) -> None:
        self._responses = list(responses)
        self._index = 0

    async def complete(self, messages: list[dict], **kwargs) -> CompletionResult:
        if self._index >= len(self._responses):
            return _resp(text="(no more canned responses)")
        r = self._responses[self._index]
        self._index += 1
        return r

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    def get_cost(self) -> dict:
        return {"total_cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0}


class StubMemory:
    """In-memory stand-in for AgentMemory (no SQLite/FastEmbed needed)."""

    def __init__(self) -> None:
        self._checkpoints: list[dict] = []

    def checkpoint(self, state: dict, iteration: int) -> None:
        self._checkpoints.append({"state": state, "iteration": iteration})

    def latest_checkpoint(self) -> dict | None:
        return self._checkpoints[-1]["state"] if self._checkpoints else None

    def core_render(self) -> str:
        return "(test core memory)"

    def _embed(self, text: str) -> list[float]:
        raise NotImplementedError("no embedding in stub")


def _make_registry() -> ToolRegistry:
    """Register echo and add stub tools."""
    reg = ToolRegistry()

    async def echo(text: str) -> str:
        """Echo text back.

        Args:
            text: Text to echo.
        """
        return text

    async def add(a: float, b: float) -> float:
        """Add two numbers.

        Args:
            a: First number.
            b: Second number.
        """
        return a + b

    reg.register(echo)
    reg.register(add)
    return reg


def _make_agent(
    responses: list[CompletionResult],
    config_overrides: dict | None = None,
) -> tuple[Agent, StubMemory, list[tuple[str, dict]]]:
    """Wire up an Agent with stubs and an event collector."""
    llm = StubLLM(responses)
    memory = StubMemory()
    tools = _make_registry()
    prompts = {"system": "You are BAU.", "planner": "Plan the goal."}
    config: dict = {"limits": {"max_iterations": 30, "max_inner_iterations": 8}}
    if config_overrides:
        for k, v in config_overrides.items():
            if isinstance(v, dict) and k in config:
                config[k].update(v)
            else:
                config[k] = v

    events: list[tuple[str, dict]] = []
    agent = Agent(
        llm, memory, tools, prompts, config,
        on_event=lambda t, p: events.append((t, p)),
    )
    return agent, memory, events


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def test_simple_end_to_end():
    """A 2-step plan executes end-to-end and produces the expected answer."""

    async def _run():
        responses = [
            # _plan
            _resp(text='{"steps": ["echo hello", "add 2+3"]}'),
            # step 0 react: tool call then text
            _resp(tool_calls=_tc("echo", {"text": "hello"})),
            _resp(text="echoed hello"),
            # step 1 react: tool call then text
            _resp(tool_calls=_tc("add", {"a": 2, "b": 3}, "call_2")),
            _resp(text="result is 5"),
            # _synthesize
            _resp(text="Done: echoed hello and computed 2+3=5"),
        ]
        agent, memory, events = _make_agent(responses)
        answer = await agent.run("echo hello and add 2+3")

        assert "Done" in answer
        types = [e[0] for e in events]
        assert types[0] == "goal"
        assert types[1] == "plan"
        assert "step_start" in types
        assert "tool_call" in types
        assert "tool_result" in types
        assert "step_done" in types
        assert types[-1] == "done"
        assert events[-1][1]["status"] == "done"
        # Two completed steps → two checkpoints
        assert len(memory._checkpoints) == 2

    asyncio.run(_run())


def test_loop_detection():
    """Repeated tool calls trigger nudge at 3 and force replan at 5."""

    async def _run():
        repeated = _tc("echo", {"text": "same"})
        responses = [
            _resp(text='{"steps": ["do something"]}'),
            # 5 identical tool calls → replan forced on 5th
            _resp(tool_calls=repeated),
            _resp(tool_calls=repeated),
            _resp(tool_calls=repeated),   # nudge at count 3
            _resp(tool_calls=repeated),
            _resp(tool_calls=repeated),   # replan at count 5
            # _replan
            _resp(text='{"steps": ["try differently"]}'),
            # new step completes
            _resp(text="done differently"),
            # _synthesize
            _resp(text="Completed after replan"),
        ]
        agent, memory, events = _make_agent(responses)
        answer = await agent.run("do something")

        assert "Completed" in answer
        types = [e[0] for e in events]
        assert "replan" in types
        # Verify nudge was injected in tool results
        tool_results = [e[1]["result"] for e in events if e[0] == "tool_result"]
        assert any("repeated the same action" in r for r in tool_results)

    asyncio.run(_run())


def test_iteration_cap_aborts():
    """Exceeding max_iterations aborts the run with status 'aborted'."""

    async def _run():
        responses = [
            # Plan with 5 steps — but max_iterations=2
            _resp(text='{"steps": ["s1", "s2", "s3", "s4", "s5"]}'),
            _resp(text="done s1"),
            _resp(text="done s2"),
            # s3 never runs — cap hit
            _resp(text="Aborted"),  # _synthesize
        ]
        agent, memory, events = _make_agent(
            responses, {"limits": {"max_iterations": 2}},
        )
        answer = await agent.run("five things")

        done_ev = [e for e in events if e[0] == "done"][0]
        assert done_ev[1]["status"] == "aborted"
        assert len(memory._checkpoints) == 2

    asyncio.run(_run())


def test_hallucinated_tool():
    """A non-existent tool name returns an error observation, not a crash."""

    async def _run():
        responses = [
            _resp(text='{"steps": ["use a tool"]}'),
            # Hallucinated tool
            _resp(tool_calls=_tc("nonexistent_tool", {"x": 1})),
            # LLM recovers and uses a real tool
            _resp(tool_calls=_tc("echo", {"text": "recovered"})),
            _resp(text="recovered from hallucination"),
            _resp(text="Done with recovery"),  # _synthesize
        ]
        agent, memory, events = _make_agent(responses)
        answer = await agent.run("use a tool")

        assert "Done" in answer
        tool_results = [e for e in events if e[0] == "tool_result"]
        assert any("Unknown tool" in e[1]["result"] for e in tool_results)

    asyncio.run(_run())


def test_replan_on_inner_cap():
    """When inner iteration cap is hit, the agent replans."""

    async def _run():
        responses = [
            _resp(text='{"steps": ["complex step"]}'),
            # Inner cap = 1: one tool call then cap triggers replan
            _resp(tool_calls=_tc("echo", {"text": "trying"})),
            # _replan
            _resp(text='{"steps": ["simpler step"]}'),
            # New step completes
            _resp(text="done simply"),
            _resp(text="Completed after replan"),  # _synthesize
        ]
        agent, memory, events = _make_agent(
            responses, {"limits": {"max_inner_iterations": 1}},
        )
        answer = await agent.run("do something complex")

        types = [e[0] for e in events]
        assert "replan" in types
        done_ev = [e for e in events if e[0] == "done"][0]
        assert done_ev[1]["status"] == "done"

    asyncio.run(_run())


def test_checkpointing():
    """Checkpoints are written after each completed step with correct iteration."""

    async def _run():
        responses = [
            _resp(text='{"steps": ["s1", "s2", "s3"]}'),
            _resp(text="done s1"),
            _resp(text="done s2"),
            _resp(text="done s3"),
            _resp(text="All three done"),  # _synthesize
        ]
        agent, memory, events = _make_agent(responses)
        await agent.run("three steps")

        assert len(memory._checkpoints) == 3
        for i, cp in enumerate(memory._checkpoints):
            assert cp["iteration"] == i
            assert cp["state"]["status"] == "running"

    asyncio.run(_run())


def test_no_exceptions_escape_run():
    """Even with a broken LLM, run() returns a string without raising."""

    async def _run():
        class BrokenLLM:
            async def complete(self, messages, **kwargs):
                raise RuntimeError("LLM is broken")

            def count_tokens(self, text):
                return 0

            def get_cost(self):
                return {"total_cost": 0.0}

        memory = StubMemory()
        tools = _make_registry()
        events: list[tuple[str, dict]] = []
        agent = Agent(
            BrokenLLM(), memory, tools,
            {"system": "BAU", "planner": "Plan"},
            {},
            on_event=lambda t, p: events.append((t, p)),
        )
        answer = await agent.run("do something")

        assert isinstance(answer, str)
        done_ev = [e for e in events if e[0] == "done"][0]
        assert done_ev[1]["status"] == "failed"

    asyncio.run(_run())


def test_events_order():
    """Events are emitted in the correct sequence for a 1-step plan."""

    async def _run():
        responses = [
            _resp(text='{"steps": ["greet"]}'),
            _resp(tool_calls=_tc("echo", {"text": "hi"})),
            _resp(text="greeted"),
            _resp(text="Final answer"),  # _synthesize
        ]
        agent, _, events = _make_agent(responses)
        await agent.run("greet")

        types = [e[0] for e in events]
        # Mandatory ordering: goal → plan → step_start → ... → step_done → done
        assert types.index("goal") < types.index("plan")
        assert types.index("plan") < types.index("step_start")
        assert types.index("step_start") < types.index("tool_call")
        assert types.index("tool_call") < types.index("tool_result")
        assert types.index("step_done") < types.index("done")

    asyncio.run(_run())


def test_cosine_similarity_helper():
    """Sanity check the cosine helper."""
    assert _cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)
    assert _cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)
    assert _cosine_similarity([0, 0, 0], [1, 0, 0]) == pytest.approx(0.0)


def test_parse_plan_json():
    """Plan parser handles JSON format."""
    assert Agent._parse_plan('{"steps": ["a", "b"]}') == ["a", "b"]
    assert Agent._parse_plan('["x", "y"]') == ["x", "y"]


def test_parse_plan_numbered():
    """Plan parser falls back to numbered list format."""
    text = "1. First step\n2. Second step\n3. Third step"
    assert Agent._parse_plan(text) == ["First step", "Second step", "Third step"]

"""Phase 5 smoke tests — TUI event rendering, slash commands, input stubs."""
from __future__ import annotations

import asyncio

import pytest

from ui import AgentUI, parse_slash_command


def _run(coro):
    """Helper to run async code in tests."""
    return asyncio.run(coro)


@pytest.fixture
def ui():
    """Create an AgentUI with default config."""
    config = {
        "model": {"provider": "test", "name": "stub"},
        "ui": {"theme": "dark", "show_token_usage": True},
    }
    return AgentUI(config)


# ------------------------------------------------------------------
# Slash command parsing
# ------------------------------------------------------------------

class TestSlashCommandParsing:

    def test_quit(self):
        cmd, args = parse_slash_command("/quit")
        assert cmd == "/quit"
        assert args == ""

    def test_exit(self):
        cmd, args = parse_slash_command("/exit")
        assert cmd == "/exit"
        assert args == ""

    def test_stats(self):
        cmd, args = parse_slash_command("/stats")
        assert cmd == "/stats"

    def test_memory_with_query(self):
        cmd, args = parse_slash_command("/memory what is Python")
        assert cmd == "/memory"
        assert args == "what is Python"

    def test_tools(self):
        cmd, args = parse_slash_command("/tools")
        assert cmd == "/tools"

    def test_clear(self):
        cmd, args = parse_slash_command("/clear")
        assert cmd == "/clear"

    def test_help(self):
        cmd, args = parse_slash_command("/help")
        assert cmd == "/help"

    def test_not_a_command(self):
        cmd, args = parse_slash_command("find all Python files")
        assert cmd == ""
        assert args == "find all Python files"

    def test_empty_input(self):
        cmd, args = parse_slash_command("")
        assert cmd == ""
        assert args == ""

    def test_case_insensitive(self):
        cmd, _ = parse_slash_command("/QUIT")
        assert cmd == "/quit"

    def test_whitespace(self):
        cmd, args = parse_slash_command("  /stats  ")
        assert cmd == "/stats"


# ------------------------------------------------------------------
# Event rendering (no crash on any event type)
# ------------------------------------------------------------------

class TestOnEvent:
    """Verify on_event handles all documented event types without crashing."""

    def test_goal(self, ui):
        ui.on_event("goal", {"goal": "Find the meaning of life"})

    def test_plan(self, ui):
        ui.on_event("plan", {"steps": ["step 1", "step 2", "step 3"]})

    def test_replan(self, ui):
        ui.on_event("replan", {"reason": "repeated action detected"})

    def test_step_start(self, ui):
        ui.on_event("step_start", {"index": 0, "description": "do thing", "total": 3})

    def test_step_done(self, ui):
        ui.on_event("step_done", {"index": 0, "result": {"text": "success"}})

    def test_thinking(self, ui):
        ui.on_event("thinking", {"text": "hmm let me think about this..."})

    def test_thinking_long(self, ui):
        ui.on_event("thinking", {"text": "x" * 1000})

    def test_tool_call(self, ui):
        ui.on_event("tool_call", {"name": "web_search", "args": {"query": "python"}})

    def test_tool_result_text(self, ui):
        ui.on_event("tool_result", {"name": "web_search", "result": "some text result"})

    def test_tool_result_json(self, ui):
        ui.on_event("tool_result", {
            "name": "web_search",
            "result": '{"title": "Python", "url": "https://python.org"}',
        })

    def test_tool_result_long(self, ui):
        ui.on_event("tool_result", {"name": "file_read", "result": "x" * 5000})

    def test_memory_op(self, ui):
        ui.on_event("memory_op", {"op": "store", "summary": "user likes Python"})

    def test_warning(self, ui):
        ui.on_event("warning", {"message": "approaching iteration cap"})

    def test_error(self, ui):
        ui.on_event("error", {"message": "LLM connection failed"})

    def test_done_success(self, ui):
        ui.on_event("done", {"answer": "The answer is 42.", "status": "done"})

    def test_done_failed(self, ui):
        ui.on_event("done", {"answer": "Task failed.", "status": "failed"})

    def test_unknown_event(self, ui):
        """Unknown events should render without crashing."""
        ui.on_event("custom_event", {"data": "hello"})

    def test_empty_payload(self, ui):
        """Events with missing keys should not crash."""
        ui.on_event("tool_call", {})
        ui.on_event("plan", {})
        ui.on_event("step_start", {})


# ------------------------------------------------------------------
# Stats rendering
# ------------------------------------------------------------------

class TestStatsRendering:

    def test_show_stats(self, ui):
        stats = {
            "llm": {
                "total_cost": 0.0042,
                "prompt_tokens": 1500,
                "completion_tokens": 800,
            },
            "memory": {
                "counts": {
                    "memories": 10,
                    "conversation": 25,
                    "checkpoints": 3,
                    "core_memory": 2,
                },
                "db_size_bytes": 65536,
            },
            "tools": {
                "web_search": {"usage": 5, "success": 4, "failure": 1, "builtin": True},
                "file_read": {"usage": 3, "success": 3, "failure": 0, "builtin": True},
            },
        }
        ui.show_stats(stats)

    def test_show_stats_empty(self, ui):
        ui.show_stats({})

    def test_show_tools(self, ui):
        tool_stats = {
            "web_search": {"usage": 5, "success": 4, "failure": 1, "builtin": True, "dangerous": False},
            "code_execute": {"usage": 2, "success": 2, "failure": 0, "builtin": True, "dangerous": True},
        }
        ui.show_tools(tool_stats)

    def test_show_tools_empty(self, ui):
        ui.show_tools({})


# ------------------------------------------------------------------
# Memory results rendering
# ------------------------------------------------------------------

class TestMemoryResults:

    def test_show_results(self, ui):
        results = [
            {"id": 1, "memory_type": "semantic", "content": "Python is great"},
            {"id": 2, "memory_type": "episodic", "content": "User asked about Python"},
        ]
        ui.show_memory_results(results)

    def test_show_empty(self, ui):
        ui.show_memory_results([])


# ------------------------------------------------------------------
# Lifecycle
# ------------------------------------------------------------------

class TestLifecycle:

    def test_banner(self, ui):
        ui.banner(model="anthropic/claude-sonnet-4-20250514", tool_count=11)

    def test_shutdown(self, ui):
        ui.shutdown()

    def test_show_help(self, ui):
        ui.show_help()


# ------------------------------------------------------------------
# Interrupt flag
# ------------------------------------------------------------------

class TestInterrupt:

    def test_interrupt_flag(self, ui):
        assert ui.interrupted is False
        ui._interrupted = True
        assert ui.interrupted is True
        # Should auto-clear
        assert ui.interrupted is False

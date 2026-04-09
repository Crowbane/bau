"""Phase 4 smoke tests — Tool registry, schema generation, built-in tools."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

# Patch workspace before importing tools
_tmp_workspace = tempfile.mkdtemp(prefix="bau_test_workspace_")

import tools as tools_mod

tools_mod.WORKSPACE = Path(_tmp_workspace).resolve()


from tools import (
    ToolRegistry,
    _safe_path,
    code_execute,
    file_edit,
    file_read,
    file_write,
    function_to_schema,
    register_builtins,
    shell_run,
    web_search,
)


def _run(coro):
    """Helper to run async code in tests."""
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def clean_workspace():
    """Ensure a clean workspace directory for each test."""
    workspace = Path(_tmp_workspace)
    for item in workspace.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    yield


@pytest.fixture
def registry():
    """Create a ToolRegistry with built-ins registered (no memory)."""
    reg = ToolRegistry()
    register_builtins(reg)
    return reg


# ------------------------------------------------------------------
# Schema generation
# ------------------------------------------------------------------

class TestFunctionToSchema:

    def test_basic_function(self):
        async def greet(name: str, excited: bool = False) -> str:
            """Say hello to someone.

            Args:
                name: The person's name.
                excited: Whether to add exclamation marks.
            """
            return f"Hello, {name}{'!!!' if excited else '.'}"

        schema = function_to_schema(greet)
        assert schema["type"] == "function"
        func = schema["function"]
        assert func["name"] == "greet"
        assert func["description"] == "Say hello to someone."
        params = func["parameters"]
        assert params["type"] == "object"
        assert "name" in params["properties"]
        assert "excited" in params["properties"]
        assert params["properties"]["name"]["type"] == "string"
        assert params["properties"]["excited"]["type"] == "boolean"
        assert "name" in params["required"]
        assert "excited" not in params["required"]

    def test_list_param(self):
        async def process(items: list[dict]) -> dict:
            """Process a list of items.

            Args:
                items: Items to process.
            """
            return {}

        schema = function_to_schema(process)
        props = schema["function"]["parameters"]["properties"]
        assert props["items"]["type"] == "array"


# ------------------------------------------------------------------
# ToolRegistry
# ------------------------------------------------------------------

class TestToolRegistry:

    def test_register_and_names(self, registry):
        names = registry.names()
        assert len(names) == 12
        expected = {
            "web_search", "web_fetch", "file_read", "file_write",
            "file_edit", "code_execute", "shell_run",
            "memory_store", "memory_query", "todo_write", "ask_user",
            "create_tool",
        }
        assert set(names) == expected

    def test_schemas_valid_openai_format(self, registry):
        schemas = registry.schemas()
        assert len(schemas) == 12
        for s in schemas:
            assert s["type"] == "function"
            func = s["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func
            params = func["parameters"]
            assert params["type"] == "object"
            assert "properties" in params

    def test_get_existing(self, registry):
        tool = registry.get("file_read")
        assert tool is not None
        assert tool.name == "file_read"
        assert tool.is_builtin is True

    def test_get_missing(self, registry):
        assert registry.get("nonexistent") is None

    def test_call_unknown_tool(self, registry):
        result = _run(registry.call("nonexistent", {}))
        assert "error" in result
        assert "Unknown tool" in result["error"]

    def test_call_catches_exception(self):
        reg = ToolRegistry()

        async def bad_tool() -> str:
            """A tool that always fails."""
            raise RuntimeError("boom")

        reg.register(bad_tool)
        result = _run(reg.call("bad_tool", {}))
        assert "error" in result
        assert "boom" in result["error"]

    def test_call_validates_missing_required(self, registry):
        result = _run(registry.call("file_read", {}))
        assert "error" in result
        assert "Missing required" in result["error"]

    def test_call_validates_unknown_args(self, registry):
        result = _run(registry.call("file_read", {"path": "x.txt", "bogus": 42}))
        assert "error" in result
        assert "Unknown arguments" in result["error"]

    def test_stats(self, registry):
        stats = registry.stats()
        assert "file_read" in stats
        assert stats["file_read"]["usage"] == 0
        assert stats["code_execute"]["dangerous"] is True
        assert stats["file_read"]["dangerous"] is False


# ------------------------------------------------------------------
# File tools (round-trip)
# ------------------------------------------------------------------

class TestFileTools:

    def test_write_read_roundtrip(self):
        async def _test():
            result = await file_write("hello.txt", "Hello, world!\nLine 2\n")
            assert result["bytes_written"] > 0
            result = await file_read("hello.txt")
            assert "Hello, world!" in result["content"]
            assert result["total_lines"] == 2
        _run(_test())

    def test_read_with_line_range(self):
        async def _test():
            await file_write("lines.txt", "one\ntwo\nthree\nfour\nfive\n")
            result = await file_read("lines.txt", start_line=2, end_line=4)
            assert "two" in result["content"]
            assert "four" in result["content"]
            assert "one" not in result["content"]
            assert "five" not in result["content"]
        _run(_test())

    def test_read_nonexistent(self):
        result = _run(file_read("nope.txt"))
        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_edit(self):
        async def _test():
            await file_write("edit_me.txt", "Hello, Alice! Nice day.")
            result = await file_edit("edit_me.txt", "Alice", "Bob")
            assert result["replaced"] is True
            content = await file_read("edit_me.txt")
            assert "Bob" in content["content"]
            assert "Alice" not in content["content"]
        _run(_test())

    def test_edit_not_found(self):
        async def _test():
            await file_write("edit2.txt", "abc")
            result = await file_edit("edit2.txt", "xyz", "replaced")
            assert "error" in result
        _run(_test())

    def test_edit_ambiguous(self):
        async def _test():
            await file_write("edit3.txt", "aaa aaa")
            result = await file_edit("edit3.txt", "aaa", "bbb")
            assert "error" in result
            assert "2 times" in result["error"]
        _run(_test())


# ------------------------------------------------------------------
# Path escape
# ------------------------------------------------------------------

class TestPathSafety:

    def test_escape_blocked(self):
        with pytest.raises(ValueError, match="escapes workspace"):
            _safe_path("../../etc/passwd")

    def test_escape_via_absolute(self):
        with pytest.raises(ValueError, match="escapes workspace"):
            _safe_path("/etc/passwd")

    def test_valid_relative(self):
        result = _safe_path("subdir/file.txt")
        assert str(result).startswith(str(tools_mod.WORKSPACE))

    def test_valid_simple(self):
        result = _safe_path("file.txt")
        assert result.name == "file.txt"


# ------------------------------------------------------------------
# Shell allowlist
# ------------------------------------------------------------------

class TestShellRun:

    def test_allowed_command(self):
        result = _run(shell_run("echo hello"))
        assert result["stdout"].strip() == "hello"
        assert result["returncode"] == 0

    def test_blocked_command(self):
        result = _run(shell_run("rm -rf /"))
        assert "error" in result
        assert "not allowed" in result["error"]

    def test_blocked_python(self):
        result = _run(shell_run("python3 -c 'print(1)'"))
        assert "error" in result
        assert "not allowed" in result["error"]


# ------------------------------------------------------------------
# Code execute
# ------------------------------------------------------------------

class TestCodeExecute:

    def test_simple_python(self):
        result = _run(code_execute("print('hello from code')"))
        assert "hello from code" in result["stdout"]
        assert result["returncode"] == 0

    def test_timeout(self):
        result = _run(code_execute("while True: pass", timeout=2))
        assert "timed out" in result["stderr"].lower()
        assert result["returncode"] == -1

    def test_error_captured(self):
        result = _run(code_execute("raise ValueError('oops')"))
        assert result["returncode"] != 0
        assert "oops" in result["stderr"]


# ------------------------------------------------------------------
# Todo write
# ------------------------------------------------------------------

class TestTodoWrite:

    def test_with_mock_memory(self):
        """Verify todo_write updates core memory."""
        stored = {}

        class FakeMemory:
            def core_set(self, label, value):
                stored[label] = value

        from tools import _make_todo_write

        todo_fn = _make_todo_write(FakeMemory())
        result = _run(todo_fn([
            {"text": "Buy milk", "done": False},
            {"text": "Write code", "done": True},
        ]))
        assert result["updated"] is True
        assert result["count"] == 2
        assert "task_ledger" in stored
        assert "- [ ] Buy milk" in stored["task_ledger"]
        assert "- [x] Write code" in stored["task_ledger"]


# ------------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def cleanup_tmp_workspace():
    yield
    shutil.rmtree(_tmp_workspace, ignore_errors=True)

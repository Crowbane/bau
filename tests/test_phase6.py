"""Phase 6 smoke tests — dynamic tool creation, AST validator, sandbox, hot-loading."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from tools import (
    DANGEROUS_BUILTINS,
    DANGEROUS_MODULES,
    IMPORT_ALLOWLIST,
    SafetyValidator,
    ToolRegistry,
    _TOOL_NAME_RE,
    function_to_schema,
    load_tool_from_file,
    register_builtins,
    sandbox_test,
    validate_tool_source,
)


def _run(coro):
    """Helper to run async code in tests."""
    return asyncio.run(coro)


# ------------------------------------------------------------------
# AST safety validator
# ------------------------------------------------------------------


class TestASTValidator:

    def test_accepts_safe_code(self):
        source = "def add(a: int, b: int) -> int:\n    return a + b"
        ok, violations = validate_tool_source(source)
        assert ok is True
        assert violations == []

    def test_accepts_allowlisted_imports(self):
        source = (
            "def compute(x: float) -> float:\n"
            "    import math\n"
            "    return math.sqrt(x)"
        )
        ok, violations = validate_tool_source(source)
        assert ok is True

    def test_rejects_import_os(self):
        source = "def bad():\n    import os\n    return os.getcwd()"
        ok, violations = validate_tool_source(source)
        assert ok is False
        assert any("os" in v for v in violations)

    def test_rejects_import_subprocess(self):
        source = "def bad():\n    import subprocess\n    return None"
        ok, violations = validate_tool_source(source)
        assert ok is False
        assert any("subprocess" in v for v in violations)

    def test_rejects_from_import_shutil(self):
        source = "def bad():\n    from shutil import rmtree\n    return None"
        ok, violations = validate_tool_source(source)
        assert ok is False
        assert any("shutil" in v for v in violations)

    def test_rejects_eval(self):
        source = "def bad(s: str):\n    return eval(s)"
        ok, violations = validate_tool_source(source)
        assert ok is False
        assert any("eval" in v for v in violations)

    def test_rejects_exec(self):
        source = "def bad(s: str):\n    exec(s)"
        ok, violations = validate_tool_source(source)
        assert ok is False
        assert any("exec" in v for v in violations)

    def test_rejects_open(self):
        source = "def bad():\n    return open('/etc/passwd').read()"
        ok, violations = validate_tool_source(source)
        assert ok is False
        assert any("open" in v for v in violations)

    def test_rejects_dunder_access(self):
        source = "def bad(obj):\n    return obj.__class__.__bases__"
        ok, violations = validate_tool_source(source)
        assert ok is False
        assert any("__class__" in v or "__bases__" in v for v in violations)

    def test_rejects_syntax_error(self):
        source = "def bad(:\n    return"
        ok, violations = validate_tool_source(source)
        assert ok is False
        assert any("SyntaxError" in v for v in violations)

    def test_rejects_compile_builtin(self):
        source = "def bad():\n    return compile('1+1', '', 'eval')"
        ok, violations = validate_tool_source(source)
        assert ok is False
        assert any("compile" in v for v in violations)


class TestToolNameRegex:

    def test_valid_names(self):
        assert _TOOL_NAME_RE.match("add")
        assert _TOOL_NAME_RE.match("calculate_average")
        assert _TOOL_NAME_RE.match("fetch_rss_v2")

    def test_invalid_names(self):
        assert not _TOOL_NAME_RE.match("Add")
        assert not _TOOL_NAME_RE.match("_private")
        assert not _TOOL_NAME_RE.match("1bad")
        assert not _TOOL_NAME_RE.match("")
        assert not _TOOL_NAME_RE.match("a" * 50)


# ------------------------------------------------------------------
# Sandbox testing
# ------------------------------------------------------------------


class TestSandbox:

    def test_simple_function_passes(self):
        source = "def add(a, b):\n    return a + b"
        tests = [
            {"input": {"a": 1, "b": 2}, "expected": 3},
            {"input": {"a": 0, "b": 0}, "expected": 0},
        ]
        result = sandbox_test(source, tests, "add")
        assert result["passed"] is True
        assert result["stage"] == "test"

    def test_rejects_dangerous_import(self):
        source = "import os\ndef bad():\n    return os.getcwd()"
        result = sandbox_test(source, [], "bad")
        assert result["passed"] is False
        assert result["stage"] == "ast"

    def test_function_not_found(self):
        source = "def foo():\n    return 1"
        result = sandbox_test(source, [], "bar")
        assert result["passed"] is False
        assert result["stage"] == "lookup"

    def test_failing_test_case(self):
        source = "def add(a, b):\n    return a - b"  # deliberately wrong
        tests = [{"input": {"a": 1, "b": 2}, "expected": 3}]
        result = sandbox_test(source, tests, "add")
        assert result["passed"] is False
        assert result["stage"] == "test"

    def test_with_allowlisted_import(self):
        source = (
            "def sqrt_val(x):\n"
            "    import math\n"
            "    return math.sqrt(x)"
        )
        tests = [{"input": {"x": 4.0}, "expected": 2.0}]
        result = sandbox_test(source, tests, "sqrt_val")
        assert result["passed"] is True


# ------------------------------------------------------------------
# Hot-loading from file
# ------------------------------------------------------------------


class TestHotLoading:

    def test_load_simple_tool(self, tmp_path):
        tool_file = tmp_path / "greet.py"
        tool_file.write_text(
            'def greet(name: str) -> str:\n'
            '    """Greet someone."""\n'
            '    return f"Hello, {name}!"\n'
        )
        func = load_tool_from_file(str(tool_file))
        assert func.__name__ == "greet"
        assert func("World") == "Hello, World!"

    def test_schema_generation_from_loaded(self, tmp_path):
        tool_file = tmp_path / "add_nums.py"
        tool_file.write_text(
            'def add_nums(a: int, b: int) -> int:\n'
            '    """Add two numbers.\n\n'
            '    Args:\n'
            '        a: First number.\n'
            '        b: Second number.\n\n'
            '    Returns:\n'
            '        Sum of a and b.\n'
            '    """\n'
            '    return a + b\n'
        )
        func = load_tool_from_file(str(tool_file))
        schema = function_to_schema(func)
        assert schema["function"]["name"] == "add_nums"
        assert "a" in schema["function"]["parameters"]["properties"]
        assert "b" in schema["function"]["parameters"]["properties"]


# ------------------------------------------------------------------
# Full creation loop with stub LLM
# ------------------------------------------------------------------


class TestCreateToolLoop:

    @pytest.fixture
    def setup(self, tmp_path):
        """Set up a registry with stub LLM and memory for tool creation tests."""
        from agent import AgentMemory

        db_path = str(tmp_path / "test.db")
        memory = AgentMemory(db_path=db_path)

        registry = ToolRegistry(memory=memory)
        registry._config = {
            "tools": {
                "generated_dir": str(tmp_path / "tools_gen"),
                "require_approval": False,
                "hard_cap": 100,
            },
        }
        registry._prompts = {
            "tool_creator": "Create a tool. Respond with JSON: {\"source\": \"...\", \"tests\": [...]}",
            "critic": "",  # skip critic for tests
        }
        registry._on_event = lambda *a, **kw: None

        # Register builtins so create_tool is available
        register_builtins(registry, memory=memory)

        return registry, memory, tmp_path

    def test_full_creation_with_stub_llm(self, setup):
        registry, memory, tmp_path = setup

        # Stub LLM returns valid source + tests
        stub_response = MagicMock()
        stub_response.text = json.dumps({
            "source": (
                "def double_value(x: int) -> int:\n"
                '    """Double an integer value.\n\n'
                "    Args:\n"
                "        x: Input integer.\n\n"
                "    Returns:\n"
                "        Doubled value.\n"
                '    """\n'
                "    return x * 2"
            ),
            "tests": [
                {"input": {"x": 5}, "expected": 10},
                {"input": {"x": 0}, "expected": 0},
                {"input": {"x": -3}, "expected": -6},
            ],
        })
        stub_response.cost = 0.0

        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value=stub_response)
        registry._llm = mock_llm

        result = _run(registry._tools["create_tool"].func(
            name="double_value",
            description="Double an integer value",
            parameters={"x": {"type": "integer"}},
            rationale="Useful for general math operations",
        ))

        assert result["created"] is True
        assert result["tool_name"] == "double_value"
        assert "double_value" in registry.names()

        # Tool file should exist
        gen_dir = tmp_path / "tools_gen"
        assert (gen_dir / "double_value.py").exists()

        # Tool should be callable
        call_result = _run(registry.call("double_value", {"x": 7}))
        assert call_result["result"] == 14

        # Tool metadata should be in SQLite
        meta = memory.tool_get("double_value")
        assert meta is not None
        assert meta["name"] == "double_value"

    def test_self_debug_retry(self, setup):
        """Stub LLM returns broken impl first, then fixed impl on retry."""
        registry, memory, tmp_path = setup

        broken_response = MagicMock()
        broken_response.text = json.dumps({
            "source": "def triple(x: int) -> int:\n    return x + x",  # wrong
            "tests": [{"input": {"x": 3}, "expected": 9}],
        })
        broken_response.cost = 0.0

        fixed_response = MagicMock()
        fixed_response.text = json.dumps({
            "source": (
                "def triple(x: int) -> int:\n"
                '    """Triple an integer.\n\n'
                "    Args:\n"
                "        x: Input.\n\n"
                "    Returns:\n"
                "        Tripled value.\n"
                '    """\n'
                "    return x * 3"
            ),
            "tests": [{"input": {"x": 3}, "expected": 9}],
        })
        fixed_response.cost = 0.0

        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(side_effect=[broken_response, fixed_response])
        registry._llm = mock_llm

        result = _run(registry._tools["create_tool"].func(
            name="triple",
            description="Triple an integer",
            parameters={"x": {"type": "integer"}},
            rationale="Math utility",
        ))

        assert result["created"] is True
        assert result["tool_name"] == "triple"
        # Should have called LLM at least twice (first broken, then fixed)
        assert mock_llm.complete.call_count >= 2

    def test_rejects_duplicate_name(self, setup):
        registry, memory, tmp_path = setup
        registry._llm = AsyncMock()

        # web_search is already registered as a builtin
        result = _run(registry._tools["create_tool"].func(
            name="web_search",
            description="Search the web",
            parameters={},
            rationale="Test",
        ))
        assert result["created"] is False
        assert "already exists" in result["errors"][0]

    def test_rejects_invalid_name(self, setup):
        registry, memory, tmp_path = setup
        registry._llm = AsyncMock()

        result = _run(registry._tools["create_tool"].func(
            name="BadName",
            description="Test",
            parameters={},
            rationale="Test",
        ))
        assert result["created"] is False
        assert "Invalid name" in result["errors"][0]

    def test_similarity_rejection(self, setup):
        """Pre-create calculate_average, then try compute_mean => blocked."""
        registry, memory, tmp_path = setup

        # First: create calculate_average via stub LLM
        avg_source = (
            "def calculate_average(numbers: list) -> float:\n"
            '    """Calculate the arithmetic mean of a list of numbers.\n\n'
            "    Args:\n"
            "        numbers: List of numeric values.\n\n"
            "    Returns:\n"
            "        Arithmetic mean.\n"
            '    """\n'
            "    if not numbers:\n"
            "        return 0.0\n"
            "    return sum(numbers) / len(numbers)"
        )
        stub_response = MagicMock()
        stub_response.text = json.dumps({
            "source": avg_source,
            "tests": [
                {"input": {"numbers": [1, 2, 3]}, "expected": 2.0},
                {"input": {"numbers": []}, "expected": 0.0},
            ],
        })
        stub_response.cost = 0.0

        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value=stub_response)
        registry._llm = mock_llm

        result1 = _run(registry._tools["create_tool"].func(
            name="calculate_average",
            description="Calculate the arithmetic mean of a list of numbers",
            parameters={"numbers": {"type": "array"}},
            rationale="General math utility",
        ))
        assert result1["created"] is True

        # Now try to create compute_mean with very similar description
        result2 = _run(registry._tools["create_tool"].func(
            name="compute_mean",
            description="Compute the arithmetic mean of a list of numbers",
            parameters={"numbers": {"type": "array"}},
            rationale="Math utility",
        ))
        assert result2["created"] is False
        assert any("similar" in e.lower() or "Too similar" in e for e in result2["errors"])


# ------------------------------------------------------------------
# Quarantine
# ------------------------------------------------------------------


class TestQuarantine:

    def test_quarantine_after_failures(self, tmp_path):
        """Simulate 6 calls with 4 failures => tool gets deprecated."""
        from agent import AgentMemory

        db_path = str(tmp_path / "test.db")
        memory = AgentMemory(db_path=db_path)

        registry = ToolRegistry(memory=memory)
        registry._config = {}
        registry._on_event = lambda *a, **kw: None

        # Create a tool that sometimes fails
        call_count = 0

        def flaky_tool(x: int) -> int:
            """A flaky tool.

            Args:
                x: Input.

            Returns:
                Result.
            """
            nonlocal call_count
            call_count += 1
            if call_count in (1, 4):  # succeed on calls 1 and 4
                return x * 2
            raise ValueError("Random failure")

        tool = registry.register(flaky_tool, dangerous=False, builtin=False)

        # Register in memory too
        memory.tool_register_meta(
            name="flaky_tool",
            description="A flaky tool",
            file_path="/tmp/flaky_tool.py",
            params_json="{}",
            source_hash="abc123",
        )

        # Make 6 calls (2 successes, 4 failures)
        for i in range(6):
            _run(registry.call("flaky_tool", {"x": 1}))

        # After 5+ uses with <50% success, tool should be quarantined
        assert "flaky_tool" not in registry.names()

        # Check it's deprecated in DB
        meta = memory.tool_get("flaky_tool")
        assert meta is not None
        assert meta["deprecated"] is True


# ------------------------------------------------------------------
# Restore on startup
# ------------------------------------------------------------------


class TestRestoreOnStartup:

    def test_restore_tool_from_db(self, tmp_path):
        """Create a tool, simulate restart, verify tool is back."""
        from agent import AgentMemory

        db_path = str(tmp_path / "test.db")
        memory = AgentMemory(db_path=db_path)

        # Write a tool file
        tools_dir = tmp_path / "tools_gen"
        tools_dir.mkdir()
        tool_file = tools_dir / "greet.py"
        tool_file.write_text(
            'def greet(name: str) -> str:\n'
            '    """Say hello.\n\n'
            '    Args:\n'
            '        name: Who to greet.\n\n'
            '    Returns:\n'
            '        Greeting.\n'
            '    """\n'
            '    return f"Hello, {name}!"\n'
        )

        # Register metadata
        memory.tool_register_meta(
            name="greet",
            description="Say hello",
            file_path=str(tool_file),
            params_json='{"name": {"type": "string"}}',
            source_hash="test_hash",
        )

        # Simulate restart: new registry, restore from DB
        registry2 = ToolRegistry(memory=memory)
        for row in memory.list_tools(deprecated=False):
            func = load_tool_from_file(row["file_path"])
            registry2.register(func, dangerous=False, builtin=False)

        assert "greet" in registry2.names()
        result = _run(registry2.call("greet", {"name": "BAU"}))
        assert result["result"] == "Hello, BAU!"


# ------------------------------------------------------------------
# Stale-tool deprecation
# ------------------------------------------------------------------


class TestStaleDeprecation:

    def test_deprecates_old_unused_tools(self, tmp_path):
        from agent import AgentMemory

        db_path = str(tmp_path / "test.db")
        memory = AgentMemory(db_path=db_path)

        # Insert a tool with old created_at and zero usage
        old_time = time.time() - (31 * 86400)  # 31 days ago
        memory._conn.execute(
            "INSERT INTO tools (name, description, file_path, parameters_json, "
            "source_hash, created_at, usage_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("old_tool", "old", "/tmp/old.py", "{}", "hash", old_time, 0),
        )
        memory._conn.commit()

        count = memory.tool_deprecate_stale(max_age_days=30)
        assert count == 1

        meta = memory.tool_get("old_tool")
        assert meta["deprecated"] is True

    def test_keeps_recently_created_tools(self, tmp_path):
        from agent import AgentMemory

        db_path = str(tmp_path / "test.db")
        memory = AgentMemory(db_path=db_path)

        memory._conn.execute(
            "INSERT INTO tools (name, description, file_path, parameters_json, "
            "source_hash, created_at, usage_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("new_tool", "new", "/tmp/new.py", "{}", "hash", time.time(), 0),
        )
        memory._conn.commit()

        count = memory.tool_deprecate_stale(max_age_days=30)
        assert count == 0

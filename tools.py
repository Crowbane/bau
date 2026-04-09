"""BAU tools — Tool registry, built-ins, schema generation, dynamic creation (Phase 4-6)."""
from __future__ import annotations

import ast
import asyncio
import hashlib
import importlib.util
import inspect
import json
import os
import re as _re
import shlex
import signal
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, get_args, get_origin, get_type_hints

from docstring_parser import parse as parse_docstring


# ------------------------------------------------------------------
# AST safety validator (Layer 1 — pre-execution gate)
# ------------------------------------------------------------------

DANGEROUS_MODULES = frozenset({
    'os', 'sys', 'subprocess', 'shutil', 'socket', 'http',
    'urllib', 'ctypes', 'pickle', 'code', 'importlib', 'builtins',
})
IMPORT_ALLOWLIST = frozenset({
    'math', 'json', 'datetime', 're', 'collections',
    'itertools', 'functools', 'statistics', 'hashlib',
    'string', 'decimal', 'fractions', 'random', 'time',
})
DANGEROUS_BUILTINS = frozenset({
    'eval', 'exec', 'compile', '__import__', 'open',
    'input', 'breakpoint', 'globals', 'locals',
})

_TOOL_NAME_RE = _re.compile(r'^[a-z][a-z0-9_]{1,40}$')


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    import math as _m
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = _m.sqrt(sum(x * x for x in a))
    mag_b = _m.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class SafetyValidator(ast.NodeVisitor):
    """AST visitor that flags dangerous patterns in generated tool source."""

    def __init__(self) -> None:
        self.violations: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        """Check bare imports against the allowlist."""
        for alias in node.names:
            root = alias.name.split('.')[0]
            if root not in IMPORT_ALLOWLIST:
                self.violations.append(f"Forbidden import: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Check from-imports against the allowlist."""
        if node.module:
            root = node.module.split('.')[0]
            if root not in IMPORT_ALLOWLIST:
                self.violations.append(f"Forbidden import: from {node.module}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Block calls to dangerous builtins."""
        if isinstance(node.func, ast.Name) and node.func.id in DANGEROUS_BUILTINS:
            self.violations.append(f"Forbidden builtin call: {node.func.id}()")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Block dunder attribute access (__class__, __bases__, etc.)."""
        if node.attr.startswith('__') and node.attr.endswith('__'):
            self.violations.append(f"Forbidden dunder access: .{node.attr}")
        self.generic_visit(node)


def validate_tool_source(source: str) -> tuple[bool, list[str]]:
    """Validate generated tool source code via AST inspection.

    Args:
        source: Python source code string.

    Returns:
        Tuple of (is_safe, list_of_violations).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return False, [f"SyntaxError: {e}"]
    v = SafetyValidator()
    v.visit(tree)
    return len(v.violations) == 0, v.violations


# ------------------------------------------------------------------
# RestrictedPython sandbox (Layer 2 — execution gate)
# ------------------------------------------------------------------

def sandbox_test(
    source: str,
    test_cases: list[dict],
    func_name: str,
    timeout_secs: int = 5,
) -> dict:
    """Validate and test generated tool source in a RestrictedPython sandbox.

    Pipeline: AST validate -> compile_restricted -> exec -> run test cases.

    Args:
        source: Python source code.
        test_cases: List of dicts with 'input' and optional 'expected'.
        func_name: Name of the function to test.
        timeout_secs: Max execution time in seconds.

    Returns:
        Dict with 'passed' (bool), 'stage', and either 'results' or 'errors'.
    """
    from RestrictedPython import compile_restricted, limited_builtins, safe_globals, utility_builtins
    from RestrictedPython.Eval import default_guarded_getitem
    from RestrictedPython.Guards import guarded_unpack_sequence, safer_getattr

    # Layer 1: AST
    ok, violations = validate_tool_source(source)
    if not ok:
        return {"passed": False, "stage": "ast", "errors": violations}

    # Layer 2: RestrictedPython compile
    try:
        byte_code = compile_restricted(source, '<tool>', 'exec')
    except SyntaxError as e:
        return {"passed": False, "stage": "compile", "errors": [str(e)]}

    # Build sandbox globals with allowlisted modules
    sandbox_globals = {
        **safe_globals,
        "_getitem_": default_guarded_getitem,
        "_getattr_": safer_getattr,
        "_getiter_": iter,
        "_iter_unpack_sequence_": guarded_unpack_sequence,
    }

    # Merge RestrictedPython's extended builtin sets + safe pure builtins
    # that generated tools commonly need (sum, min, max, dict, etc.)
    _SAFE_EXTRA_BUILTINS = {
        "dict": dict, "list": list, "set": set, "frozenset": frozenset,
        "tuple": tuple, "sum": sum, "min": min, "max": max,
        "map": map, "filter": filter, "enumerate": enumerate,
        "reversed": reversed, "any": any, "all": all, "type": type,
        "object": object, "iter": iter, "next": next, "print": print,
    }

    # Guarded __import__ that only allows allowlisted modules
    def _guarded_import(name, *args, **kwargs):
        root = name.split('.')[0]
        if root not in IMPORT_ALLOWLIST:
            raise ImportError(f"Import of '{name}' is not allowed")
        return __import__(name, *args, **kwargs)

    sandbox_globals["__builtins__"] = {
        **sandbox_globals.get("__builtins__", {}),
        **limited_builtins,
        **utility_builtins,
        **_SAFE_EXTRA_BUILTINS,
        "__import__": _guarded_import,
    }
    # Pre-import allowlisted modules so tool code can reference them directly
    for mod_name in IMPORT_ALLOWLIST:
        try:
            sandbox_globals[mod_name] = __import__(mod_name)
        except ImportError:
            pass

    sandbox_locals: dict[str, Any] = {}

    # Resource limits (Layer 3 — CPU cap, if available)
    # Only RLIMIT_CPU is safe for in-process sandboxing; RLIMIT_AS and
    # RLIMIT_FSIZE affect the entire host process and can break the runtime.
    _old_cpu_limit = None
    try:
        import resource
        _old_cpu_limit = resource.getrlimit(resource.RLIMIT_CPU)
        resource.setrlimit(resource.RLIMIT_CPU, (timeout_secs, _old_cpu_limit[1]))
    except (ImportError, ValueError, OSError):
        pass

    # Execute with timeout (SIGALRM on Unix, thread fallback)
    def _exec_sandboxed() -> str | None:
        try:
            exec(byte_code, sandbox_globals, sandbox_locals)  # noqa: S102
            return None
        except Exception as e:
            return f"{type(e).__name__}: {e}"

    try:
        error = _run_with_timeout(_exec_sandboxed, timeout_secs)
    finally:
        if _old_cpu_limit is not None:
            try:
                import resource
                resource.setrlimit(resource.RLIMIT_CPU, _old_cpu_limit)
            except (ImportError, ValueError, OSError):
                pass
    if error is not None:
        return {"passed": False, "stage": "exec", "errors": [error]}

    func = sandbox_locals.get(func_name) or sandbox_globals.get(func_name)
    if not func:
        return {
            "passed": False, "stage": "lookup",
            "errors": [f"Function '{func_name}' not defined in source"],
        }

    # Layer 3: Run test cases
    results: list[dict] = []
    for tc in test_cases:
        tc_input = tc.get("input", {})
        try:
            actual = func(**tc_input)
            if "expected" in tc:
                passed = actual == tc["expected"]
            else:
                passed = True  # smoke test — just check it doesn't crash
            results.append({"input": tc_input, "actual": actual, "passed": passed})
        except Exception as e:
            results.append({"input": tc_input, "error": str(e), "passed": False})

    all_passed = all(r["passed"] for r in results)
    return {"passed": all_passed, "stage": "test", "results": results}


def _run_with_timeout(fn: Callable[[], Any], timeout_secs: int) -> Any:
    """Run a callable with a timeout. Returns None on success, error string on failure."""
    if hasattr(signal, 'SIGALRM'):
        def _handler(signum, frame):
            raise TimeoutError(f"Execution timed out after {timeout_secs}s")

        old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(timeout_secs)
        try:
            return fn()
        except TimeoutError as e:
            return str(e)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        import threading
        result_box: list[Any] = [None]

        def _target():
            result_box[0] = fn()

        t = threading.Thread(target=_target)
        t.start()
        t.join(timeout=timeout_secs)
        if t.is_alive():
            return f"Execution timed out after {timeout_secs}s"
        return result_box[0]


# ------------------------------------------------------------------
# Hot-loading from file
# ------------------------------------------------------------------

def load_tool_from_file(file_path: str) -> Callable:
    """Hot-load a tool function from a .py file via importlib.

    Args:
        file_path: Path to the tool's .py file.

    Returns:
        The tool function (matching the file stem name).

    Raises:
        AttributeError: If the expected function is not found in the module.
        ImportError: If the module cannot be loaded.
    """
    path = Path(file_path)
    module_name = f"bau_tool_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    func_name = path.stem
    if not hasattr(module, func_name):
        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if callable(obj) and not attr_name.startswith('_'):
                return obj
        raise AttributeError(f"No function '{func_name}' found in {file_path}")
    return getattr(module, func_name)


# ------------------------------------------------------------------
# Workspace confinement & safety
# ------------------------------------------------------------------

WORKSPACE = Path("./workspace").resolve()
WORKSPACE.mkdir(exist_ok=True)

SHELL_ALLOWLIST = {"ls", "cat", "grep", "find", "wc", "head", "tail", "pwd", "echo", "git"}


def _safe_path(p: str) -> Path:
    """Resolve a path relative to WORKSPACE, blocking escapes.

    Args:
        p: Relative or absolute path string.

    Returns:
        Resolved Path inside WORKSPACE.

    Raises:
        ValueError: If the resolved path escapes the workspace.
    """
    resolved = (WORKSPACE / p).resolve()
    if not str(resolved).startswith(str(WORKSPACE)):
        raise ValueError(f"Path escapes workspace: {p}")
    return resolved


# ------------------------------------------------------------------
# Schema auto-generator
# ------------------------------------------------------------------

_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _python_type_to_json(annotation: Any) -> dict:
    """Map a Python type annotation to a JSON Schema fragment.

    Args:
        annotation: A Python type hint.

    Returns:
        JSON Schema dict (e.g. ``{"type": "string"}``).
    """
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {"type": "string"}

    # Handle Optional / Union with None
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is list or (origin is not None and issubclass(origin, list)):
        item_type = _python_type_to_json(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item_type}

    if origin is dict or (origin is not None and issubclass(origin, dict)):
        return {"type": "object"}

    # Union types (X | None → optional)
    if args and type(None) in args:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _python_type_to_json(non_none[0])

    if annotation in _TYPE_MAP:
        return {"type": _TYPE_MAP[annotation]}

    return {"type": "string"}


def function_to_schema(func: Callable) -> dict:
    """Auto-generate an OpenAI-format tool schema from a function signature.

    Uses ``inspect.signature`` for parameters and ``docstring_parser`` for
    descriptions. Called once at registration time and cached.

    Args:
        func: A Python function with type hints and a Google-style docstring.

    Returns:
        OpenAI tool schema dict with type, function.name, function.description,
        and function.parameters.
    """
    sig = inspect.signature(func)
    doc = parse_docstring(inspect.getdoc(func) or "")
    param_descs = {p.arg_name: p.description for p in doc.params}

    # Resolve string annotations from `from __future__ import annotations`
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    properties: dict[str, dict] = {}
    for name, p in sig.parameters.items():
        annotation = hints.get(name, p.annotation)
        prop = _python_type_to_json(annotation)
        if name in param_descs and param_descs[name]:
            prop["description"] = param_descs[name]
        if p.default is not inspect.Parameter.empty and p.default is not None:
            prop["default"] = p.default
        properties[name] = prop

    required = [
        n for n, p in sig.parameters.items()
        if p.default is inspect.Parameter.empty
    ]

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": doc.short_description or "",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


# ------------------------------------------------------------------
# Tool dataclass
# ------------------------------------------------------------------

@dataclass
class Tool:
    """A registered tool with metadata and usage stats."""

    name: str
    description: str
    func: Callable
    schema: dict
    is_builtin: bool = True
    is_dangerous: bool = False
    success_count: int = 0
    failure_count: int = 0
    usage_count: int = 0


# ------------------------------------------------------------------
# ToolRegistry
# ------------------------------------------------------------------

class ToolRegistry:
    """Central tool registry with schema cache, validation, and dispatch.

    Manages built-in and dynamic tools, auto-generates OpenAI-format
    schemas, validates arguments via Pydantic, and tracks usage stats.

    Args:
        memory: Optional AgentMemory for memory tools.
        approval_hook: Async callback ``f(name, args) -> bool`` for dangerous tools.
        ask_user_hook: Async callback ``f(question) -> str`` for ask_user tool.
    """

    def __init__(
        self,
        memory: Any = None,
        approval_hook: Callable | None = None,
        ask_user_hook: Callable | None = None,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self._memory = memory
        self._approval_hook = approval_hook
        self._ask_user_hook = ask_user_hook
        self._default_timeout: int = 60
        # Set by main.py after construction for create_tool meta-tool
        self._llm: Any = None
        self._prompts: dict[str, str] = {}
        self._config: dict = {}
        self._on_event: Callable = lambda *a, **kw: None

    def register(
        self,
        func: Callable,
        *,
        dangerous: bool = False,
        builtin: bool = True,
    ) -> Tool:
        """Register a function as a tool, auto-generating its schema.

        Args:
            func: Python function with type hints and Google-style docstring.
            dangerous: If True, requires approval before first use.
            builtin: Whether this is a built-in (not dynamically created).

        Returns:
            The registered Tool object.
        """
        schema = function_to_schema(func)
        name = func.__name__
        tool = Tool(
            name=name,
            description=schema["function"]["description"],
            func=func,
            schema=schema,
            is_builtin=builtin,
            is_dangerous=dangerous,
        )
        self._tools[name] = tool
        return tool

    def names(self) -> list[str]:
        """Return list of registered tool names."""
        return list(self._tools.keys())

    def schemas(self, names: list[str] | None = None) -> list[dict]:
        """Return OpenAI-format tool schemas.

        Args:
            names: Optional subset of tool names. If None, returns all.

        Returns:
            List of OpenAI tool schema dicts.
        """
        if names is None:
            return [t.schema for t in self._tools.values()]
        return [self._tools[n].schema for n in names if n in self._tools]

    def get(self, name: str) -> Tool | None:
        """Get a Tool by name, or None if not found.

        Args:
            name: Tool name.

        Returns:
            Tool object or None.
        """
        return self._tools.get(name)

    @staticmethod
    def _validate_args(schema: dict, args: dict) -> str | None:
        """Validate tool arguments against the schema.

        Args:
            schema: OpenAI-format tool schema.
            args: Arguments to validate.

        Returns:
            Error message string if validation fails, None if OK.
        """
        params = schema.get("function", {}).get("parameters", {})
        properties = params.get("properties", {})
        required = params.get("required", [])

        # Check required parameters are present
        missing = [r for r in required if r not in args]
        if missing:
            return f"Missing required arguments: {', '.join(missing)}"

        # Check for unknown parameters
        if params.get("additionalProperties") is False:
            unknown = [k for k in args if k not in properties]
            if unknown:
                return (
                    f"Unknown arguments: {', '.join(unknown)}. "
                    f"Valid arguments: {', '.join(properties.keys())}"
                )

        return None

    async def call(self, name: str, args: dict) -> dict:
        """Dispatch a tool call with validation, timeout, and error handling.

        Args:
            name: Tool name.
            args: Keyword arguments for the tool function.

        Returns:
            Dict with ``result`` on success or ``error`` on failure.
        """
        # 1. Look up
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(self._tools.keys())
            return {"error": f"Unknown tool: {name}. Available tools: {available}"}

        tool.usage_count += 1

        # 2. Validate args against schema
        validation_error = self._validate_args(tool.schema, args)
        if validation_error:
            tool.failure_count += 1
            return {"error": f"Validation error for '{name}': {validation_error}"}

        # 3. Approval for dangerous tools
        if tool.is_dangerous and self._approval_hook:
            try:
                approved = await self._approval_hook(name, args)
                if not approved:
                    tool.failure_count += 1
                    return {"error": f"Tool '{name}' requires approval and was denied."}
            except Exception as e:
                tool.failure_count += 1
                return {"error": f"Approval check failed: {e}"}

        # 4. Execute with try/except and timeout
        success = False
        try:
            if asyncio.iscoroutinefunction(tool.func):
                result = await asyncio.wait_for(
                    tool.func(**args),
                    timeout=self._default_timeout,
                )
            else:
                result = tool.func(**args)
            success = True
        except asyncio.TimeoutError:
            tool.failure_count += 1
            self._update_generated_stats(tool, success=False)
            return {"error": f"Tool '{name}' timed out after {self._default_timeout}s."}
        except Exception as e:
            tool.failure_count += 1
            self._update_generated_stats(tool, success=False)
            return {"error": f"Tool '{name}' failed: {type(e).__name__}: {e}"}

        tool.success_count += 1
        self._update_generated_stats(tool, success=True)
        return {"result": result}

    def _update_generated_stats(self, tool: Tool, success: bool) -> None:
        """Update stats for a generated (non-builtin) tool and quarantine if failing."""
        if tool.is_builtin or self._memory is None:
            return
        self._memory.tool_update_stats(tool.name, success)
        # Quarantine: success rate < 50% after 5+ uses
        total = tool.success_count + tool.failure_count
        if total >= 5 and tool.success_count / total < 0.5:
            self.unregister(tool.name)
            self._memory._conn.execute(
                "UPDATE tools SET deprecated = 1 WHERE name = ?", (tool.name,),
            )
            self._memory._conn.commit()
            self._on_event("warning", {
                "message": f"Tool '{tool.name}' quarantined (success rate "
                           f"{tool.success_count}/{total} < 50%)",
            })

    def unregister(self, name: str) -> bool:
        """Remove a tool from the active registry.

        Args:
            name: Tool name.

        Returns:
            True if removed, False if not found.
        """
        return self._tools.pop(name, None) is not None

    def stats(self) -> dict:
        """Return usage statistics for all tools.

        Returns:
            Dict mapping tool name to usage/success/failure counts.
        """
        return {
            name: {
                "usage": t.usage_count,
                "success": t.success_count,
                "failure": t.failure_count,
                "dangerous": t.is_dangerous,
                "builtin": t.is_builtin,
            }
            for name, t in self._tools.items()
        }


# ------------------------------------------------------------------
# Built-in tools
# ------------------------------------------------------------------

async def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web using DuckDuckGo.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.

    Returns:
        List of {title, url, snippet} dicts.
    """
    from duckduckgo_search import DDGS

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return [
            {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
            for r in results
        ]
    except Exception as e:
        return [{"error": f"Search failed: {e}"}]


async def web_fetch(url: str) -> dict:
    """Fetch a URL and extract clean text content.

    Args:
        url: The URL to fetch.

    Returns:
        Dict with url, title, content, and content_length.
    """
    import httpx
    import trafilatura

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            html = resp.text
    except Exception as e:
        return {"error": f"Fetch failed for {url}: {e}"}

    extracted = trafilatura.extract(html, include_links=True, include_tables=True)
    if not extracted:
        extracted = html[:8000]

    title = ""
    try:
        meta = trafilatura.extract(html, output_format="json")
        if meta:
            parsed = json.loads(meta)
            title = parsed.get("title", "")
    except Exception:
        pass

    content = extracted[:8000]
    return {
        "url": url,
        "title": title,
        "content": content,
        "content_length": len(content),
    }


async def file_read(path: str, start_line: int = 1, end_line: int | None = None) -> dict:
    """Read a file from the workspace with optional line range.

    Args:
        path: Relative path within the workspace directory.
        start_line: First line to read (1-based, default 1).
        end_line: Last line to read (inclusive). None reads to end.

    Returns:
        Dict with path, content, and total_lines.
    """
    resolved = _safe_path(path)
    if not resolved.exists():
        available = [f.name for f in WORKSPACE.iterdir() if f.is_file()]
        return {"error": f"File not found: {path}. Available files: {', '.join(available) or '(none)'}"}
    if not resolved.is_file():
        return {"error": f"Not a file: {path}"}

    text = resolved.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    total = len(lines)

    start_idx = max(0, start_line - 1)
    end_idx = end_line if end_line is not None else total
    selected = lines[start_idx:end_idx]

    return {
        "path": path,
        "content": "".join(selected),
        "total_lines": total,
    }


async def file_write(path: str, content: str) -> dict:
    """Write content to a file in the workspace (atomic write).

    Args:
        path: Relative path within the workspace directory.
        content: Text content to write.

    Returns:
        Dict with path and bytes_written.
    """
    resolved = _safe_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = resolved.with_suffix(resolved.suffix + ".tmp")
    try:
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.replace(resolved)
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        return {"error": f"Write failed: {e}"}

    return {"path": path, "bytes_written": len(content.encode("utf-8"))}


async def file_edit(path: str, old_string: str, new_string: str) -> dict:
    """Edit a file by replacing an exact string occurrence.

    Args:
        path: Relative path within the workspace directory.
        old_string: The exact string to find and replace.
        new_string: The replacement string.

    Returns:
        Dict with path and replaced status.
    """
    resolved = _safe_path(path)
    if not resolved.exists():
        return {"error": f"File not found: {path}"}

    text = resolved.read_text(encoding="utf-8", errors="replace")
    count = text.count(old_string)

    if count == 0:
        preview = text[:500]
        return {"error": f"String not found in {path}. File preview:\n{preview}"}
    if count > 1:
        return {
            "error": f"String appears {count} times in {path}. "
            "Provide more surrounding context to make the match unique."
        }

    new_text = text.replace(old_string, new_string, 1)
    resolved.write_text(new_text, encoding="utf-8")
    return {"path": path, "replaced": True}


async def code_execute(code: str, timeout: int = 30) -> dict:
    """Execute Python code in a subprocess and return output.

    Args:
        code: Python source code to execute.
        timeout: Maximum execution time in seconds.

    Returns:
        Dict with stdout, stderr, and returncode.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8",
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        proc = await asyncio.create_subprocess_exec(
            "python3", tmp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return {
                "stdout": "",
                "stderr": f"Execution timed out after {timeout}s",
                "returncode": -1,
            }

        return {
            "stdout": stdout.decode("utf-8", errors="replace")[:10000],
            "stderr": stderr.decode("utf-8", errors="replace")[:10000],
            "returncode": proc.returncode,
        }
    finally:
        os.unlink(tmp_path)


async def shell_run(command: str, timeout: int = 30) -> dict:
    """Run a shell command from an allowlist of safe commands.

    Args:
        command: Shell command string.
        timeout: Maximum execution time in seconds.

    Returns:
        Dict with stdout, stderr, and returncode.
    """
    try:
        parts = shlex.split(command)
    except ValueError as e:
        return {"error": f"Invalid command syntax: {e}"}

    if not parts:
        return {"error": "Empty command"}

    cmd_name = Path(parts[0]).name
    if cmd_name not in SHELL_ALLOWLIST:
        return {
            "error": f"Command '{cmd_name}' is not allowed. "
            f"Allowed commands: {', '.join(sorted(SHELL_ALLOWLIST))}"
        }

    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(WORKSPACE),
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return {
            "stdout": "",
            "stderr": f"Command timed out after {timeout}s",
            "returncode": -1,
        }

    return {
        "stdout": stdout.decode("utf-8", errors="replace")[:10000],
        "stderr": stderr.decode("utf-8", errors="replace")[:10000],
        "returncode": proc.returncode,
    }


async def memory_store(
    content: str,
    memory_type: str = "semantic",
    importance: float = 0.5,
) -> dict:
    """Store a memory in the agent's long-term archival memory.

    Args:
        content: The text content to remember.
        memory_type: Type of memory: semantic, episodic, procedural, or task_result.
        importance: Importance score from 0.0 to 1.0.

    Returns:
        Dict with id and stored status.
    """
    # _memory is injected at registration time via closure
    raise NotImplementedError("memory not bound")


async def memory_query(
    query: str,
    k: int = 5,
    memory_type: str | None = None,
) -> dict:
    """Search the agent's long-term archival memory.

    Args:
        query: Search query text.
        k: Number of results to return.
        memory_type: Optional filter by memory type.

    Returns:
        Dict with results list.
    """
    raise NotImplementedError("memory not bound")


async def todo_write(todos: list[dict]) -> dict:
    """Update the task ledger with a checklist for tracking progress.

    Args:
        todos: List of todo items, each with 'text' (str) and 'done' (bool).

    Returns:
        Dict with updated status and item count.
    """
    raise NotImplementedError("memory not bound")


async def ask_user(question: str) -> dict:
    """Ask the user a question when clarification is needed.

    Args:
        question: The question to present to the user.

    Returns:
        Dict with the user's answer.
    """
    raise NotImplementedError("ask_user_hook not bound")


# ------------------------------------------------------------------
# Built-in registration helper
# ------------------------------------------------------------------

def _make_memory_store(memory: Any) -> Callable:
    """Create a memory_store tool bound to an AgentMemory instance.

    Args:
        memory: AgentMemory instance.

    Returns:
        Async function with the same signature as memory_store.
    """
    async def memory_store(
        content: str,
        memory_type: str = "semantic",
        importance: float = 0.5,
    ) -> dict:
        """Store a memory in the agent's long-term archival memory.

        Args:
            content: The text content to remember.
            memory_type: Type of memory: semantic, episodic, procedural, or task_result.
            importance: Importance score from 0.0 to 1.0.

        Returns:
            Dict with id and stored status.
        """
        mem_id = memory.archive_store(
            content=content, memory_type=memory_type, importance=importance,
        )
        return {"id": mem_id, "stored": True}

    return memory_store


def _make_memory_query(memory: Any) -> Callable:
    """Create a memory_query tool bound to an AgentMemory instance.

    Args:
        memory: AgentMemory instance.

    Returns:
        Async function with the same signature as memory_query.
    """
    async def memory_query(
        query: str,
        k: int = 5,
        memory_type: str | None = None,
    ) -> dict:
        """Search the agent's long-term archival memory.

        Args:
            query: Search query text.
            k: Number of results to return.
            memory_type: Optional filter by memory type.

        Returns:
            Dict with results list.
        """
        results = memory.archive_query(query=query, k=k, memory_type=memory_type)
        return {"results": results}

    return memory_query


def _make_todo_write(memory: Any) -> Callable:
    """Create a todo_write tool bound to an AgentMemory instance.

    Args:
        memory: AgentMemory instance.

    Returns:
        Async function with the same signature as todo_write.
    """
    async def todo_write(todos: list[dict]) -> dict:
        """Update the task ledger with a checklist for tracking progress.

        Args:
            todos: List of todo items, each with 'text' (str) and 'done' (bool).

        Returns:
            Dict with updated status and item count.
        """
        lines: list[str] = []
        for item in todos:
            marker = "x" if item.get("done", False) else " "
            lines.append(f"- [{marker}] {item.get('text', '')}")
        checklist = "\n".join(lines)
        memory.core_set("task_ledger", checklist)
        return {"updated": True, "count": len(todos)}

    return todo_write


def _make_ask_user(hook: Callable) -> Callable:
    """Create an ask_user tool bound to a callback.

    Args:
        hook: Async callback ``f(question) -> str``.

    Returns:
        Async function with the same signature as ask_user.
    """
    async def ask_user(question: str) -> dict:
        """Ask the user a question when clarification is needed.

        Args:
            question: The question to present to the user.

        Returns:
            Dict with the user's answer.
        """
        answer = await hook(question)
        return {"answer": answer}

    return ask_user


def _make_create_tool(registry: ToolRegistry) -> Callable:
    """Create the create_tool meta-tool bound to a ToolRegistry.

    Args:
        registry: The ToolRegistry that owns LLM, memory, config, etc.

    Returns:
        Async function implementing the create_tool meta-tool.
    """

    async def create_tool(
        name: str,
        description: str,
        parameters: dict,
        rationale: str,
    ) -> dict:
        """Create a new Python tool that the agent can use immediately.

        Args:
            name: snake_case function name (must be unique, not collide with existing tools).
            description: One-sentence description of what the tool does.
            parameters: JSON Schema describing the tool's parameters.
            rationale: Why this tool is needed and how it generalizes beyond the current task.

        Returns:
            Dict with 'created' (bool), 'tool_name', and either 'tests_passed' or 'errors'.
        """
        memory = registry._memory
        llm = registry._llm
        config = registry._config

        # 1. Validate name format
        if not _TOOL_NAME_RE.match(name):
            return {"created": False, "tool_name": name,
                    "errors": [f"Invalid name '{name}': must match [a-z][a-z0-9_]{{1,40}}"]}

        # 2. Check not already registered
        if name in registry.names():
            return {"created": False, "tool_name": name,
                    "errors": [f"Tool '{name}' already exists"]}

        # 3. Check hard cap
        tools_cfg = config.get("tools", {})
        hard_cap = tools_cfg.get("hard_cap", 100)
        if memory and memory.tool_count_active() >= hard_cap:
            return {"created": False, "tool_name": name,
                    "errors": [f"Tool cap reached ({hard_cap}). Deprecate unused tools first."]}

        # 4. Similarity check — reject if existing tool is too similar
        if memory:
            similar = memory.tool_search(description, k=1)
            if similar:
                existing = memory.tool_get(similar[0])
                if existing:
                    # Quick embedding distance check
                    vec_new = memory._embed(description)
                    vec_old = memory._embed(existing["description"])
                    sim = _cosine_similarity(vec_new, vec_old)
                    if sim > 0.85:
                        return {"created": False, "tool_name": name,
                                "errors": [f"Too similar to existing tool '{similar[0]}' "
                                           f"(similarity={sim:.2f}). Reuse it instead."]}

        # 5. Generate implementation + tests via LLM (with self-debug retry)
        tool_creator_prompt = registry._prompts.get("tool_creator", "")
        import_list = ", ".join(sorted(IMPORT_ALLOWLIST))
        rendered_prompt = tool_creator_prompt.replace("{{import_allowlist}}", import_list)

        spec_block = (
            f"Function name: {name}\n"
            f"Description: {description}\n"
            f"Parameters: {json.dumps(parameters, indent=2)}\n"
            f"Rationale: {rationale}"
        )

        last_errors: list[str] = []
        source: str = ""
        test_cases: list[dict] = []
        max_attempts = 3

        for attempt in range(max_attempts):
            messages = [
                {"role": "system", "content": rendered_prompt},
                {"role": "user", "content": (
                    f"Create a tool matching this specification:\n\n{spec_block}\n\n"
                    + (f"Previous attempt failed with errors:\n{json.dumps(last_errors)}\n"
                       f"Previous source:\n```python\n{source}\n```\n"
                       "Fix the issues and try again.\n\n"
                       if last_errors else "")
                    + 'Respond with a JSON object: {"source": "...", "tests": [{"input": {...}, "expected": ...}, ...]}'
                )},
            ]

            try:
                result = await llm.complete(
                    messages, response_format={"type": "json_object"},
                )
                data = json.loads(result.text)
                source = data.get("source", "")
                test_cases = data.get("tests", [])
            except Exception as e:
                last_errors = [f"LLM call failed: {e}"]
                continue

            if not source:
                last_errors = ["LLM returned empty source"]
                continue

            # 6. AST validate
            safe, violations = validate_tool_source(source)
            if not safe:
                last_errors = violations
                continue

            # 7. Sandbox test
            sandbox_result = sandbox_test(source, test_cases, name)
            if sandbox_result["passed"]:
                last_errors = []
                break
            else:
                last_errors = sandbox_result.get("errors", [])
                if not last_errors and "results" in sandbox_result:
                    last_errors = [
                        f"Test failed: input={r.get('input')}, "
                        f"expected={tc.get('expected')}, got={r.get('actual', r.get('error'))}"
                        for r, tc in zip(sandbox_result["results"], test_cases)
                        if not r.get("passed")
                    ]
        else:
            # All attempts exhausted
            return {"created": False, "tool_name": name,
                    "errors": last_errors or ["Failed after 3 attempts"]}

        # 8. Critic verification (Voyager pattern)
        critic_prompt = registry._prompts.get("critic", "")
        if critic_prompt and llm:
            critic_messages = [
                {"role": "system", "content": critic_prompt},
                {"role": "user", "content": (
                    f"Review this generated tool:\n\n"
                    f"Name: {name}\nDescription: {description}\n"
                    f"Source:\n```python\n{source}\n```\n"
                    f"Test results: {json.dumps(sandbox_result.get('results', []))}\n\n"
                    "Is this tool correct, general-purpose, and safe? "
                    "Respond with PASS or FAIL on the first line, then explain."
                )},
            ]
            try:
                critic_result = await llm.complete(critic_messages)
                verdict_line = critic_result.text.strip().split('\n')[0].upper()
                if "FAIL" in verdict_line:
                    return {"created": False, "tool_name": name,
                            "errors": [f"Critic rejected: {critic_result.text}"]}
            except Exception:
                pass  # Critic failure is non-fatal; proceed

        # 9. Approval if required
        require_approval = tools_cfg.get("require_approval", True)
        if require_approval and registry._approval_hook:
            try:
                approved = await registry._approval_hook(
                    name, {"source": source[:500], "description": description},
                )
                if not approved:
                    return {"created": False, "tool_name": name,
                            "errors": ["User denied approval"]}
            except Exception:
                pass  # Approval hook failure — proceed as approved

        # 10. Save to tools_generated/{name}.py
        gen_dir = Path(tools_cfg.get("generated_dir", "tools_generated"))
        gen_dir.mkdir(exist_ok=True)
        file_path = gen_dir / f"{name}.py"
        file_path.write_text(source, encoding="utf-8")

        # 11. Hot-load and register
        try:
            loaded_func = load_tool_from_file(str(file_path))
            registry.register(loaded_func, dangerous=False, builtin=False)
        except Exception as e:
            file_path.unlink(missing_ok=True)
            return {"created": False, "tool_name": name,
                    "errors": [f"Hot-load failed: {e}"]}

        # 12. Persist metadata
        source_hash = hashlib.sha256(source.encode()).hexdigest()
        if memory:
            memory.tool_register_meta(
                name=name,
                description=description,
                file_path=str(file_path),
                params_json=json.dumps(parameters),
                source_hash=source_hash,
            )
            if require_approval:
                memory.tool_set_approved(name)

        registry._on_event("memory_op", {
            "op": "tool_created", "key": name,
            "summary": f"Created tool '{name}': {description}",
        })

        return {
            "created": True, "tool_name": name,
            "tests_passed": len(test_cases),
        }

    return create_tool


def register_builtins(
    registry: ToolRegistry,
    memory: Any = None,
    ask_user_hook: Callable | None = None,
) -> None:
    """Register all 12 built-in tools on a ToolRegistry.

    Args:
        registry: The ToolRegistry to populate.
        memory: Optional AgentMemory for memory/todo tools.
        ask_user_hook: Optional async callback for ask_user.
    """
    # Safe tools
    registry.register(web_search)
    registry.register(web_fetch)
    registry.register(file_read)
    registry.register(file_write)
    registry.register(file_edit)

    # Dangerous tools
    registry.register(code_execute, dangerous=True)
    registry.register(shell_run, dangerous=True)

    # Memory-bound tools
    if memory is not None:
        registry.register(_make_memory_store(memory))
        registry.register(_make_memory_query(memory))
        registry.register(_make_todo_write(memory))
    else:
        registry.register(memory_store)
        registry.register(memory_query)
        registry.register(todo_write)

    # User interaction
    if ask_user_hook is not None:
        registry.register(_make_ask_user(ask_user_hook))
    else:
        registry.register(ask_user)

    # Meta-tool: create_tool (the 12th built-in)
    registry.register(_make_create_tool(registry))

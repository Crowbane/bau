"""BAU TUI — Rich + prompt_toolkit interface (Phase 5)."""
from __future__ import annotations

import json

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.theme import Theme


# ------------------------------------------------------------------
# Theme
# ------------------------------------------------------------------

_THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "dim": "dim italic",
    "tool": "magenta",
    "memory": "blue",
    "step": "bold cyan",
})


# ------------------------------------------------------------------
# AgentUI
# ------------------------------------------------------------------

class AgentUI:
    """Terminal interface for BAU using Rich + prompt_toolkit.

    Renders agent events (plans, tool calls, memory ops, etc.) via Rich
    and handles user input via prompt_toolkit. Imports nothing from
    agent.py, llm.py, or tools.py — all data flows through plain dicts
    and strings.

    Args:
        config: Parsed config dict (from config.yaml).
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        ui_cfg = config.get("ui", {})

        self._console = Console(theme=_THEME, highlight=False)
        self._show_tokens = ui_cfg.get("show_token_usage", True)

        # prompt_toolkit session
        history_path = ".bau_history"
        self._bindings = self._build_keybindings()
        self._session: PromptSession[str] = PromptSession(
            history=FileHistory(history_path),
            auto_suggest=AutoSuggestFromHistory(),
            multiline=True,
            prompt_continuation="  ",
            enable_history_search=True,
            mouse_support=False,
            key_bindings=self._bindings,
        )

        # Interrupt flag — checked by the agent loop between iterations
        self._interrupted = False

        # Event dispatch table
        self._dispatch: dict[str, callable] = {
            "goal": self._render_goal,
            "plan": self._render_plan,
            "replan": self._render_replan,
            "step_start": self._render_step_start,
            "step_done": self._render_step_done,
            "thinking": self._render_thinking,
            "tool_call": self._render_tool_call,
            "tool_result": self._render_tool_result,
            "memory_op": self._render_memory_op,
            "warning": self._render_warning,
            "error": self._render_error,
            "done": self._render_done,
        }

    # ------------------------------------------------------------------
    # Keybindings
    # ------------------------------------------------------------------

    @staticmethod
    def _build_keybindings() -> KeyBindings:
        """Build prompt_toolkit key bindings.

        Returns:
            Configured KeyBindings object.
        """
        kb = KeyBindings()

        @kb.add("enter")
        def _submit(event):
            """Submit on Enter (single-line mode feel)."""
            buf = event.current_buffer
            # If text ends with backslash, allow continuation
            if buf.text.endswith("\\"):
                buf.insert_text("\n")
            else:
                buf.validate_and_handle()

        @kb.add("escape", "enter")
        def _newline(event):
            """Insert newline on Esc+Enter for multiline input."""
            event.current_buffer.insert_text("\n")

        return kb

    # ------------------------------------------------------------------
    # Input methods
    # ------------------------------------------------------------------

    async def get_user_input(self, prompt: str = "› ") -> str:
        """Get input from the user via prompt_toolkit.

        Args:
            prompt: The prompt string to display.

        Returns:
            User input string.
        """
        with patch_stdout():
            text = await self._session.prompt_async(
                HTML(f"<b>{prompt}</b>"),
            )
        return text

    async def confirm(self, message: str) -> bool:
        """Ask for yes/no confirmation (for tool approval).

        Args:
            message: Confirmation question.

        Returns:
            True if user confirms.
        """
        self._console.print(f"[warning]{message}[/]")
        with patch_stdout():
            answer = await self._session.prompt_async(
                HTML("<b>[y/n] › </b>"),
                multiline=False,
            )
        return answer.strip().lower() in ("y", "yes")

    async def ask_user(self, question: str) -> str:
        """Present an agent question to the user and return their answer.

        Args:
            question: The question text from the agent.

        Returns:
            User's answer string.
        """
        self._console.print(
            Panel(question, title="[bold]Agent asks[/]", border_style="cyan"),
        )
        with patch_stdout():
            answer = await self._session.prompt_async(
                HTML("<b>answer › </b>"),
                multiline=False,
            )
        return answer.strip()

    # ------------------------------------------------------------------
    # Event handler (main dispatch)
    # ------------------------------------------------------------------

    def on_event(self, event_type: str, payload: dict) -> None:
        """Handle an agent event by dispatching to the appropriate renderer.

        Args:
            event_type: One of the documented event types.
            payload: Event-specific data dict.
        """
        handler = self._dispatch.get(event_type)
        if handler:
            try:
                handler(payload)
            except Exception as e:
                self._console.print(f"[error]UI render error ({event_type}): {e}[/]")
        else:
            # Unknown event — show it dimmed
            self._console.print(f"[dim][{event_type}] {payload}[/]")

    # ------------------------------------------------------------------
    # Event renderers
    # ------------------------------------------------------------------

    def _render_goal(self, payload: dict) -> None:
        """Render a goal event."""
        self._console.print()
        self._console.print(
            Panel(
                f"[bold]{payload['goal']}[/]",
                title="\U0001f3af Goal",
                border_style="cyan",
                padding=(0, 1),
            ),
        )

    def _render_plan(self, payload: dict) -> None:
        """Render a plan event as a numbered list."""
        steps = payload.get("steps", [])
        lines = [f"  {i}. {step}" for i, step in enumerate(steps, 1)]
        body = "\n".join(lines) if lines else "(empty plan)"
        self._console.print(
            Panel(body, title="[bold]Plan[/]", border_style="cyan", padding=(0, 1)),
        )

    def _render_replan(self, payload: dict) -> None:
        """Render a replan notice."""
        self._console.print(
            f"\n[warning]\u21bb Replanning: {payload.get('reason', '?')}[/]",
        )

    def _render_step_start(self, payload: dict) -> None:
        """Render step start."""
        idx = payload.get("index", 0)
        desc = payload.get("description", "")
        total = payload.get("total", "?")
        self._console.print(f"\n[step]Step {idx + 1}/{total}:[/] {desc}")

    def _render_step_done(self, payload: dict) -> None:
        """Render step completion."""
        idx = payload.get("index", 0)
        result = payload.get("result", {})
        brief = str(result.get("text", "done"))[:200]
        self._console.print(f"[success]\u2713[/] Step {idx + 1} done — {brief}")

    def _render_thinking(self, payload: dict) -> None:
        """Render thinking text (dimmed)."""
        text = payload.get("text", "")
        if text:
            # Truncate long thinking text
            if len(text) > 500:
                text = text[:500] + "..."
            self._console.print(f"[dim]{text}[/]")

    def _render_tool_call(self, payload: dict) -> None:
        """Render a tool call with pretty-printed args."""
        name = payload.get("name", "?")
        args = payload.get("args", {})
        args_str = _format_args(args)
        self._console.print(f"  [tool]\u2192 {name}[/]({args_str})")

    def _render_tool_result(self, payload: dict) -> None:
        """Render a tool result in a panel."""
        name = payload.get("name", "")
        result = str(payload.get("result", ""))

        # Truncate very long results
        if len(result) > 2000:
            result = result[:2000] + "\n... (truncated)"

        # Try to syntax-highlight JSON results
        content = _maybe_syntax(result)
        self._console.print(
            Panel(
                content,
                title=f"[dim]{name} result[/]",
                border_style="dim",
                padding=(0, 1),
            ),
        )

    def _render_memory_op(self, payload: dict) -> None:
        """Render a memory operation."""
        op = payload.get("op", "?")
        summary = payload.get("summary", payload.get("key", ""))
        self._console.print(f"  [memory]\U0001f9e0 {op}: {summary}[/]")

    def _render_warning(self, payload: dict) -> None:
        """Render a warning message."""
        self._console.print(f"[warning]\u26a0 {payload.get('message', '')}[/]")

    def _render_error(self, payload: dict) -> None:
        """Render an error message."""
        self._console.print(f"[error]\u2717 {payload.get('message', '')}[/]")

    def _render_done(self, payload: dict) -> None:
        """Render the final answer."""
        status = payload.get("status", "done")
        answer = payload.get("answer", "")
        style = "green" if status == "done" else "red"
        self._console.print()
        self._console.print(
            Panel(
                answer,
                title=f"[bold {style}]\u2713 Done ({status})[/]",
                border_style=style,
                padding=(0, 1),
            ),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def banner(self, model: str = "", tool_count: int = 0) -> None:
        """Display the startup banner.

        Args:
            model: Model identifier string.
            tool_count: Number of registered tools.
        """
        self._console.print()
        self._console.print(
            Panel(
                "[bold cyan]BAU[/] — autonomous agent\n"
                f"[dim]model:[/] {model or '?'}  "
                f"[dim]tools:[/] {tool_count}",
                border_style="cyan",
                padding=(0, 1),
            ),
        )
        self._console.print(
            "[dim]Enter a goal. "
            "Press Enter to submit, Esc+Enter for newline. "
            "Type /help for commands.[/]\n",
        )

    def show_stats(self, stats: dict) -> None:
        """Display agent/memory/tool statistics.

        Args:
            stats: Combined stats dict from agent, memory, and tools.
        """
        lines: list[str] = []

        # LLM / cost
        llm = stats.get("llm", {})
        if llm:
            cost = llm.get("total_cost", 0)
            prompt_t = llm.get("prompt_tokens", 0)
            comp_t = llm.get("completion_tokens", 0)
            lines.append(f"[bold]LLM[/]  tokens: {prompt_t:,} in / {comp_t:,} out  cost: ${cost:.4f}")

        # Memory
        mem = stats.get("memory", {})
        counts = mem.get("counts", {})
        if counts:
            db_kb = mem.get("db_size_bytes", 0) / 1024
            lines.append(
                f"[bold]Memory[/]  archival: {counts.get('memories', 0)}  "
                f"conversation: {counts.get('conversation', 0)}  "
                f"checkpoints: {counts.get('checkpoints', 0)}  "
                f"db: {db_kb:.0f} KB",
            )

        # Tools
        tool_stats = stats.get("tools", {})
        if tool_stats:
            total_usage = sum(t.get("usage", 0) for t in tool_stats.values())
            total_success = sum(t.get("success", 0) for t in tool_stats.values())
            lines.append(
                f"[bold]Tools[/]  registered: {len(tool_stats)}  "
                f"calls: {total_usage}  success: {total_success}",
            )

        body = "\n".join(lines) if lines else "(no stats available)"
        self._console.print(
            Panel(body, title="[bold]Stats[/]", border_style="cyan", padding=(0, 1)),
        )

    def show_tools(self, tool_stats: dict) -> None:
        """Display tool list with usage stats.

        Args:
            tool_stats: Dict mapping tool name to stats dict.
        """
        if not tool_stats:
            self._console.print("[dim]No tools registered.[/]")
            return

        lines: list[str] = []
        for name, ts in sorted(tool_stats.items()):
            usage = ts.get("usage", 0)
            success = ts.get("success", 0)
            failure = ts.get("failure", 0)
            rate = f"{success / usage * 100:.0f}%" if usage > 0 else "-"
            marker = "[dim](builtin)[/]" if ts.get("builtin") else "[yellow](dynamic)[/]"
            danger = " [red]⚠[/]" if ts.get("dangerous") else ""
            lines.append(
                f"  {name:<20s} {marker}  "
                f"used: {usage}  success: {rate}{danger}",
            )

        body = "\n".join(lines)
        self._console.print(
            Panel(body, title="[bold]Tools[/]", border_style="magenta", padding=(0, 1)),
        )

    def show_memory_results(self, results: list[dict]) -> None:
        """Display memory search results.

        Args:
            results: List of memory result dicts from archive_query.
        """
        if not results:
            self._console.print("[dim]No memories found.[/]")
            return

        for r in results:
            mid = r.get("id", "?")
            mtype = r.get("memory_type", "?")
            content = r.get("content", "")[:300]
            self._console.print(
                f"  [memory]#{mid}[/] [{mtype}] {content}",
            )

    def show_help(self) -> None:
        """Display available slash commands."""
        help_text = (
            "[bold]/quit, /exit[/]     — exit BAU\n"
            "[bold]/stats[/]           — show token, cost, memory, tool stats\n"
            "[bold]/tools[/]           — list registered tools with success rates\n"
            "[bold]/memory <query>[/]  — search archival memory\n"
            "[bold]/clear[/]           — clear the screen\n"
            "[bold]/reset[/]           — abandon current run, clear scratchpad\n"
            "[bold]/help[/]            — show this help"
        )
        self._console.print(
            Panel(help_text, title="[bold]Commands[/]", border_style="cyan", padding=(0, 1)),
        )

    def shutdown(self) -> None:
        """Display shutdown message."""
        self._console.print("\n[dim]BAU offline.[/]")

    @property
    def interrupted(self) -> bool:
        """Check and clear the interrupt flag."""
        if self._interrupted:
            self._interrupted = False
            return True
        return False


# ------------------------------------------------------------------
# Slash command parser
# ------------------------------------------------------------------

def parse_slash_command(text: str) -> tuple[str, str]:
    """Parse a slash command from user input.

    Args:
        text: Raw input string.

    Returns:
        Tuple of (command, args). Command is empty string if not a slash command.
    """
    stripped = text.strip()
    if not stripped.startswith("/"):
        return ("", stripped)

    parts = stripped.split(maxsplit=1)
    command = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""
    return (command, args)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _format_args(args: dict) -> str:
    """Pretty-format tool call arguments (compact but readable).

    Args:
        args: Tool arguments dict.

    Returns:
        Formatted string.
    """
    if not args:
        return ""
    parts: list[str] = []
    for k, v in args.items():
        val = repr(v) if isinstance(v, str) else str(v)
        # Truncate long values
        if len(val) > 80:
            val = val[:77] + "..."
        parts.append(f"{k}={val}")
    return ", ".join(parts)


def _maybe_syntax(text: str) -> str | Syntax:
    """Try to parse text as JSON and return Syntax-highlighted version.

    Args:
        text: Possibly-JSON text.

    Returns:
        Syntax object if valid JSON, otherwise the original string.
    """
    stripped = text.strip()
    if stripped and stripped[0] in ("{", "["):
        try:
            parsed = json.loads(stripped)
            formatted = json.dumps(parsed, indent=2)
            return Syntax(formatted, "json", theme="monokai", word_wrap=True)
        except (json.JSONDecodeError, TypeError):
            pass
    return text

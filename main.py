"""BAU — Entry point, config loading, lifecycle."""
from __future__ import annotations

import asyncio
import signal
from pathlib import Path

import yaml


def load_config(path: str = "config.yaml") -> dict:
    """Load and validate the YAML configuration file.

    Args:
        path: Path to the config file.

    Returns:
        Parsed config dictionary.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If required config keys are missing.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if not config:
        raise ValueError(f"Config file is empty: {path}")
    if "model" not in config:
        raise ValueError("Config missing required 'model' section")

    model = config["model"]
    for key in ("provider", "name"):
        if key not in model:
            raise ValueError(f"Config 'model' section missing required key: '{key}'")

    return config


def load_prompts(directory: str = "prompts") -> dict[str, str]:
    """Load all markdown prompt files from a directory.

    Args:
        directory: Path to the prompts directory.

    Returns:
        Dict mapping prompt name (file stem) to content string.

    Raises:
        FileNotFoundError: If prompts directory doesn't exist.
        ValueError: If no .md files are found.
    """
    prompts_dir = Path(directory)
    if not prompts_dir.is_dir():
        raise FileNotFoundError(f"Prompts directory not found: {directory}")

    prompts: dict[str, str] = {}
    for md_file in sorted(prompts_dir.glob("*.md")):
        prompts[md_file.stem] = md_file.read_text()

    if not prompts:
        raise ValueError(f"No .md files found in {directory}")

    return prompts


def render_prompt(template: str, **variables: str) -> str:
    """Replace {{placeholder}} tokens in a prompt template.

    Args:
        template: Prompt text with {{key}} placeholders.
        **variables: Key-value pairs for substitution.

    Returns:
        Rendered prompt string with placeholders replaced.
    """
    result = template
    for key, value in variables.items():
        result = result.replace(f"{{{{{key}}}}}", str(value))
    return result


# ------------------------------------------------------------------
# REPL
# ------------------------------------------------------------------

async def repl(ui, agent, memory, reset_sigint=None) -> None:
    """Main interactive loop: read goals, run agent, handle commands.

    Args:
        ui: AgentUI instance.
        agent: Agent instance.
        memory: AgentMemory instance (for /memory search).
        reset_sigint: Optional callable to reset the Ctrl-C counter.
    """
    from ui import parse_slash_command

    model = f"{agent._config['model']['provider']}/{agent._config['model']['name']}"
    tool_count = len(agent._tools.names())
    ui.banner(model=model, tool_count=tool_count)

    while True:
        try:
            raw = await ui.get_user_input()
        except (EOFError, KeyboardInterrupt):
            break

        if not raw.strip():
            continue

        cmd, args = parse_slash_command(raw)

        match cmd:
            case "/quit" | "/exit":
                break
            case "/help":
                ui.show_help()
            case "/stats":
                ui.show_stats(agent.stats())
            case "/tools":
                ui.show_tools(agent._tools.stats())
            case "/memory":
                if args:
                    results = memory.archive_query(args, k=5)
                    ui.show_memory_results(results)
                else:
                    ui.on_event("warning", {"message": "Usage: /memory <query>"})
            case "/clear":
                ui._console.clear()
            case "/reset":
                agent.interrupt()
                ui.on_event("warning", {"message": "Run abandoned."})
            case "":
                # Not a slash command — treat as a goal
                try:
                    await agent.run(raw.strip())
                except KeyboardInterrupt:
                    ui.on_event("warning", {"message": "interrupted"})
                    agent.interrupt()
                except Exception as e:
                    ui.on_event("error", {"message": f"{type(e).__name__}: {e}"})
                finally:
                    if reset_sigint:
                        reset_sigint()
            case _:
                ui.on_event("warning", {"message": f"Unknown command: {cmd}. Type /help."})

    ui.shutdown()


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

async def main() -> None:
    """Entry point — initialize all components and run the REPL."""
    import logging

    from agent import Agent, AgentMemory
    from llm import LLMClient
    from tools import ToolRegistry, load_tool_from_file, register_builtins
    from ui import AgentUI

    log = logging.getLogger("bau")

    config = load_config("config.yaml")
    prompts = load_prompts("prompts/")

    # Ensure data, tools_generated, and workspace directories exist
    Path("data").mkdir(exist_ok=True)
    Path("workspace").mkdir(exist_ok=True)
    tools_dir = Path(config.get("tools", {}).get("generated_dir", "tools_generated"))
    tools_dir.mkdir(exist_ok=True)

    llm = LLMClient(config)
    db_path = config.get("memory", {}).get("db_path", "data/bau.db")
    memory = AgentMemory(db_path=db_path)

    # Build the UI
    ui = AgentUI(config)

    # Approval and ask_user callbacks wired through the UI
    async def approval_hook(name: str, args: dict) -> bool:
        return await ui.confirm(f"Allow tool [bold]{name}[/] with args {args}?")

    async def ask_user_hook(question: str) -> str:
        return await ui.ask_user(question)

    # Build tool registry with all 12 built-ins
    registry = ToolRegistry(
        memory=memory,
        approval_hook=approval_hook,
        ask_user_hook=ask_user_hook,
    )
    # Wire LLM, prompts, config, and event callback into registry for create_tool
    registry._llm = llm
    registry._prompts = prompts
    registry._config = config
    registry._on_event = ui.on_event

    register_builtins(registry, memory=memory, ask_user_hook=ask_user_hook)

    # Stale-tool deprecation (TroVE discipline) — run on startup
    stale_count = memory.tool_deprecate_stale(max_age_days=30)
    if stale_count > 0:
        log.info("Deprecated %d stale tools", stale_count)

    # Restore generated tools from previous sessions
    for row in memory.list_tools(deprecated=False):
        try:
            func = load_tool_from_file(row["file_path"])
            registry.register(func, dangerous=False, builtin=False)
        except Exception as e:
            log.warning("Failed to restore tool %s: %s", row["name"], e)

    agent = Agent(
        llm=llm,
        memory=memory,
        tools=registry,
        prompts=prompts,
        config=config,
        on_event=ui.on_event,
    )

    # Install SIGINT handler: first Ctrl-C sets the agent's interrupt flag
    # (graceful stop between iterations); second Ctrl-C raises KeyboardInterrupt.
    loop = asyncio.get_running_loop()
    _sigint_count = 0

    def _handle_sigint():
        nonlocal _sigint_count
        _sigint_count += 1
        if _sigint_count == 1:
            agent.interrupt()
            ui.on_event("warning", {"message": "interrupted — finishing current step…"})
        else:
            raise KeyboardInterrupt

    def _reset_sigint():
        nonlocal _sigint_count
        _sigint_count = 0

    loop.add_signal_handler(signal.SIGINT, _handle_sigint)

    try:
        await repl(ui, agent, memory, reset_sigint=_reset_sigint)
    finally:
        loop.remove_signal_handler(signal.SIGINT)
        memory.close()


if __name__ == "__main__":
    asyncio.run(main())

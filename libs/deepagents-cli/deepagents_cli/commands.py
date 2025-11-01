"""Command handlers for slash commands and bash execution."""

import subprocess
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from .config import COLORS, DEEP_AGENTS_ASCII, console
from .ui import (
    TokenTracker,
    extract_dmail_timeline,
    render_dmail_audit,
    resolve_checkpoint_alias,
    show_interactive_help,
)


def handle_command(command: str, agent, token_tracker: TokenTracker, agent_dir: Path | None = None, enable_dmail: bool = False) -> str | bool:
    """Handle slash commands. Returns 'exit' to exit, True if handled, False to pass to agent.

    Args:
        command: The command string (with or without leading /)
        agent: The agent instance
        token_tracker: Token usage tracker
        agent_dir: Directory for agent config (required for /dmail config)
        enable_dmail: Whether D-Mail is currently enabled

    Returns:
        'exit' to exit, True if handled, False to pass to agent
    """
    parts = command.strip().lstrip("/").split()
    if not parts:
        return False

    cmd = parts[0].lower()
    subcmd = parts[1] if len(parts) > 1 else None

    if cmd in ["quit", "exit", "q"]:
        return "exit"

    if cmd == "clear":
        # Reset agent conversation state
        agent.checkpointer = InMemorySaver()

        # Reset token tracking to baseline
        token_tracker.reset()

        # Clear screen and show fresh UI
        console.clear()
        console.print(DEEP_AGENTS_ASCII, style=f"bold {COLORS['primary']}")
        console.print()
        console.print(
            "... Fresh start! Screen cleared and conversation reset.", style=COLORS["agent"]
        )
        console.print()
        return True

    if cmd == "help":
        show_interactive_help()
        return True

    if cmd == "tokens":
        token_tracker.display_session()
        return True

    if cmd == "dmail":
        # Access agent state using the same thread_id as execution.py for consistency
        config_obj = {"configurable": {"thread_id": "main"}}
        state = agent.get_state(config_obj).values
        checkpoints = state.get("dmail_checkpoints", [])
        timeline = extract_dmail_timeline(state)

        if subcmd == "checkpoints":
            if checkpoints:
                console.print()
                console.print("[bold]Checkpoint Aliases:[/bold]", style=COLORS["primary"])
                console.print()
                for cp in reversed(checkpoints[-10:]):
                    name = cp.get("name", "unnamed")
                    reason = cp.get("reason", "")
                    created_at = cp.get("created_at", "")
                    time_str = created_at.split("T")[1][:8] if "T" in created_at else created_at
                    console.print(f"  {name:20} {time_str:12} {reason}", style="dim")
                console.print()
            else:
                console.print()
                console.print("[dim]No checkpoints yet[/dim]")
                console.print()
            return True

        if subcmd == "log":
            if timeline:
                console.print()
                console.print("[bold]D-Mail Timeline:[/bold]", style=COLORS["primary"])
                console.print()
                for entry in timeline:
                    from_id = entry.get("from_checkpoint_id", "unknown")
                    to_id = entry.get("to_checkpoint_id", "unknown")

                    from_alias = resolve_checkpoint_alias(from_id, checkpoints)
                    to_alias = resolve_checkpoint_alias(to_id, checkpoints)
                    message = entry.get("message", "")
                    preview = message[:80] + "..." if len(message) > 80 else message
                    console.print(f"  {from_alias} → {to_alias}")
                    console.print(f"    {preview!r}", style="dim")
                console.print()
            else:
                console.print()
                console.print("[dim]No D-Mail rewinds yet[/dim]")
                console.print()
            return True

        if subcmd == "config":
            if not agent_dir:
                console.print()
                console.print("[red]Error:[/red] agent_dir not provided")
                console.print()
                return True

            from .config import load_config, save_config

            config = load_config(agent_dir)
            args = parts[2:]

            if not args:
                console.print()
                console.print("[bold]D-Mail Configuration:[/bold]", style=COLORS["primary"])
                console.print(f"  enabled: {config.get('dmail', {}).get('enabled', False)}")
                console.print(f"  auto_checkpoints: {config.get('dmail', {}).get('auto_checkpoints', True)}")
                console.print(f"  max_auto_per_run: {config.get('dmail', {}).get('max_auto_per_run', 3)}")
                console.print(f"  before_subagent: {config.get('dmail', {}).get('before_subagent', True)}")
                console.print(
                    f"  before_code_iteration: {config.get('dmail', {}).get('before_code_iteration', True)}"
                )
                console.print()
                return True

            if args[0] == "auto" and len(args) > 1:
                value = args[1].lower() in ("on", "true", "1")
                config.setdefault("dmail", {})["auto_checkpoints"] = value
                save_config(agent_dir, config)
                console.print()
                console.print(f"[green]✓[/green] auto_checkpoints set to {value}")
                console.print("[dim]Restart CLI for changes to take effect[/dim]")
                console.print()
                return True

            if args[0] == "max-auto" and len(args) > 1:
                try:
                    value = int(args[1])
                    config.setdefault("dmail", {})["max_auto_per_run"] = value
                    save_config(agent_dir, config)
                    console.print()
                    console.print(f"[green]✓[/green] max_auto_per_run set to {value}")
                    console.print("[dim]Restart CLI for changes to take effect[/dim]")
                    console.print()
                except ValueError:
                    console.print()
                    console.print("[red]Error:[/red] max-auto requires an integer")
                    console.print()
                return True

            console.print()
            console.print(f"[yellow]Unknown config option: {args[0]}[/yellow]")
            console.print("[dim]Available: auto on|off, max-auto N[/dim]")
            console.print()
            return True

        render_dmail_audit(checkpoints, timeline, enabled=enable_dmail)
        return True

    console.print()
    console.print(f"[yellow]Unknown command: /{cmd}[/yellow]")
    console.print("[dim]Type /help for available commands.[/dim]")
    console.print()
    return True


def execute_bash_command(command: str) -> bool:
    """Execute a bash command and display output. Returns True if handled."""
    cmd = command.strip().lstrip("!")

    if not cmd:
        return True

    try:
        console.print()
        console.print(f"[dim]$ {cmd}[/dim]")

        # Execute the command
        result = subprocess.run(
            cmd, check=False, shell=True, capture_output=True, text=True, timeout=30, cwd=Path.cwd()
        )

        # Display output
        if result.stdout:
            console.print(result.stdout, style=COLORS["dim"], markup=False)
        if result.stderr:
            console.print(result.stderr, style="red", markup=False)

        # Show return code if non-zero
        if result.returncode != 0:
            console.print(f"[dim]Exit code: {result.returncode}[/dim]")

        console.print()
        return True

    except subprocess.TimeoutExpired:
        console.print("[red]Command timed out after 30 seconds[/red]")
        console.print()
        return True
    except Exception as e:
        console.print(f"[red]Error executing command: {e}[/red]")
        console.print()
        return True

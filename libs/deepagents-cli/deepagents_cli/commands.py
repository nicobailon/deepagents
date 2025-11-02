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

    def _coerce_bool(value, default):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.lower()
            if lowered in {"true", "1", "yes", "on"}:
                return True
            if lowered in {"false", "0", "no", "off"}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        return default

    def _coerce_int(value, default):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

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
            dmail_cfg = config.get("dmail", {}) if isinstance(config.get("dmail"), dict) else {}
            args = parts[2:]

            def _parse_bool_arg(raw: str) -> bool:
                lowered = raw.lower()
                truthy = {"on", "true", "1", "yes", "enable", "enabled"}
                falsy = {"off", "false", "0", "no", "disable", "disabled"}
                if lowered in truthy:
                    return True
                if lowered in falsy:
                    return False
                raise ValueError

            if not args:
                console.print()
                console.print("[bold]D-Mail Configuration:[/bold]", style=COLORS["primary"])
                enabled = _coerce_bool(dmail_cfg.get("enabled"), False)
                auto_checkpoints = _coerce_bool(dmail_cfg.get("auto_checkpoints"), True)
                max_auto_before = _coerce_int(
                    dmail_cfg.get("max_auto_before_per_run", dmail_cfg.get("max_auto_per_run")), 15
                )
                max_auto_after = _coerce_int(dmail_cfg.get("max_auto_after_per_run"), 20)

                legacy_before_values = {
                    key: dmail_cfg[key]
                    for key in ("before_subagent", "before_code_iteration")
                    if key in dmail_cfg
                }

                if "before_every_tool" in dmail_cfg:
                    before_every_tool = _coerce_bool(dmail_cfg.get("before_every_tool"), True)
                elif legacy_before_values:
                    before_every_tool = any(_coerce_bool(val, False) for val in legacy_before_values.values())
                else:
                    before_every_tool = True

                after_every_tool = _coerce_bool(dmail_cfg.get("after_every_tool"), True)
                after_agent_response = _coerce_bool(dmail_cfg.get("after_agent_response"), True)
                before_first_user_message = _coerce_bool(dmail_cfg.get("before_first_user_message"), True)

                console.print(f"  enabled: {enabled}")
                console.print(f"  auto_checkpoints: {auto_checkpoints}")
                console.print(f"  max_auto_before_per_run: {max_auto_before}")
                console.print(f"  max_auto_after_per_run: {max_auto_after}")
                console.print(f"  before_every_tool: {before_every_tool}")
                console.print(f"  after_every_tool: {after_every_tool}")
                console.print(f"  after_agent_response: {after_agent_response}")
                console.print(f"  before_first_user_message: {before_first_user_message}")

                legacy_notes = []
                if "max_auto_per_run" in dmail_cfg:
                    legacy_notes.append(f"max_auto_per_run={dmail_cfg['max_auto_per_run']}")
                if legacy_before_values:
                    for key, val in legacy_before_values.items():
                        legacy_notes.append(f"{key}={val}")

                if legacy_notes:
                    console.print()
                    console.print(
                        "[yellow]Legacy keys detected:[/yellow] " + ", ".join(legacy_notes),
                        style=COLORS["dim"],
                    )
                    console.print(
                        "[dim]These values are ignored once new fields are set; use the options above.[/dim]"
                    )

                console.print()
                return True

            option = args[0].lower()
            dmail_section = config.setdefault("dmail", {})

            if option == "auto" and len(args) > 1:
                value = _parse_bool_arg(args[1])
                dmail_section["auto_checkpoints"] = value
                save_config(agent_dir, config)
                console.print()
                console.print(f"[green]✓[/green] auto_checkpoints set to {value}")
                console.print("[dim]Restart CLI for changes to take effect[/dim]")
                console.print()
                return True

            if option in {"max-auto-before", "max-auto"} and len(args) > 1:
                try:
                    value = int(args[1])
                except ValueError:
                    console.print()
                    console.print("[red]Error:[/red] max-auto-before requires an integer")
                    console.print()
                    return True

                dmail_section["max_auto_before_per_run"] = value
                save_config(agent_dir, config)
                console.print()
                console.print(f"[green]✓[/green] max_auto_before_per_run set to {value}")
                if option == "max-auto":
                    console.print("[dim]Note: max-auto is deprecated; use max-auto-before going forward.[/dim]")
                console.print("[dim]Restart CLI for changes to take effect[/dim]")
                console.print()
                return True

            if option == "max-auto-after" and len(args) > 1:
                try:
                    value = int(args[1])
                except ValueError:
                    console.print()
                    console.print("[red]Error:[/red] max-auto-after requires an integer")
                    console.print()
                    return True

                dmail_section["max_auto_after_per_run"] = value
                save_config(agent_dir, config)
                console.print()
                console.print(f"[green]✓[/green] max_auto_after_per_run set to {value}")
                console.print("[dim]Restart CLI for changes to take effect[/dim]")
                console.print()
                return True

            toggle_map = {
                "before-every-tool": "before_every_tool",
                "after-every-tool": "after_every_tool",
                "after-response": "after_agent_response",
                "before-first-message": "before_first_user_message",
            }

            if option in toggle_map and len(args) > 1:
                try:
                    value = _parse_bool_arg(args[1])
                except ValueError:
                    console.print()
                    console.print("[red]Error:[/red] Value must be on/off")
                    console.print()
                    return True

                dmail_section[toggle_map[option]] = value
                save_config(agent_dir, config)
                console.print()
                console.print(f"[green]✓[/green] {toggle_map[option]} set to {value}")
                console.print("[dim]Restart CLI for changes to take effect[/dim]")
                console.print()
                return True

            console.print()
            console.print(f"[yellow]Unknown config option: {args[0]}[/yellow]")
            console.print(
                "[dim]Available: auto on|off, max-auto-before N, max-auto-after N, before-every-tool on|off, after-every-tool on|off, after-response on|off, before-first-message on|off[/dim]"
            )
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

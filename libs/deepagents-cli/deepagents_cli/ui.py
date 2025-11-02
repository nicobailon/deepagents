"""UI rendering and display utilities for the CLI."""

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import SystemMessage
from rich import box
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .config import COLORS, COMMANDS, DEEP_AGENTS_ASCII, MAX_ARG_LENGTH, console
from .file_ops import FileOperationRecord


def truncate_value(value: str, max_length: int = MAX_ARG_LENGTH) -> str:
    """Truncate a string value if it exceeds max_length."""
    if len(value) > max_length:
        return value[:max_length] + "..."
    return value


def format_tool_display(tool_name: str, tool_args: dict) -> str:
    """Format tool calls for display with tool-specific smart formatting.

    Shows the most relevant information for each tool type rather than all arguments.

    Args:
        tool_name: Name of the tool being called
        tool_args: Dictionary of tool arguments

    Returns:
        Formatted string for display (e.g., "read_file(config.py)")

    Examples:
        read_file(path="/long/path/file.py") → "read_file(file.py)"
        web_search(query="how to code", max_results=5) → 'web_search("how to code")'
        shell(command="pip install foo") → 'shell("pip install foo")'
    """

    def abbreviate_path(path_str: str, max_length: int = 60) -> str:
        """Abbreviate a file path intelligently - show basename or relative path."""
        try:
            path = Path(path_str)

            # If it's just a filename (no directory parts), return as-is
            if len(path.parts) == 1:
                return path_str

            # Try to get relative path from current working directory
            try:
                rel_path = path.relative_to(Path.cwd())
                rel_str = str(rel_path)
                # Use relative if it's shorter and not too long
                if len(rel_str) < len(path_str) and len(rel_str) <= max_length:
                    return rel_str
            except (ValueError, Exception):
                pass

            # If absolute path is reasonable length, use it
            if len(path_str) <= max_length:
                return path_str

            # Otherwise, just show basename (filename only)
            return path.name
        except Exception:
            # Fallback to original string if any error
            return truncate_value(path_str, max_length)

    # Tool-specific formatting - show the most important argument(s)
    if tool_name in ("read_file", "write_file", "edit_file"):
        # File operations: show the primary file path argument (file_path or path)
        path_value = tool_args.get("file_path")
        if path_value is None:
            path_value = tool_args.get("path")
        if path_value is not None:
            path = abbreviate_path(str(path_value))
            return f"{tool_name}({path})"

    elif tool_name == "web_search":
        # Web search: show the query string
        if "query" in tool_args:
            query = str(tool_args["query"])
            query = truncate_value(query, 100)
            return f'{tool_name}("{query}")'

    elif tool_name == "grep":
        # Grep: show the search pattern
        if "pattern" in tool_args:
            pattern = str(tool_args["pattern"])
            pattern = truncate_value(pattern, 70)
            return f'{tool_name}("{pattern}")'

    elif tool_name == "shell":
        # Shell: show the command being executed
        if "command" in tool_args:
            command = str(tool_args["command"])
            command = truncate_value(command, 120)
            return f'{tool_name}("{command}")'

    elif tool_name == "ls":
        # ls: show directory, or empty if current directory
        if tool_args.get("path"):
            path = abbreviate_path(str(tool_args["path"]))
            return f"{tool_name}({path})"
        return f"{tool_name}()"

    elif tool_name == "glob":
        # Glob: show the pattern
        if "pattern" in tool_args:
            pattern = str(tool_args["pattern"])
            pattern = truncate_value(pattern, 80)
            return f'{tool_name}("{pattern}")'

    elif tool_name == "http_request":
        # HTTP: show method and URL
        parts = []
        if "method" in tool_args:
            parts.append(str(tool_args["method"]).upper())
        if "url" in tool_args:
            url = str(tool_args["url"])
            url = truncate_value(url, 80)
            parts.append(url)
        if parts:
            return f"{tool_name}({' '.join(parts)})"

    elif tool_name == "task":
        # Task: show the task description
        if "description" in tool_args:
            desc = str(tool_args["description"])
            desc = truncate_value(desc, 100)
            return f'{tool_name}("{desc}")'

    elif tool_name == "write_todos":
        # Todos: show count of items
        if "todos" in tool_args and isinstance(tool_args["todos"], list):
            count = len(tool_args["todos"])
            return f"{tool_name}({count} items)"

    # Fallback: generic formatting for unknown tools
    # Show all arguments in key=value format
    args_str = ", ".join(f"{k}={truncate_value(str(v), 50)}" for k, v in tool_args.items())
    return f"{tool_name}({args_str})"


def format_tool_message_content(content: Any) -> str:
    """Convert ToolMessage content into a printable string."""
    if content is None:
        return ""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            else:
                try:
                    parts.append(json.dumps(item))
                except Exception:
                    parts.append(str(item))
        return "\n".join(parts)
    return str(content)


class TokenTracker:
    """Track token usage across the conversation."""

    def __init__(self):
        self.baseline_context = 0  # Baseline system context (system + agent.md + tools)
        self.current_context = 0  # Total context including messages
        self.last_output = 0

    def set_baseline(self, tokens: int):
        """Set the baseline context token count.

        Args:
            tokens: The baseline token count (system prompt + agent.md + tools)
        """
        self.baseline_context = tokens
        self.current_context = tokens

    def reset(self):
        """Reset to baseline (for /clear command)."""
        self.current_context = self.baseline_context
        self.last_output = 0

    def add(self, input_tokens: int, output_tokens: int):
        """Add tokens from a response."""
        # input_tokens IS the current context size (what was sent to the model)
        self.current_context = input_tokens
        self.last_output = output_tokens

    def display_last(self):
        """Display current context size after this turn."""
        if self.last_output and self.last_output >= 1000:
            console.print(f"  Generated: {self.last_output:,} tokens", style="dim")
        if self.current_context:
            console.print(f"  Current context: {self.current_context:,} tokens", style="dim")

    def display_session(self):
        """Display current context size."""
        console.print("\n[bold]Token Usage:[/bold]", style=COLORS["primary"])

        # Check if we've had any actual API calls yet (current > baseline means we have conversation)
        has_conversation = self.current_context > self.baseline_context

        if self.baseline_context > 0:
            console.print(
                f"  Baseline: {self.baseline_context:,} tokens [dim](system + agent.md)[/dim]",
                style=COLORS["dim"],
            )

            if not has_conversation:
                # Before first message - warn that tools aren't counted yet
                console.print(
                    "  [dim]Note: Tool definitions (~5k tokens) included after first message[/dim]"
                )

        if has_conversation:
            tools_and_conversation = self.current_context - self.baseline_context
            console.print(
                f"  Tools + conversation: {tools_and_conversation:,} tokens", style=COLORS["dim"]
            )

        console.print(f"  Total: {self.current_context:,} tokens", style="bold " + COLORS["dim"])
        console.print()


def render_todo_list(todos: list[dict]) -> None:
    """Render todo list as a rich Panel with checkboxes."""
    if not todos:
        return

    lines = []
    for todo in todos:
        status = todo.get("status", "pending")
        content = todo.get("content", "")

        if status == "completed":
            icon = "☑"
            style = "green"
        elif status == "in_progress":
            icon = "⏳"
            style = "yellow"
        else:  # pending
            icon = "☐"
            style = "dim"

        lines.append(f"[{style}]{icon} {content}[/{style}]")

    panel = Panel(
        "\n".join(lines),
        title="[bold]Task List[/bold]",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(0, 1),
    )
    console.print(panel)


def render_summary_panel(summary_content: str) -> None:
    """Render conversation summary as a collapsible panel."""
    # Extract just the summary text, removing any metadata
    summary_text = summary_content.strip()

    # Truncate if very long
    if len(summary_text) > 500:
        preview = summary_text[:500] + "..."
    else:
        preview = summary_text

    panel = Panel(
        f"[dim]Context exceeded threshold. Conversation history summarized to preserve context.[/dim]\n\n"
        f"[yellow]Summary:[/yellow]\n{preview}\n\n"
        f"[dim]Recent messages kept in full. Continuing with reduced context...[/dim]",
        title="[bold yellow]⚠ Context Summarized[/bold yellow]",
        border_style="yellow",
        box=box.ROUNDED,
        padding=(1, 2),
    )
    console.print(panel)


def _format_line_span(start: int | None, end: int | None) -> str:
    if start is None and end is None:
        return ""
    if start is not None and end is None:
        return f"(starting at line {start})"
    if start is None and end is not None:
        return f"(through line {end})"
    if start == end:
        return f"(line {start})"
    return f"(lines {start}-{end})"


def render_file_operation(record: FileOperationRecord) -> None:
    """Render a concise summary of a filesystem tool call."""
    label_lookup = {
        "read_file": "Read",
        "write_file": "Write",
        "edit_file": "Update",
    }
    label = label_lookup.get(record.tool_name, record.tool_name)
    header = Text()
    header.append("⏺ ", style=COLORS["tool"])
    header.append(f"{label}({record.display_path})", style=f"bold {COLORS['tool']}")
    console.print(header)

    def _print_detail(message: str, *, style: str = COLORS["dim"]) -> None:
        detail = Text()
        detail.append("  ⎿  ", style=style)
        detail.append(message, style=style)
        console.print(detail)

    if record.status == "error":
        _print_detail(record.error or "Error executing file operation", style="red")
        return

    if record.tool_name == "read_file":
        lines = record.metrics.lines_read
        span = _format_line_span(record.metrics.start_line, record.metrics.end_line)
        detail = f"Read {lines} line{'s' if lines != 1 else ''}"
        if span:
            detail = f"{detail} {span}"
        _print_detail(detail)
    else:
        bytes_written = record.metrics.bytes_written
        if record.tool_name == "write_file":
            lines = record.metrics.lines_written
            detail = f"Wrote {lines} line{'s' if lines != 1 else ''}"
            if record.metrics.lines_added:
                detail = f"{detail} (+{record.metrics.lines_added})"
        else:
            added = record.metrics.lines_added
            removed = record.metrics.lines_removed
            detail = f"Edited {record.metrics.lines_written} total line{'s' if record.metrics.lines_written != 1 else ''}"
            if added or removed:
                detail = f"{detail} (+{added} / -{removed})"
        if bytes_written:
            detail = f"{detail} · {bytes_written} bytes"
        _print_detail(detail)

    if record.diff:
        render_diff(record)


def render_diff(record: FileOperationRecord) -> None:
    """Render diff for a file operation."""
    if not record.diff:
        return
    render_diff_block(record.diff, f"Diff {record.display_path}")


def render_diff_block(diff: str, title: str) -> None:
    """Render a diff string inside a Rich panel."""
    syntax = Syntax(diff, "diff", theme="monokai", line_numbers=False)
    panel = Panel(
        syntax, title=title, border_style=COLORS["primary"], box=box.ROUNDED, padding=(0, 1)
    )
    console.print(panel)


def show_interactive_help():
    """Show available commands during interactive session."""
    console.print()
    console.print("[bold]Interactive Commands:[/bold]", style=COLORS["primary"])
    console.print()

    for cmd, desc in COMMANDS.items():
        console.print(f"  /{cmd:<12} {desc}", style=COLORS["dim"])

    console.print()
    console.print("[bold]Editing Features:[/bold]", style=COLORS["primary"])
    console.print("  Enter           Submit your message", style=COLORS["dim"])
    console.print(
        "  Alt+Enter       Insert newline (Option+Enter on Mac, or ESC then Enter)",
        style=COLORS["dim"],
    )
    console.print(
        "  Ctrl+E          Open in external editor (nano by default)", style=COLORS["dim"]
    )
    console.print("  Ctrl+T          Toggle auto-approve mode", style=COLORS["dim"])
    console.print("  Arrow keys      Navigate input", style=COLORS["dim"])
    console.print("  Ctrl+C          Cancel input or interrupt agent mid-work", style=COLORS["dim"])
    console.print()
    console.print("[bold]Special Features:[/bold]", style=COLORS["primary"])
    console.print(
        "  @filename       Type @ to auto-complete files and inject content", style=COLORS["dim"]
    )
    console.print("  /command        Type / to see available commands", style=COLORS["dim"])
    console.print(
        "  !command        Type ! to run bash commands (e.g., !ls, !git status)",
        style=COLORS["dim"],
    )
    console.print(
        "                  Completions appear automatically as you type", style=COLORS["dim"]
    )
    console.print()
    console.print("[bold]Auto-Approve Mode:[/bold]", style=COLORS["primary"])
    console.print("  Ctrl+T          Toggle auto-approve mode", style=COLORS["dim"])
    console.print(
        "  --auto-approve  Start CLI with auto-approve enabled (via command line)",
        style=COLORS["dim"],
    )
    console.print(
        "  When enabled, tool actions execute without confirmation prompts", style=COLORS["dim"]
    )
    console.print()


def extract_dmail_timeline(state: dict) -> list[dict]:
    """Extract D-Mail events from SystemMessage entries in messages.

    Args:
        state: Agent state dictionary containing messages

    Returns:
        List of D-Mail event dictionaries with checkpoint IDs, messages, and attachments
    """
    timeline = []
    messages = state.get("messages", [])

    for msg in messages:
        if isinstance(msg, SystemMessage) and msg.name == "dmail" and msg.additional_kwargs.get("dmail"):
            timeline.append(
                {
                    "from_checkpoint_id": msg.additional_kwargs.get("from_checkpoint_id"),
                    "to_checkpoint_id": msg.additional_kwargs.get("to_checkpoint_id"),
                    "message": msg.content,
                    "attached_mem_paths": msg.additional_kwargs.get("attached_mem_paths", []),
                    "reason": msg.additional_kwargs.get("reason"),
                }
            )

    return timeline


def resolve_checkpoint_alias(checkpoint_id: str, checkpoints: list[dict]) -> str:
    """Resolve checkpoint ID to alias name, fallback to short ID.

    Args:
        checkpoint_id: The checkpoint ID to resolve
        checkpoints: List of checkpoint alias dictionaries

    Returns:
        Resolved alias name or shortened checkpoint ID
    """
    if not checkpoint_id:
        return "unknown"

    for cp in checkpoints:
        if cp.get("id") == checkpoint_id:
            return cp.get("name") or checkpoint_id[:8]

    return checkpoint_id[:8]


def estimate_abandoned_tokens(timeline: list[dict]) -> int:
    """Rough estimate of context saved by D-Mail rewinds.

    Uses simple character-based heuristic (4 chars ≈ 1 token).

    Args:
        timeline: List of D-Mail event dictionaries

    Returns:
        Estimated token count (approximate)
    """
    total_chars = sum(len(entry.get("message", "")) for entry in timeline)
    return total_chars // 4


def render_dmail_audit(checkpoints: list[dict], timeline: list[dict], enabled: bool = True) -> None:
    """Render D-Mail audit pane with checkpoints and timeline.

    Args:
        checkpoints: List of checkpoint alias dictionaries
        timeline: List of D-Mail event dictionaries
        enabled: Whether D-Mail is currently enabled
    """
    if not enabled:
        console.print()
        console.print("[bold]D-Mail is not enabled.[/bold]", style="yellow")
        console.print()
        console.print("To enable, run with:")
        console.print("  deepagents --enable-dmail", style="dim")
        console.print()
        console.print("Or add to ~/.deepagents/<agent>/config.toml:")
        console.print("  [dmail]", style="dim")
        console.print("  enabled = true", style="dim")
        console.print()
        return

    console.print()
    console.print(Panel("[bold]D-Mail Audit[/bold]", style=COLORS["primary"], box=box.ROUNDED))
    console.print()

    recent_checkpoints = checkpoints[-5:] if len(checkpoints) > 5 else checkpoints

    if recent_checkpoints:
        console.print("[bold]Checkpoint Aliases[/bold] (5 most recent):", style=COLORS["primary"])
        console.print()

        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        table.add_column("Status", style="bold", width=3)
        table.add_column("Name", style="cyan")
        table.add_column("Time", style="dim")
        table.add_column("Type", style="dim")

        for cp in reversed(recent_checkpoints):
            name = cp.get("name", "unnamed")
            reason = cp.get("reason", "")
            created_at = cp.get("created_at", "")

            # Defensive timestamp parsing: handle ISO format or fallback gracefully
            time_str = ""
            if created_at:
                if "T" in created_at:
                    parts = created_at.split("T")
                    if len(parts) > 1 and len(parts[1]) >= 5:
                        time_str = parts[1][:5]  # Extract HH:MM from ISO timestamp
                    else:
                        time_str = created_at[:8]  # Fallback to first 8 chars
                else:
                    time_str = created_at[:8] if len(created_at) >= 8 else created_at

            is_auto = reason and reason.startswith("auto:")
            status_icon = "[yellow]●[/yellow]" if is_auto else "[green]●[/green]"
            type_str = "auto" if is_auto else "manual"

            table.add_row(status_icon, name, time_str, type_str)

        console.print(table)
        console.print()
    else:
        console.print("[dim]No checkpoints yet[/dim]")
        console.print()

    if timeline:
        console.print(f"[bold]D-Mail Timeline[/bold] ({len(timeline)} rewinds):", style=COLORS["primary"])
        console.print()

        for entry in timeline:
            from_id = entry.get("from_checkpoint_id", "unknown")
            to_id = entry.get("to_checkpoint_id", "unknown")
            message = entry.get("message", "")
            attachments = entry.get("attached_mem_paths", [])

            from_alias = resolve_checkpoint_alias(from_id, checkpoints)
            to_alias = resolve_checkpoint_alias(to_id, checkpoints)

            console.print(f"🕰️  [cyan]{from_alias}[/cyan] → [cyan]{to_alias}[/cyan]")

            if message:
                preview = message[:100] + "..." if len(message) > 100 else message
                console.print(f"   {preview!r}", style="dim")

            if attachments:
                for path in attachments:
                    display_path = truncate_value(path, max_length=60)
                    console.print(f"   Attached: {display_path}", style="dim yellow")

            console.print()

        abandoned = estimate_abandoned_tokens(timeline)
        if abandoned > 0:
            console.print(f"[dim]Abandoned Context: ~{abandoned:,} tokens pruned (est.)[/dim]")
            console.print()
    else:
        console.print("[dim]No D-Mail rewinds yet[/dim]")
        console.print()


def show_help():
    """Show help information."""
    console.print()
    console.print(DEEP_AGENTS_ASCII, style=f"bold {COLORS['primary']}")
    console.print()

    console.print("[bold]Usage:[/bold]", style=COLORS["primary"])
    console.print(
        "  deepagents [--agent NAME] [--auto-approve] [--enable-dmail]  Start interactive session"
    )
    console.print("  deepagents list                                          List all available agents")
    console.print(
        "  deepagents reset --agent AGENT                               Reset agent to default prompt"
    )
    console.print(
        "  deepagents reset --agent AGENT --target SOURCE               Reset agent to copy of another agent"
    )
    console.print("  deepagents help                                          Show this help message")
    console.print()

    console.print("[bold]Examples:[/bold]", style=COLORS["primary"])
    console.print(
        "  deepagents                              # Start with default agent", style=COLORS["dim"]
    )
    console.print(
        "  deepagents --agent mybot                # Start with agent named 'mybot'",
        style=COLORS["dim"],
    )
    console.print(
        "  deepagents --auto-approve               # Start with auto-approve enabled",
        style=COLORS["dim"],
    )
    console.print(
        "  deepagents list                         # List all agents", style=COLORS["dim"]
    )
    console.print(
        "  deepagents reset --agent mybot          # Reset mybot to default", style=COLORS["dim"]
    )
    console.print(
        "  deepagents reset --agent mybot --target other # Reset mybot to copy of 'other' agent",
        style=COLORS["dim"],
    )
    console.print()

    console.print("[bold]D-Mail Options:[/bold]", style=COLORS["primary"])
    console.print("  --enable-dmail              Enable D-Mail temporal rollback")
    console.print(
        "  --dmail-auto-checkpoints    Enable auto-checkpoints (default: True if D-Mail enabled)"
    )
    console.print("  --dmail-max-auto N          Max auto-checkpoints per run (default: 3)")
    console.print()
    console.print("[bold]D-Mail Commands:[/bold]", style=COLORS["primary"])
    console.print("  /dmail                      Show D-Mail audit (checkpoints and rewind history)")
    console.print("  /dmail checkpoints          Show only checkpoint aliases")
    console.print("  /dmail log                  Show only D-Mail timeline")
    console.print("  /dmail config               Show current D-Mail configuration")
    console.print("  /dmail config auto on|off   Toggle auto-checkpoints (persists to config.toml)")
    console.print("  /dmail config max-auto N    Set max auto-checkpoints (persists to config.toml)")
    console.print()
    console.print("[dim]Note: Config changes require CLI restart to take effect[/dim]")
    console.print()

    console.print("[bold]Long-term Memory:[/bold]", style=COLORS["primary"])
    console.print(
        "  By default, long-term memory is ENABLED using agent name 'agent'.", style=COLORS["dim"]
    )
    console.print("  Memory includes:", style=COLORS["dim"])
    console.print("  - Persistent agent.md file with your instructions", style=COLORS["dim"])
    console.print("  - /memories/ folder for storing context across sessions", style=COLORS["dim"])
    console.print()

    console.print("[bold]Agent Storage:[/bold]", style=COLORS["primary"])
    console.print("  Agents are stored in: ~/.deepagents/AGENT_NAME/", style=COLORS["dim"])
    console.print("  Each agent has an agent.md file containing its prompt", style=COLORS["dim"])
    console.print()

    console.print("[bold]Interactive Features:[/bold]", style=COLORS["primary"])
    console.print("  Enter           Submit your message", style=COLORS["dim"])
    console.print(
        "  Alt+Enter       Insert newline for multi-line (Option+Enter or ESC then Enter)",
        style=COLORS["dim"],
    )
    console.print("  Ctrl+J          Insert newline (alternative)", style=COLORS["dim"])
    console.print("  Ctrl+T          Toggle auto-approve mode", style=COLORS["dim"])
    console.print("  Arrow keys      Navigate input", style=COLORS["dim"])
    console.print(
        "  @filename       Type @ to auto-complete files and inject content", style=COLORS["dim"]
    )
    console.print(
        "  /command        Type / to see available commands (auto-completes)", style=COLORS["dim"]
    )
    console.print()

    console.print("[bold]Interactive Commands:[/bold]", style=COLORS["primary"])
    console.print("  /help           Show available commands and features", style=COLORS["dim"])
    console.print("  /clear          Clear screen and reset conversation", style=COLORS["dim"])
    console.print("  /tokens         Show token usage for current session", style=COLORS["dim"])
    console.print("  /quit, /exit    Exit the session", style=COLORS["dim"])
    console.print(
        "  quit, exit, q   Exit the session (just type and press Enter)", style=COLORS["dim"]
    )
    console.print()

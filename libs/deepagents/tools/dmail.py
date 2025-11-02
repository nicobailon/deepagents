"""D-Mail tools for checkpoint management and temporal rollback."""

from __future__ import annotations

from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from langgraph.constants import END
from langgraph.types import Command, Overwrite

_PENDING_ALIASES_KEY = "__dmail_pending_aliases__"
_REQUEST_KEY = "__dmail_request__"
_CHECKPOINTS_KEY = "dmail_checkpoints"


def _copy_runtime_list(runtime: ToolRuntime | None, key: str) -> list[dict[str, Any]]:
    """Return a shallow copy of the list stored under ``key`` in runtime state."""
    if runtime is None or getattr(runtime, "state", None) is None:
        msg = "D-Mail tools require runtime context; ensure enable_dmail=True on create_deep_agent."
        raise ValueError(msg)

    existing = runtime.state.get(key)
    if isinstance(existing, Overwrite):
        existing = existing.value

    if isinstance(existing, list):
        return list(existing)

    return []


@tool
def mark_checkpoint(
    runtime: ToolRuntime,
    *,
    name: str | None = None,
    reason: str | None = None,
) -> Command:
    """Record a checkpoint alias request for D-Mail middleware to bind."""
    pending = _copy_runtime_list(runtime, _PENDING_ALIASES_KEY)
    pending.append({"name": name, "reason": reason})
    return Command(update={_PENDING_ALIASES_KEY: Overwrite(pending)})


@tool
def list_checkpoints(limit: int = 10, *, runtime: ToolRuntime) -> list[dict[str, Any]]:
    """List recent checkpoint aliases in reverse-chronological order."""
    checkpoints = _copy_runtime_list(runtime, _CHECKPOINTS_KEY)
    if limit < 0:
        limit = 0
    if limit:
        checkpoints = checkpoints[-limit:]
    checkpoints.reverse()
    return checkpoints


@tool
def send_dmail(
    checkpoint: str,
    message: str,
    *,
    attach_files: list[str] | None = None,
    persist_attachments: bool = False,
    runtime: ToolRuntime | None = None,  # noqa: ARG001
) -> Command:
    """Request a temporal rollback to ``checkpoint`` with a compressed message.

    Args:
        checkpoint: Checkpoint alias or ID to rewind to.
        message: Compact action-oriented note for your past self.
        attach_files: Optional list of file paths to preserve across the rewind.
        persist_attachments: If True, copy attach_files to /memories/dmail/ for persistence.
        runtime: Injected tool runtime (automatically provided).

    Returns:
        Command to end current run and trigger the D-Mail fork.
    """
    if not checkpoint:
        msg = "The checkpoint parameter must be provided."
        raise ValueError(msg)
    if not message:
        msg = "The message parameter must be provided."
        raise ValueError(msg)

    update: dict[str, Any] = {
        "checkpoint": checkpoint,
        "message": message,
        "attach_files": attach_files or [],
        "persist_attachments": persist_attachments,
    }
    return Command(update={_REQUEST_KEY: update}, goto=END)


__all__ = ["list_checkpoints", "mark_checkpoint", "send_dmail"]

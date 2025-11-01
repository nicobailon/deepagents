"""Runtime helpers for executing deepagents graphs."""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RunnableConfig

from deepagents.constants import DMAIL_RESUME_FLAG


def run_until_stable(
    graph: CompiledStateGraph,
    inputs: Any,
    *,
    config: RunnableConfig | None = None,
) -> Mapping[str, Any] | MutableMapping[str, Any]:
    """Invoke ``graph`` and auto-resume when D-Mail requests a rewind.

    The helper mirrors the driver loop described in the D-Mail design: it executes the
    graph once with ``inputs`` and, whenever the run signals that a rewind occurred,
    automatically calls ``graph.invoke(None, config=config)`` to continue execution from
    the forked checkpoint. Interrupts (HITL) are surfaced to the caller unchanged.

    Args:
        graph: Compiled LangGraph agent.
        inputs: Initial input for the invocation.
        config: Optional runnable configuration (thread id, checkpoint id, etc.).

    Returns:
        Final graph output after all required resumes have completed.
    """
    result: Mapping[str, Any] | MutableMapping[str, Any] = graph.invoke(inputs, config=config)
    while True:
        if _has_interrupt(result):
            break
        if not _needs_resume(result):
            break
        result = graph.invoke(None, config=config)

    if isinstance(result, MutableMapping) and DMAIL_RESUME_FLAG in result:
        result = result.copy()
        result.pop(DMAIL_RESUME_FLAG, None)
    elif isinstance(result, Mapping) and DMAIL_RESUME_FLAG in result:
        result = dict(result)
        result.pop(DMAIL_RESUME_FLAG, None)
    return result


def _needs_resume(result: Mapping[str, Any]) -> bool:
    return bool(result.get(DMAIL_RESUME_FLAG))


def _has_interrupt(result: Mapping[str, Any]) -> bool:
    return bool(result.get("__interrupt__"))


__all__ = ["run_until_stable"]

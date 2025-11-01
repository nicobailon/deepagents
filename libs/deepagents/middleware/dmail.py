"""Middleware enabling Temporal Rollback (\"D-Mail\") support."""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable, MutableMapping, NotRequired, Protocol, TypedDict, cast

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import AgentState, ModelRequest, ModelResponse
from langchain.tools import ToolRuntime
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Overwrite

from deepagents.constants import DMAIL_RESUME_FLAG
from deepagents.backends.protocol import BackendFactory, BackendProtocol

logger = logging.getLogger(__name__)

_PENDING_ALIASES_KEY = "__dmail_pending_aliases__"
_REQUEST_KEY = "__dmail_request__"
_CHECKPOINTS_KEY = "dmail_checkpoints"
_DEFAULT_ALIAS_PREFIX_LEN = 8
DMAIL_TOOL_NAMES = {"mark_checkpoint", "list_checkpoints", "send_dmail"}


class _GraphInterface(Protocol):
    def get_state(self, config: dict[str, Any] | None) -> Any:
        ...

    def update_state(self, config: dict[str, Any], values: dict[str, Any]) -> None:
        ...


class _RuntimeInterface(Protocol):
    graph: _GraphInterface
    config: dict[str, Any] | None


class DMailCheckpointDict(TypedDict, total=False):
    id: str
    name: str
    created_at: str
    reason: str | None


class DMailState(AgentState):
    dmail_checkpoints: NotRequired[list[DMailCheckpointDict]]
    __dmail_pending_aliases__: NotRequired[list[dict[str, Any]]]
    __dmail_request__: NotRequired[dict[str, Any] | Overwrite]
    _dmail_resume_required__: NotRequired[bool]


@dataclass
class DMailCheckpoint:
    """Representation of a named checkpoint in agent state."""

    id: str
    name: str
    created_at: str
    reason: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "reason": self.reason,
        }


@dataclass
class DMailThresholds:
    """Configuration for auto-checkpoint thresholds.

    Note: file_bytes_threshold and tool_result_token_threshold from the design
    are not yet implemented. Current heuristics use simpler triggers.
    """

    max_auto_per_run: int = 3
    before_subagent: bool = True
    before_code_iteration: bool = True


DMAIL_SYSTEM_PROMPT = """### Temporal rollback ("D-Mail")

You can set savepoints and rewind to them with a compressed note to your past self. Use the tools below when intentional timeline compression is helpful.

Tools:
* `mark_checkpoint(name?, reason?)` — set a savepoint before exploring.
* `list_checkpoints(limit?)` — list recent checkpoint aliases.
* `send_dmail(checkpoint, message, attach_files?, persist_attachments?)` — rewind to `checkpoint` and append a short message. Use `attach_files` with `persist_attachments=True` to preserve important files across the rewind (saved to /memories/dmail/).

Guidelines:
* Keep messages concise, action-oriented, and aimed at your past self.
* Prefer the format "WHAT WAS DONE / WHAT MATTERS / DO NEXT" inside the message.
* Use D-Mail only when you need to prune exploration or failed paths.
* Use `persist_attachments=True` when the past timeline needs access to specific artifacts (they'll be stored in /memories/dmail/ and survive the rewind).
"""


class DMailMiddleware(AgentMiddleware):
    """Middleware that binds aliases and applies D-Mail rollbacks."""

    name = "DMailMiddleware"
    state_schema = DMailState

    def __init__(
        self,
        *,
        max_checkpoints: int = 20,
        system_prompt: str | None = DMAIL_SYSTEM_PROMPT,
        auto_checkpoints: bool = False,
        thresholds: DMailThresholds | None = None,
        backend: BackendProtocol | BackendFactory | None = None,
    ) -> None:
        """Create the middleware.

        Args:
            max_checkpoints: Maximum number of checkpoints to retain in state.
            system_prompt: Optional system prompt instructions to append.
            auto_checkpoints: Enable automatic checkpoint creation before expensive operations.
            thresholds: Configuration for auto-checkpoint heuristics.
            backend: Optional backend for attachment persistence (if None, inferred from runtime).
        """
        super().__init__()
        self.max_checkpoints = max_checkpoints
        self.system_prompt = system_prompt
        self.auto_checkpoints = auto_checkpoints
        self.thresholds = thresholds if thresholds is not None else DMailThresholds()
        self.backend: BackendProtocol | BackendFactory | None = backend

    # --- Model prompt augmentation -------------------------------------------------
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        if self.system_prompt:
            request.system_prompt = request.system_prompt + "\n\n" + self.system_prompt if request.system_prompt else self.system_prompt
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        if self.system_prompt:
            request.system_prompt = request.system_prompt + "\n\n" + self.system_prompt if request.system_prompt else self.system_prompt
        return await handler(request)

    # --- Tool interception ---------------------------------------------------------
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        tool_name = request.tool_call.get("name")
        args = request.tool_call.get("args") or {}

        if not isinstance(tool_name, str):
            return handler(request)

        if self.auto_checkpoints and self._should_auto_checkpoint(tool_name, args, request.state):
            pending = self._coerce_entries(request.state.get(_PENDING_ALIASES_KEY))
            pending.append({"name": None, "reason": f"auto:{tool_name}"})
            request.state[_PENDING_ALIASES_KEY] = pending

        if tool_name == "send_dmail":
            error_message = self._validate_send_dmail_request(request)
            if error_message:
                return ToolMessage(error_message, tool_call_id=request.tool_call.get("id"), name=tool_name)

            if args.get("persist_attachments"):
                try:
                    attached_paths = self._persist_attachments(request.runtime, args.get("attach_files", []))
                    result = handler(request)
                    if hasattr(result, "update") and isinstance(result.update, dict) and _REQUEST_KEY in result.update:
                        result_update = result.update[_REQUEST_KEY]
                        if isinstance(result_update, dict):
                            result_update["attached_mem_paths"] = attached_paths
                    return result
                except ValueError as exc:
                    return ToolMessage(str(exc), tool_call_id=request.tool_call.get("id"), name=tool_name)

        if tool_name in DMAIL_TOOL_NAMES:
            try:
                return handler(request)
            except ValueError as exc:
                return ToolMessage(str(exc), tool_call_id=request.tool_call.get("id"), name=tool_name)
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Any]],
    ) -> Any:
        tool_name = request.tool_call.get("name")
        args = request.tool_call.get("args") or {}

        if not isinstance(tool_name, str):
            return await handler(request)

        if self.auto_checkpoints and self._should_auto_checkpoint(tool_name, args, request.state):
            pending = self._coerce_entries(request.state.get(_PENDING_ALIASES_KEY))
            pending.append({"name": None, "reason": f"auto:{tool_name}"})
            request.state[_PENDING_ALIASES_KEY] = pending

        if tool_name == "send_dmail":
            error_message = self._validate_send_dmail_request(request)
            if error_message:
                return ToolMessage(error_message, tool_call_id=request.tool_call.get("id"), name=tool_name)

            if args.get("persist_attachments"):
                try:
                    attached_paths = self._persist_attachments(request.runtime, args.get("attach_files", []))
                    result = await handler(request)
                    if hasattr(result, "update") and isinstance(result.update, dict) and _REQUEST_KEY in result.update:
                        result_update = result.update[_REQUEST_KEY]
                        if isinstance(result_update, dict):
                            result_update["attached_mem_paths"] = attached_paths
                    return result
                except ValueError as exc:
                    return ToolMessage(str(exc), tool_call_id=request.tool_call.get("id"), name=tool_name)

        if tool_name in DMAIL_TOOL_NAMES:
            try:
                return await handler(request)
            except ValueError as exc:
                return ToolMessage(str(exc), tool_call_id=request.tool_call.get("id"), name=tool_name)
        return await handler(request)

    # --- Post-agent processing -----------------------------------------------------
    def after_agent(self, state: AgentState, runtime: Runtime[Any]) -> None:
        typed_state = cast(DMailState, state)
        runtime_like = cast(_RuntimeInterface, runtime)
        self._bind_pending_aliases(typed_state, runtime_like)
        self._apply_pending_request(typed_state, runtime_like)

    async def aafter_agent(self, state: AgentState, runtime: Runtime[Any]) -> None:
        typed_state = cast(DMailState, state)
        runtime_like = cast(_RuntimeInterface, runtime)
        self._bind_pending_aliases(typed_state, runtime_like)
        self._apply_pending_request(typed_state, runtime_like)

    # --- Helpers -------------------------------------------------------------------
    def _bind_pending_aliases(self, state: DMailState, runtime: _RuntimeInterface) -> None:
        state_dict = cast(dict[str, Any], state)
        pending = state.get(_PENDING_ALIASES_KEY)
        if not pending:
            return

        _, checkpoint_id = self._current_snapshot(runtime)
        checkpoints = self._load_checkpoints(state)
        existing_names = {cp.name for cp in checkpoints}

        entries = self._coerce_entries(pending)
        for entry in entries:
            name = entry.get("name")
            reason = entry.get("reason")
            alias = self._ensure_unique_alias(name, checkpoint_id, existing_names)
            checkpoints.append(
                DMailCheckpoint(
                    id=checkpoint_id,
                    name=alias,
                    created_at=datetime.now(UTC).isoformat(),
                    reason=reason,
                )
            )
            existing_names.add(alias)

        # Enforce retention policy
        if self.max_checkpoints > 0 and len(checkpoints) > self.max_checkpoints:
            checkpoints = checkpoints[-self.max_checkpoints :]

        state_dict[_CHECKPOINTS_KEY] = [cp.to_dict() for cp in checkpoints]
        state_dict.pop(_PENDING_ALIASES_KEY, None)

    def _apply_pending_request(self, state: DMailState, runtime: _RuntimeInterface) -> None:
        state_dict = cast(dict[str, Any], state)
        payload = state.get(_REQUEST_KEY)
        if not payload:
            return

        if isinstance(payload, Overwrite):
            payload = payload.value

        if not isinstance(payload, dict):
            state_dict.pop(_REQUEST_KEY, None)
            return

        target = payload.get("checkpoint")
        if not isinstance(target, str):
            msg = "send_dmail requires a checkpoint alias or id."
            raise ValueError(msg)

        checkpoints = self._load_checkpoints(state)
        alias_map = {cp.name: cp.id for cp in checkpoints}
        alias_entry = next((cp for cp in checkpoints if cp.name == target or cp.id == target), None)
        checkpoint_id = alias_map.get(target, target)
        if not isinstance(checkpoint_id, str):
            msg = f"Unknown checkpoint identifier: {target}"
            raise ValueError(msg)

        _, current_id = self._current_snapshot(runtime)
        if current_id == checkpoint_id:
            state_dict.pop(_REQUEST_KEY, None)
            return

        message = payload.get("message", "")
        if not message:
            msg = "send_dmail requires a non-empty message."
            raise ValueError(msg)

        fork_config = self._fork_config(runtime.config, checkpoint_id)
        attached_raw = payload.get("attached_mem_paths")
        attached_paths: list[str] = []
        if isinstance(attached_raw, list):
            attached_paths = [str(path) for path in attached_raw]
        content_parts = [f"D-MAIL from future:\n{message}"]
        if attached_paths:
            content_parts.append(f"\n\nAttached files: {', '.join(attached_paths)}")

        dmail_message = SystemMessage(
            content="".join(content_parts),
            name="dmail",
            additional_kwargs={
                "dmail": True,
                "from_checkpoint_id": current_id,
                "to_checkpoint_id": checkpoint_id,
            },
        )
        if alias_entry and alias_entry.reason:
            dmail_message.additional_kwargs["reason"] = alias_entry.reason
        if attached_paths:
            dmail_message.additional_kwargs["attached_mem_paths"] = attached_paths

        runtime.graph.update_state(
            config=fork_config,
            values={
                "messages": [dmail_message],
            },
        )
        state_dict[DMAIL_RESUME_FLAG] = True
        state_dict.pop(_REQUEST_KEY, None)

    def _validate_send_dmail_request(self, request: ToolCallRequest) -> str | None:
        args = request.tool_call.get("args") or {}
        checkpoint = args.get("checkpoint")
        if not isinstance(checkpoint, str) or not checkpoint.strip():
            return "send_dmail requires a checkpoint alias. Use mark_checkpoint() first."

        message = args.get("message")
        if not isinstance(message, str) or not message.strip():
            return "send_dmail requires a non-empty message."

        checkpoints = self._load_checkpoints(request.state)
        if not checkpoints:
            return None
        alias_names = {cp.name for cp in checkpoints}
        alias_ids = {cp.id for cp in checkpoints}
        if checkpoint in alias_names or checkpoint in alias_ids:
            return None
        # Allow raw checkpoint identifiers even if they have no alias recorded.
        return None

    def _load_checkpoints(self, state: AgentState) -> list[DMailCheckpoint]:
        raw = state.get(_CHECKPOINTS_KEY)
        if not raw:
            return []

        if isinstance(raw, Overwrite):
            raw = raw.value

        if not isinstance(raw, list):
            return []

        checkpoints: list[DMailCheckpoint] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            checkpoint_id = item.get("id")
            name = item.get("name")
            created_at = item.get("created_at")
            if not isinstance(checkpoint_id, str) or not isinstance(name, str) or not isinstance(created_at, str):
                continue
            reason_value = item.get("reason")
            checkpoints.append(
                DMailCheckpoint(
                    id=checkpoint_id,
                    name=name,
                    created_at=created_at,
                    reason=reason_value if isinstance(reason_value, str) else None,
                )
            )
        return checkpoints

    def _ensure_unique_alias(self, requested_name: str | None, checkpoint_id: str, existing: set[str]) -> str:
        base = (requested_name or checkpoint_id[:_DEFAULT_ALIAS_PREFIX_LEN]).strip() or checkpoint_id[:_DEFAULT_ALIAS_PREFIX_LEN]
        candidate = base
        idx = 2
        while candidate in existing:
            candidate = f"{base}-{idx}"
            idx += 1
        return candidate

    def _current_snapshot(self, runtime: _RuntimeInterface) -> tuple[Any, str]:
        try:
            snapshot = runtime.graph.get_state(runtime.config)
        except Exception as exc:  # pragma: no cover - defensive
            msg = "D-Mail requires a configured LangGraph checkpointer."
            raise RuntimeError(msg) from exc

        checkpoint_id = getattr(snapshot, "checkpoint_id", None)
        if not isinstance(checkpoint_id, str):
            config = getattr(snapshot, "config", {})
            if isinstance(config, dict):
                checkpoint_id = config.get("configurable", {}).get("checkpoint_id")
        if not isinstance(checkpoint_id, str):
            msg = "D-Mail middleware could not determine the current checkpoint id."
            raise RuntimeError(msg)

        return snapshot, checkpoint_id

    def _fork_config(self, config: dict[str, Any] | None, checkpoint_id: str) -> dict[str, Any]:
        base = deepcopy(config) if isinstance(config, dict) else {}
        configurable = dict(base.get("configurable", {}))
        thread_id = configurable.get("thread_id")
        if not isinstance(thread_id, str):
            msg = (
                "D-Mail requires a thread_id in the LangGraph config. "
                "Provide a Checkpointer when creating the agent and set "
                "RunnableConfig.configurable['thread_id'] when invoking."
            )
            raise RuntimeError(msg)
        configurable["checkpoint_id"] = checkpoint_id
        base["configurable"] = configurable
        return base

    @staticmethod
    def _coerce_entries(value: Any) -> list[dict[str, Any]]:
        if isinstance(value, Overwrite):
            candidate = value.value
        else:
            candidate = value

        if isinstance(candidate, list):
            return [entry for entry in candidate if isinstance(entry, dict)]
        return []

    def _should_auto_checkpoint(self, tool_name: str, args: dict[str, Any], state: AgentState) -> bool:
        auto_count = sum(
            1
            for cp in self._load_checkpoints(state)
            if isinstance(cp.reason, str) and cp.reason.startswith("auto:")
        )
        if auto_count >= self.thresholds.max_auto_per_run:
            return False

        if tool_name == "read_file":
            limit = args.get("limit")
            if limit is None:
                return True

        if tool_name == "task" and self.thresholds.before_subagent:
            return True

        if tool_name == "edit_file" and self.thresholds.before_code_iteration:
            return True

        return False

    def _persist_attachments(self, runtime: ToolRuntime, paths: list[str]) -> list[str]:
        if not paths:
            return []

        ts = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        dest_prefix = f"/memories/dmail/{ts}/"
        attached: list[str] = []

        from deepagents.backends.composite import CompositeBackend
        from deepagents.backends.state import StateBackend
        from deepagents.backends.store import StoreBackend

        backend_override = self.backend
        if backend_override is not None:
            resolved_backend: BackendProtocol = (
                backend_override(runtime) if callable(backend_override) else backend_override
            )
        else:
            composite = CompositeBackend(
                default=StateBackend(runtime),
                routes={"/memories/": cast(BackendProtocol, StoreBackend(runtime))},
            )
            resolved_backend = cast(BackendProtocol, composite)

        for p in paths:
            if p.startswith("/memories/"):
                attached.append(p)
                continue

            if ".." in p:
                continue

            state_mapping = cast(MutableMapping[str, Any], getattr(runtime, "state", {}))
            files = state_mapping.get("files", {})
            if isinstance(files, Overwrite):
                files = files.value
            if not isinstance(files, dict):
                continue

            fd = files.get(p)
            if isinstance(fd, dict) and "content" in fd:
                content_lines = fd["content"]
                if isinstance(content_lines, list):
                    content = "\n".join(content_lines)
                else:
                    content = str(content_lines)

                safe_path = p.lstrip("/").replace("..", "")
                dest_path = dest_prefix + safe_path
                try:
                    resolved_backend.write(dest_path, content)
                    attached.append(dest_path)
                except Exception as exc:  # pragma: no cover - defensive logging only
                    logger.warning(
                        "Failed to persist D-Mail attachment from %s to %s: %s",
                        p,
                        dest_path,
                        exc,
                    )

        return attached


__all__ = ["DMailMiddleware", "DMAIL_SYSTEM_PROMPT", "DMailThresholds"]

"""Middleware enabling Temporal Rollback (\"D-Mail\") support."""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable, MutableMapping, NotRequired, Protocol, Sequence, TypedDict, cast

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import AgentState, ModelRequest, ModelResponse
from langchain.tools import ToolRuntime
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Overwrite

from deepagents.constants import DMAIL_RESUME_FLAG
from deepagents.backends.protocol import BackendFactory, BackendProtocol

logger = logging.getLogger(__name__)

_PENDING_ALIASES_KEY = "__dmail_pending_aliases__"
_REQUEST_KEY = "__dmail_request__"
_CHECKPOINTS_KEY = "dmail_checkpoints"
_COUNTERS_KEY = "dmail_counters"
_COUNTER_BEFORE = "before"
_COUNTER_AFTER = "after"
_RUN_EPOCH = "run_epoch"
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
    """Configuration for auto-checkpoint thresholds."""

    before_first_user_message: bool = True
    before_every_tool: bool = True
    max_auto_before_per_run: int = 15
    after_every_tool: bool = True
    after_agent_response: bool = True
    max_auto_after_per_run: int = 20

    def __post_init__(self) -> None:
        if self.max_auto_before_per_run < 0:
            self.max_auto_before_per_run = 0
        if self.max_auto_after_per_run < 0:
            self.max_auto_after_per_run = 0


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
        max_checkpoints: int = 64,
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

    # --- Run lifecycle hooks ------------------------------------------------------
    def before_agent(self, state: AgentState, runtime: Runtime[Any]) -> None:
        typed_state = cast(DMailState, state)
        runtime_like = cast(_RuntimeInterface, runtime)
        self._before_agent_impl(typed_state, runtime_like)

    async def abefore_agent(self, state: AgentState, runtime: Runtime[Any]) -> None:
        typed_state = cast(DMailState, state)
        runtime_like = cast(_RuntimeInterface, runtime)
        self._before_agent_impl(typed_state, runtime_like)

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

        typed_state = cast(DMailState, request.state)
        runtime_for_auto: _RuntimeInterface | None = None
        auto_enabled = self.auto_checkpoints and tool_name not in DMAIL_TOOL_NAMES

        if auto_enabled:
            self._sync_counters(typed_state)
            if self.thresholds.before_every_tool:
                runtime_for_auto = self._resolve_runtime(request.runtime)
                self._maybe_bind_auto_checkpoint(
                    typed_state,
                    runtime_for_auto,
                    name=None,
                    reason=f"auto:before_{tool_name}",
                    counter_key=_COUNTER_BEFORE,
                    limit=self.thresholds.max_auto_before_per_run,
                )

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
        try:
            result = handler(request)
        except Exception:
            if auto_enabled and self.thresholds.after_every_tool:
                runtime_for_auto = runtime_for_auto or self._resolve_runtime(request.runtime)
                self._maybe_bind_auto_checkpoint(
                    typed_state,
                    runtime_for_auto,
                    name=None,
                    reason=f"auto:after_{tool_name}_error",
                    counter_key=_COUNTER_AFTER,
                    limit=self.thresholds.max_auto_after_per_run,
                )
            raise

        if auto_enabled and self.thresholds.after_every_tool:
            runtime_for_auto = runtime_for_auto or self._resolve_runtime(request.runtime)
            status = "error" if self._is_error_result(result) else "success"
            self._maybe_bind_auto_checkpoint(
                typed_state,
                runtime_for_auto,
                name=None,
                reason=f"auto:after_{tool_name}_{status}",
                counter_key=_COUNTER_AFTER,
                limit=self.thresholds.max_auto_after_per_run,
            )
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Any]],
    ) -> Any:
        tool_name = request.tool_call.get("name")
        args = request.tool_call.get("args") or {}

        if not isinstance(tool_name, str):
            return await handler(request)

        typed_state = cast(DMailState, request.state)
        runtime_for_auto: _RuntimeInterface | None = None
        auto_enabled = self.auto_checkpoints and tool_name not in DMAIL_TOOL_NAMES

        if auto_enabled:
            self._sync_counters(typed_state)
            if self.thresholds.before_every_tool:
                runtime_for_auto = self._resolve_runtime(request.runtime)
                self._maybe_bind_auto_checkpoint(
                    typed_state,
                    runtime_for_auto,
                    name=None,
                    reason=f"auto:before_{tool_name}",
                    counter_key=_COUNTER_BEFORE,
                    limit=self.thresholds.max_auto_before_per_run,
                )

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
        try:
            result = await handler(request)
        except Exception:
            if auto_enabled and self.thresholds.after_every_tool:
                runtime_for_auto = runtime_for_auto or self._resolve_runtime(request.runtime)
                self._maybe_bind_auto_checkpoint(
                    typed_state,
                    runtime_for_auto,
                    name=None,
                    reason=f"auto:after_{tool_name}_error",
                    counter_key=_COUNTER_AFTER,
                    limit=self.thresholds.max_auto_after_per_run,
                )
            raise

        if auto_enabled and self.thresholds.after_every_tool:
            runtime_for_auto = runtime_for_auto or self._resolve_runtime(request.runtime)
            status = "error" if self._is_error_result(result) else "success"
            self._maybe_bind_auto_checkpoint(
                typed_state,
                runtime_for_auto,
                name=None,
                reason=f"auto:after_{tool_name}_{status}",
                counter_key=_COUNTER_AFTER,
                limit=self.thresholds.max_auto_after_per_run,
            )
        return result

    # --- Post-agent processing -----------------------------------------------------
    def after_agent(self, state: AgentState, runtime: Runtime[Any]) -> None:
        typed_state = cast(DMailState, state)
        runtime_like = cast(_RuntimeInterface, runtime)
        self._after_agent_impl(typed_state, runtime_like)

    async def aafter_agent(self, state: AgentState, runtime: Runtime[Any]) -> None:
        typed_state = cast(DMailState, state)
        runtime_like = cast(_RuntimeInterface, runtime)
        self._after_agent_impl(typed_state, runtime_like)

    # --- Helpers -------------------------------------------------------------------
    def _before_agent_impl(self, state: DMailState, runtime: _RuntimeInterface) -> None:
        self._sync_counters(state)
        if not (self.auto_checkpoints and self.thresholds.before_first_user_message):
            return

        if self._load_checkpoints(state):
            return

        messages = self._coerce_messages(state.get("messages"))
        if not messages:
            return

        user_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
        ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]

        if len(user_messages) != 1 or ai_messages:
            return

        self._maybe_bind_auto_checkpoint(
            state,
            runtime,
            name="task-start",
            reason="auto:before_first_message",
            counter_key=_COUNTER_BEFORE,
            limit=self.thresholds.max_auto_before_per_run,
        )

    def _after_agent_impl(self, state: DMailState, runtime: _RuntimeInterface) -> None:
        if self.auto_checkpoints and self.thresholds.after_agent_response:
            messages = self._coerce_messages(state.get("messages"))
            if self._is_user_ai_boundary(messages):
                self._maybe_bind_auto_checkpoint(
                    state,
                    runtime,
                    name=None,
                    reason="auto:after_response",
                    counter_key=_COUNTER_AFTER,
                    limit=self.thresholds.max_auto_after_per_run,
                )

        self._bind_pending_aliases(state, runtime)
        self._apply_pending_request(state, runtime)

    def _resolve_runtime(self, runtime: Any) -> _RuntimeInterface:
        if hasattr(runtime, "graph") and hasattr(runtime, "config"):
            return cast(_RuntimeInterface, runtime)

        context = getattr(runtime, "context", None)
        if context is not None and hasattr(context, "graph") and hasattr(context, "config"):
            return cast(_RuntimeInterface, context)

        msg = "D-Mail auto-checkpoints require runtime context; ensure enable_dmail=True."
        raise RuntimeError(msg)

    def _maybe_bind_auto_checkpoint(
        self,
        state: DMailState,
        runtime: _RuntimeInterface,
        *,
        name: str | None,
        reason: str,
        counter_key: str,
        limit: int,
    ) -> bool:
        if not self._counter_allows(state, counter_key, limit):
            return False

        self._append_alias(state, runtime, name=name, reason=reason)
        self._record_counter_increment(state, counter_key)
        return True

    def _counter_allows(self, state: DMailState, counter_key: str, limit: int) -> bool:
        counters = self._sync_counters(state)
        if limit == 0:
            return False
        if limit > 0 and counters[counter_key] >= limit:
            return False
        return True

    def _record_counter_increment(self, state: DMailState, counter_key: str) -> None:
        counters = self._get_counters(state)
        counters[counter_key] += 1
        cast(dict[str, Any], state)[_COUNTERS_KEY] = counters

    def _sync_counters(self, state: DMailState) -> dict[str, int]:
        counters = self._get_counters(state)
        epoch = self._current_run_epoch(state)
        if counters[_RUN_EPOCH] != epoch:
            counters = self._reset_counters(state, epoch)
        return counters

    def _get_counters(self, state: DMailState) -> dict[str, int]:
        raw = state.get(_COUNTERS_KEY)
        if isinstance(raw, Overwrite):
            raw = raw.value

        if isinstance(raw, dict):
            counters = {
                _COUNTER_BEFORE: self._coerce_int(raw.get(_COUNTER_BEFORE), default=0),
                _COUNTER_AFTER: self._coerce_int(raw.get(_COUNTER_AFTER), default=0),
                _RUN_EPOCH: self._coerce_int(raw.get(_RUN_EPOCH), default=-1),
            }
        else:
            counters = {_COUNTER_BEFORE: 0, _COUNTER_AFTER: 0, _RUN_EPOCH: -1}

        cast(dict[str, Any], state)[_COUNTERS_KEY] = counters
        return counters

    def _reset_counters(self, state: DMailState, epoch: int) -> dict[str, int]:
        counters = {_COUNTER_BEFORE: 0, _COUNTER_AFTER: 0, _RUN_EPOCH: epoch}
        cast(dict[str, Any], state)[_COUNTERS_KEY] = counters
        return counters

    @staticmethod
    def _coerce_int(value: Any, default: int = 0) -> int:
        try:
            coerced = int(value)
        except (TypeError, ValueError):
            return default
        return coerced

    def _current_run_epoch(self, state: DMailState) -> int:
        messages = self._coerce_messages(state.get("messages"))
        return sum(1 for msg in messages if isinstance(msg, HumanMessage))

    def _coerce_messages(self, value: Any) -> list[BaseMessage]:
        if isinstance(value, Overwrite):
            value = value.value
        if not isinstance(value, list):
            return []
        return [msg for msg in value if isinstance(msg, BaseMessage)]

    def _is_user_ai_boundary(self, messages: Sequence[BaseMessage]) -> bool:
        filtered = [msg for msg in messages if isinstance(msg, (HumanMessage, AIMessage))]
        if len(filtered) < 2:
            return False
        return isinstance(filtered[-1], AIMessage) and isinstance(filtered[-2], HumanMessage)

    def _append_alias(
        self,
        state: DMailState,
        runtime: _RuntimeInterface,
        *,
        name: str | None,
        reason: str | None,
    ) -> str:
        state_dict = cast(dict[str, Any], state)
        _, checkpoint_id = self._current_snapshot(runtime)
        checkpoints = self._load_checkpoints(state)
        existing_names = {cp.name for cp in checkpoints}
        alias = self._ensure_unique_alias(name, checkpoint_id, existing_names)
        checkpoints.append(
            DMailCheckpoint(
                id=checkpoint_id,
                name=alias,
                created_at=datetime.now(UTC).isoformat(),
                reason=reason if isinstance(reason, str) else None,
            )
        )

        if self.max_checkpoints > 0 and len(checkpoints) > self.max_checkpoints:
            checkpoints = checkpoints[-self.max_checkpoints :]

        state_dict[_CHECKPOINTS_KEY] = [cp.to_dict() for cp in checkpoints]
        return alias

    def _is_error_result(self, result: Any) -> bool:
        if isinstance(result, ToolMessage):
            content = result.content
            if isinstance(content, str):
                lowered = content.strip().lower()
                return lowered.startswith("error:") or "failed" in lowered
            if isinstance(content, list):
                text_parts = [part.get("text") for part in content if isinstance(part, dict)]
                joined = " ".join(part for part in text_parts if isinstance(part, str))
                lowered = joined.strip().lower()
                if lowered:
                    return lowered.startswith("error:") or "failed" in lowered
        elif isinstance(result, str):
            lowered = result.strip().lower()
            return lowered.startswith("error:") or "failed" in lowered
        elif isinstance(result, Exception):  # pragma: no cover - defensive
            return True
        return False

    def _bind_pending_aliases(self, state: DMailState, runtime: _RuntimeInterface) -> None:
        pending = state.get(_PENDING_ALIASES_KEY)
        entries = self._coerce_entries(pending)
        if not entries:
            return

        for entry in entries:
            raw_name = entry.get("name")
            name = raw_name if isinstance(raw_name, str) and raw_name.strip() else None
            raw_reason = entry.get("reason")
            reason = raw_reason if isinstance(raw_reason, str) else None
            self._append_alias(state, runtime, name=name, reason=reason)

        cast(dict[str, Any], state).pop(_PENDING_ALIASES_KEY, None)

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

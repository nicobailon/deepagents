from collections import deque
from typing import Any
from unittest.mock import MagicMock, patch

from langchain.tools import ToolRuntime
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.constants import END
from langgraph.types import Overwrite

from deepagents import DMailMiddleware, create_deep_agent, run_until_stable
from deepagents.constants import DMAIL_RESUME_FLAG
from deepagents.middleware.dmail import DMAIL_SYSTEM_PROMPT, DMailThresholds
from deepagents.middleware.subagents import SubAgentMiddleware
from deepagents.tools.dmail import list_checkpoints, mark_checkpoint, send_dmail
from langchain.agents.middleware.summarization import SummarizationMiddleware

CHECKPOINTS_KEY = "dmail_checkpoints"


class DummySnapshot:
    def __init__(self, checkpoint_id: str):
        self.config = {"configurable": {"checkpoint_id": checkpoint_id}}


class DummyGraph:
    def __init__(self, checkpoint_id: str):
        self.snapshot = DummySnapshot(checkpoint_id)
        self.updated: deque[tuple[dict[str, Any], dict[str, Any]]] = deque()
        self.last_get_state_config: dict[str, Any] | None = None

    def get_state(self, config: dict[str, Any] | None) -> DummySnapshot:
        self.last_get_state_config = config
        return self.snapshot

    def update_state(self, config: dict[str, Any], values: dict[str, Any]) -> None:
        self.updated.append((config, values))


class DummyRuntime:
    def __init__(self, checkpoint_id: str, thread_id: str):
        self.graph = DummyGraph(checkpoint_id=checkpoint_id)
        self.config = {"configurable": {"thread_id": thread_id}}


class FakeGraph:
    def __init__(self, outputs: list[dict[str, Any]]):
        self.outputs = deque(outputs)
        self.invocations: list[Any] = []

    def invoke(self, inputs: Any, config: dict[str, Any] | None = None) -> dict[str, Any]:
        self.invocations.append(inputs)
        return self.outputs.popleft()


class StubChatModel(BaseChatModel):
    def __init__(self, response: str = "ok") -> None:
        super().__init__()
        self._response = response

    @property
    def _llm_type(self) -> str:
        return "stub-chat-model"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ARG002
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self._response))])


def _tool_runtime(state: dict[str, Any], context: Any | None = None) -> ToolRuntime:
    return ToolRuntime(state=state, context=context, tool_call_id="", store=None, stream_writer=lambda _: None, config={})


class TestDMailTools:
    def test_mark_checkpoint_appends_pending_alias(self):
        state: dict[str, Any] = {CHECKPOINTS_KEY: []}
        runtime = _tool_runtime(state)
        command = mark_checkpoint.invoke({"runtime": runtime, "name": "alpha", "reason": "test"})
        assert isinstance(command.update, dict)
        pending = command.update["__dmail_pending_aliases__"]
        assert isinstance(pending, Overwrite)
        assert pending.value == [{"name": "alpha", "reason": "test"}]

    def test_list_checkpoints_returns_latest_first(self):
        checkpoints = [{"name": "a", "id": "1", "created_at": "t1"}, {"name": "b", "id": "2", "created_at": "t2"}]
        state = {"dmail_checkpoints": checkpoints}
        runtime = _tool_runtime(state)
        result = list_checkpoints.invoke({"limit": 1, "runtime": runtime})
        assert result == [checkpoints[-1]]

    def test_send_dmail_returns_command(self):
        command = send_dmail.invoke({"checkpoint": "alpha", "message": "Important note"})
        assert command.goto == END
        req = command.update["__dmail_request__"]
        assert req["checkpoint"] == "alpha"
        assert req["message"] == "Important note"
        assert req["attach_files"] == []
        assert req["persist_attachments"] is False


class TestDMailMiddleware:
    def test_binds_pending_aliases_and_enforces_max(self):
        state: dict[str, Any] = {
            "__dmail_pending_aliases__": [{"name": None, "reason": "first"}, {"name": "alpha", "reason": "explicit"}],
            "dmail_checkpoints": [{"id": "old", "name": "old", "created_at": "old-ts"}],
        }
        runtime = DummyRuntime(checkpoint_id="chk-001", thread_id="thread-1")
        middleware = DMailMiddleware(max_checkpoints=2)

        middleware.after_agent(state, runtime)

        checkpoints = state["dmail_checkpoints"]
        assert len(checkpoints) == 2  # trimmed to max_checkpoints
        assert all(isinstance(cp["name"], str) for cp in checkpoints)
        assert all(cp["id"] == "chk-001" for cp in checkpoints)
        assert "__dmail_pending_aliases__" not in state

    def test_apply_pending_request_updates_graph_and_sets_flag(self):
        state: dict[str, Any] = {
            "dmail_checkpoints": [{"id": "chk-000", "name": "alpha", "created_at": "now", "reason": "exploration"}],
            "__dmail_request__": {"checkpoint": "alpha", "message": "Focus on test coverage."},
        }
        runtime = DummyRuntime(checkpoint_id="future-123", thread_id="thread-1")
        middleware = DMailMiddleware()

        middleware.after_agent(state, runtime)

        config, values = runtime.graph.updated.pop()
        assert config["configurable"]["thread_id"] == "thread-1"
        assert config["configurable"]["checkpoint_id"] == "chk-000"
        dmail_message = values["messages"][0]
        assert isinstance(dmail_message, SystemMessage)
        assert "D-MAIL from future" in dmail_message.content
        assert dmail_message.additional_kwargs["from_checkpoint_id"] == "future-123"
        assert dmail_message.additional_kwargs["to_checkpoint_id"] == "chk-000"
        assert dmail_message.additional_kwargs["reason"] == "exploration"
        assert state[DMAIL_RESUME_FLAG] is True
        assert "__dmail_request__" not in state

    def test_system_prompt_is_appended(self):
        middleware = DMailMiddleware()
        request = type("Req", (), {"system_prompt": None})

        class Handler:
            def __call__(self, model_request):
                return model_request

        result = middleware.wrap_model_call(request, Handler())
        assert result.system_prompt == DMAIL_SYSTEM_PROMPT

    def test_send_dmail_allows_raw_checkpoint_id(self):
        state: dict[str, Any] = {CHECKPOINTS_KEY: [{"id": "chk-123", "name": "alpha", "created_at": "now"}]}
        runtime = _tool_runtime(state)
        request = ToolCallRequest(
            tool_call={"name": "send_dmail", "args": {"checkpoint": "chk-999", "message": "note"}, "id": "call-1"},
            tool=send_dmail,
            state=state,
            runtime=runtime,
        )
        middleware = DMailMiddleware()
        handler_called: list[bool] = []

        def handler(req):
            handler_called.append(True)
            return send_dmail.invoke(req.tool_call["args"])

        result = middleware.wrap_tool_call(request, handler)
        assert handler_called == [True]
        assert hasattr(result, "goto") and result.goto == END
        assert DMAIL_RESUME_FLAG not in state

    def test_send_dmail_empty_message_returns_tool_message(self):
        state: dict[str, Any] = {CHECKPOINTS_KEY: [{"id": "chk-123", "name": "alpha", "created_at": "now"}]}
        runtime = _tool_runtime(state)
        request = ToolCallRequest(
            tool_call={"name": "send_dmail", "args": {"checkpoint": "alpha", "message": ""}, "id": "call-1"},
            tool=send_dmail,
            state=state,
            runtime=runtime,
        )
        middleware = DMailMiddleware()
        handler_called: list[bool] = []

        def handler(req):  # pragma: no cover - should not be called
            handler_called.append(True)
            return send_dmail.invoke(req.tool_call["args"])

        result = middleware.wrap_tool_call(request, handler)
        assert isinstance(result, ToolMessage)
        assert "non-empty message" in result.content
        assert not handler_called
        assert DMAIL_RESUME_FLAG not in state


class TestCreateDeepAgentIntegration:
    def test_enable_dmail_adds_tools(self):
        model = StubChatModel("ok")
        agent = create_deep_agent(
            model=model,
            tools=[],
            enable_dmail=True,
        )
        tool_names = agent.nodes["tools"].bound._tools_by_name.keys()
        assert {"mark_checkpoint", "list_checkpoints", "send_dmail"}.issubset(tool_names)

    def test_dmail_middleware_ordering(self):
        model = StubChatModel("ok")
        fake_agent = MagicMock()
        fake_agent.with_config.return_value = fake_agent

        with patch("deepagents.graph.create_agent", return_value=fake_agent) as mock_create_agent:
            create_deep_agent(model=model, tools=[], enable_dmail=True)

        passed_middleware = mock_create_agent.call_args.kwargs["middleware"]
        dmail_indices = [idx for idx, mw in enumerate(passed_middleware) if isinstance(mw, DMailMiddleware)]
        summarizer_indices = [idx for idx, mw in enumerate(passed_middleware) if isinstance(mw, SummarizationMiddleware)]
        assert dmail_indices, "D-Mail middleware missing from top-level stack"
        assert summarizer_indices, "Summarization middleware missing"
        assert max(dmail_indices) < min(summarizer_indices)

    def test_subagent_configuration_includes_dmail(self):
        model = StubChatModel("ok")
        fake_agent = MagicMock()
        fake_agent.with_config.return_value = fake_agent

        original_init = SubAgentMiddleware.__init__
        recorded: dict[str, list[Any] | None] = {}

        def tracking_init(self, *args, **kwargs):  # type: ignore[override]
            recorded["default_tools"] = kwargs.get("default_tools")
            recorded["default_middleware"] = kwargs.get("default_middleware")
            return original_init(self, *args, **kwargs)

        with patch("deepagents.graph.create_agent", return_value=fake_agent):
            with patch.object(SubAgentMiddleware, "__init__", tracking_init):
                create_deep_agent(model=model, tools=[], enable_dmail=True)

        tools = recorded["default_tools"] or []
        assert any(getattr(tool, "name", "") == "mark_checkpoint" for tool in tools)
        middleware_list = recorded["default_middleware"] or []
        assert any(isinstance(mw, DMailMiddleware) for mw in middleware_list)


class TestRunUntilStable:
    def test_reinvokes_when_resume_flag_present(self):
        outputs = [
            {"messages": [ToolMessage(content="rewind", name="send_dmail", tool_call_id="1")], DMAIL_RESUME_FLAG: True},
            {"messages": [AIMessage(content="done")]},
        ]
        graph = FakeGraph(outputs)

        result = run_until_stable(graph, {"messages": [HumanMessage(content="start")]})

        assert len(graph.invocations) == 2
        assert not result.get("_dmail_resume_required__")
        assert result["messages"][0].content == "done"


class TestAutoCheckpointingV2:
    def test_before_first_user_message_creates_task_start_once(self):
        state: dict[str, Any] = {"messages": [HumanMessage(content="hello")], "dmail_checkpoints": []}
        runtime = DummyRuntime(checkpoint_id="chk-001", thread_id="thread-1")
        middleware = DMailMiddleware(auto_checkpoints=True)

        middleware.before_agent(state, runtime)

        checkpoints = state["dmail_checkpoints"]
        assert len(checkpoints) == 1
        assert checkpoints[0]["reason"] == "auto:before_first_message"
        assert checkpoints[0]["name"].startswith("task-start")
        counters = state["dmail_counters"]
        assert counters["before"] == 1

        # Re-running before_agent should not create a duplicate when a checkpoint already exists.
        middleware.before_agent(state, runtime)
        assert len(state["dmail_checkpoints"]) == 1

    def test_before_every_tool_binds_alias_and_counts_quota(self):
        thresholds = DMailThresholds(
            before_first_user_message=False,
            after_every_tool=False,
            after_agent_response=False,
            max_auto_before_per_run=1,
        )
        middleware = DMailMiddleware(auto_checkpoints=True, thresholds=thresholds)
        runtime = DummyRuntime(checkpoint_id="chk-010", thread_id="thread-1")
        state: dict[str, Any] = {"messages": [HumanMessage(content="turn")], "dmail_checkpoints": []}
        middleware.before_agent(state, runtime)

        def handler(req):
            return None

        request = ToolCallRequest(
            tool_call={"name": "read_file", "args": {}, "id": "call-1"},
            tool=lambda: None,
            state=state,
            runtime=_tool_runtime(state, context=runtime),
        )
        middleware.wrap_tool_call(request, handler)

        checkpoints = state["dmail_checkpoints"]
        assert len(checkpoints) == 1
        assert checkpoints[0]["reason"] == "auto:before_read_file"
        assert state["dmail_counters"]["before"] == 1

        # Quota reached; subsequent calls should not add new before-checkpoints.
        request_2 = ToolCallRequest(
            tool_call={"name": "read_file", "args": {}, "id": "call-2"},
            tool=lambda: None,
            state=state,
            runtime=_tool_runtime(state, context=runtime),
        )
        middleware.wrap_tool_call(request_2, handler)
        assert len(state["dmail_checkpoints"]) == 1

    def test_before_every_tool_skips_dmail_tools(self):
        thresholds = DMailThresholds(before_first_user_message=False)
        middleware = DMailMiddleware(auto_checkpoints=True, thresholds=thresholds)
        runtime = DummyRuntime(checkpoint_id="chk-020", thread_id="thread-1")
        state: dict[str, Any] = {"messages": [HumanMessage(content="turn")], "dmail_checkpoints": []}

        middleware.before_agent(state, runtime)

        def handler(req):
            return send_dmail.invoke(req.tool_call["args"])

        request = ToolCallRequest(
            tool_call={"name": "send_dmail", "args": {"checkpoint": "alpha", "message": "note"}, "id": "call-1"},
            tool=send_dmail,
            state=state,
            runtime=_tool_runtime(state, context=runtime),
        )

        middleware.wrap_tool_call(request, handler)
        assert not state.get("dmail_checkpoints")  # No auto-checkpoint recorded
        assert state["dmail_counters"]["before"] == 0

    def test_after_every_tool_success_and_error_cases(self):
        thresholds = DMailThresholds(
            before_first_user_message=False,
            before_every_tool=False,
            after_every_tool=True,
            after_agent_response=False,
            max_auto_after_per_run=5,
        )
        middleware = DMailMiddleware(auto_checkpoints=True, thresholds=thresholds)

        # Success case
        runtime_success = DummyRuntime(checkpoint_id="chk-030", thread_id="thread-1")
        state_success: dict[str, Any] = {"messages": [HumanMessage(content="run")], "dmail_checkpoints": []}
        middleware.before_agent(state_success, runtime_success)

        def success_handler(req):
            return "ok"

        request_success = ToolCallRequest(
            tool_call={"name": "shell", "args": {}, "id": "call-success"},
            tool=lambda: None,
            state=state_success,
            runtime=_tool_runtime(state_success, context=runtime_success),
        )
        middleware.wrap_tool_call(request_success, success_handler)
        checkpoints = state_success["dmail_checkpoints"]
        assert checkpoints[-1]["reason"] == "auto:after_shell_success"
        assert state_success["dmail_counters"]["after"] == 1

        # Error via ToolMessage
        runtime_error = DummyRuntime(checkpoint_id="chk-031", thread_id="thread-1")
        state_error: dict[str, Any] = {"messages": [HumanMessage(content="run")], "dmail_checkpoints": []}
        middleware.before_agent(state_error, runtime_error)

        def toolerror_handler(req):
            return ToolMessage(content="Error: failed", tool_call_id=req.tool_call.get("id"), name="shell")

        request_error = ToolCallRequest(
            tool_call={"name": "shell", "args": {}, "id": "call-error"},
            tool=lambda: None,
            state=state_error,
            runtime=_tool_runtime(state_error, context=runtime_error),
        )
        middleware.wrap_tool_call(request_error, toolerror_handler)
        assert state_error["dmail_checkpoints"][-1]["reason"] == "auto:after_shell_error"

        # Error via exception
        runtime_exc = DummyRuntime(checkpoint_id="chk-032", thread_id="thread-1")
        state_exc: dict[str, Any] = {"messages": [HumanMessage(content="run")], "dmail_checkpoints": []}
        middleware.before_agent(state_exc, runtime_exc)

        def raise_handler(req):
            raise ValueError("boom")

        request_exc = ToolCallRequest(
            tool_call={"name": "shell", "args": {}, "id": "call-exc"},
            tool=lambda: None,
            state=state_exc,
            runtime=_tool_runtime(state_exc, context=runtime_exc),
        )
        try:
            middleware.wrap_tool_call(request_exc, raise_handler)
        except ValueError:
            pass
        assert state_exc["dmail_checkpoints"][-1]["reason"] == "auto:after_shell_error"

    def test_after_agent_response_on_user_ai_boundary(self):
        thresholds = DMailThresholds(
            before_first_user_message=False,
            before_every_tool=False,
            after_every_tool=False,
            after_agent_response=True,
        )
        middleware = DMailMiddleware(auto_checkpoints=True, thresholds=thresholds)
        runtime = DummyRuntime(checkpoint_id="chk-040", thread_id="thread-1")
        state: dict[str, Any] = {
            "messages": [HumanMessage(content="question"), AIMessage(content="answer")],
            "dmail_checkpoints": [],
        }

        middleware.after_agent(state, runtime)
        checkpoints = state["dmail_checkpoints"]
        assert len(checkpoints) == 1
        assert checkpoints[0]["reason"] == "auto:after_response"
        assert state["dmail_counters"]["after"] == 1

    def test_counters_reset_each_user_turn(self):
        thresholds = DMailThresholds(
            before_first_user_message=False,
            after_every_tool=False,
            after_agent_response=False,
            max_auto_before_per_run=1,
        )
        middleware = DMailMiddleware(auto_checkpoints=True, thresholds=thresholds)
        runtime = DummyRuntime(checkpoint_id="chk-050", thread_id="thread-1")
        state: dict[str, Any] = {
            "messages": [HumanMessage(content="turn1")],
            "dmail_checkpoints": [],
        }

        middleware.before_agent(state, runtime)

        def handler(req):
            return None

        request = ToolCallRequest(
            tool_call={"name": "ls", "args": {}, "id": "call-1"},
            tool=lambda: None,
            state=state,
            runtime=_tool_runtime(state, context=runtime),
        )
        middleware.wrap_tool_call(request, handler)
        assert state["dmail_counters"]["before"] == 1

        state["messages"].append(AIMessage(content="ack"))
        state["messages"].append(HumanMessage(content="turn2"))
        middleware.before_agent(state, runtime)
        assert state["dmail_counters"]["before"] == 0

    def test_retention_cap_does_not_drop_recent_aliases_when_large_quota(self):
        thresholds = DMailThresholds(
            before_first_user_message=False,
            after_every_tool=False,
            after_agent_response=False,
        )
        state: dict[str, Any] = {
            "dmail_checkpoints": [
                {"id": "old", "name": "old-1", "created_at": "t1", "reason": "manual"},
                {"id": "old", "name": "old-2", "created_at": "t2", "reason": "manual"},
                {"id": "old", "name": "old-3", "created_at": "t3", "reason": "manual"},
            ]
        }
        runtime = DummyRuntime(checkpoint_id="chk-060", thread_id="thread-1")
        middleware = DMailMiddleware(auto_checkpoints=True, thresholds=thresholds, max_checkpoints=3)

        def handler(req):
            return None

        request = ToolCallRequest(
            tool_call={"name": "shell", "args": {}, "id": "call-1"},
            tool=lambda: None,
            state=state,
            runtime=_tool_runtime(state, context=runtime),
        )

        middleware.wrap_tool_call(request, handler)

        checkpoints = state["dmail_checkpoints"]
        assert len(checkpoints) == 3
        names = {cp["name"] for cp in checkpoints}
        assert "old-1" not in names  # Oldest entry trimmed
        assert any(cp["reason"] == "auto:before_shell" for cp in checkpoints)

    def test_send_dmail_with_attachments(self):
        from langgraph.types import Command

        command = send_dmail.invoke({
            "checkpoint": "alpha",
            "message": "Key findings",
            "attach_files": ["/analysis/results.txt"],
            "persist_attachments": True,
        })
        assert command.goto == END
        req = command.update["__dmail_request__"]
        assert req["attach_files"] == ["/analysis/results.txt"]
        assert req["persist_attachments"] is True

    def test_persist_attachments_copies_state_files(self):
        state: dict[str, Any] = {
            "files": {
                "/scratch/notes.txt": {"content": ["line1", "line2"], "type": "file"},
            }
        }
        runtime = DummyRuntime(checkpoint_id="chk-123", thread_id="thread-1")
        runtime.state = state
        runtime.store = MagicMock()
        middleware = DMailMiddleware()

        attached = middleware._persist_attachments(runtime, ["/scratch/notes.txt"])
        assert len(attached) == 1
        assert "/memories/dmail/" in attached[0]
        assert "notes.txt" in attached[0]

    def test_dmail_message_includes_attachments(self):
        state: dict[str, Any] = {
            "dmail_checkpoints": [{"id": "chk-000", "name": "alpha", "created_at": "now", "reason": "test"}],
            "__dmail_request__": {
                "checkpoint": "alpha",
                "message": "Important findings.",
                "attached_mem_paths": ["/memories/dmail/20250131120000/results.txt"],
            },
        }
        runtime = DummyRuntime(checkpoint_id="future-123", thread_id="thread-1")
        middleware = DMailMiddleware()

        middleware.after_agent(state, runtime)

        config, values = runtime.graph.updated.pop()
        dmail_message = values["messages"][0]
        assert isinstance(dmail_message, SystemMessage)
        assert "/memories/dmail/20250131120000/results.txt" in dmail_message.content
        assert dmail_message.additional_kwargs.get("attached_mem_paths") == ["/memories/dmail/20250131120000/results.txt"]

    def test_persist_attachments_rejects_path_traversal(self):
        state: dict[str, Any] = {
            "files": {
                "/safe/file.txt": {"content": ["safe content"], "type": "file"},
                "../../../etc/passwd": {"content": ["malicious"], "type": "file"},
            }
        }
        runtime = DummyRuntime(checkpoint_id="chk-123", thread_id="thread-1")
        runtime.state = state
        runtime.store = MagicMock()
        middleware = DMailMiddleware()

        attached = middleware._persist_attachments(runtime, ["/safe/file.txt", "../../../etc/passwd"])
        assert len(attached) == 1
        assert "/memories/dmail/" in attached[0]
        assert "file.txt" in attached[0]
        assert "../../../etc/passwd" not in str(attached)

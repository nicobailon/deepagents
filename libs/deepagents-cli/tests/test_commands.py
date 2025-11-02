"""Tests for slash command handlers."""

from unittest.mock import Mock

from langchain_core.messages import SystemMessage

from deepagents_cli.commands import handle_command
from deepagents_cli.ui import TokenTracker


def test_dmail_command_with_no_checkpoints(tmp_path):
    """Test /dmail command when no checkpoints exist."""
    # Mock agent with get_state method
    mock_agent = Mock()
    mock_agent.get_state.return_value.values = {
        "dmail_checkpoints": [],
        "messages": [],
    }

    # Create token tracker
    token_tracker = TokenTracker()

    # Create agent directory
    agent_dir = tmp_path / "test-agent"
    agent_dir.mkdir()

    # Handle /dmail command
    result = handle_command("/dmail", mock_agent, token_tracker, agent_dir, enable_dmail=True)

    # Should return True (command was handled)
    assert result is True

    # Check that get_state was called
    mock_agent.get_state.assert_called_once()


def test_dmail_checkpoints_subcommand(tmp_path, capsys):
    """Test /dmail checkpoints subcommand displays checkpoint list."""
    # Mock agent with get_state method
    mock_agent = Mock()
    mock_agent.get_state.return_value.values = {
        "dmail_checkpoints": [
            {
                "id": "checkpoint-1",
                "name": "cp1",
                "reason": "manual",
                "created_at": "2024-01-01T10:30:45",
            },
            {
                "id": "checkpoint-2",
                "name": "cp2",
                "reason": "auto",
                "created_at": "2024-01-01T10:31:00",
            },
        ],
        "messages": [],
    }

    # Create token tracker
    token_tracker = TokenTracker()

    # Create agent directory
    agent_dir = tmp_path / "test-agent"
    agent_dir.mkdir()

    # Handle /dmail checkpoints command
    result = handle_command(
        "/dmail checkpoints", mock_agent, token_tracker, agent_dir, enable_dmail=True
    )

    # Should return True (command was handled)
    assert result is True

    # Verify output contains checkpoint names
    captured = capsys.readouterr()
    assert "cp1" in captured.out or "cp2" in captured.out or "Checkpoint Aliases" in captured.out


def test_dmail_log_subcommand(tmp_path, capsys):
    """Test /dmail log subcommand displays D-Mail timeline."""
    # Create D-Mail message
    dmail_message = SystemMessage(
        content="Rewinding to earlier checkpoint",
        name="dmail",
        additional_kwargs={
            "dmail": True,
            "from_checkpoint_id": "checkpoint-1",
            "to_checkpoint_id": "checkpoint-2",
            "attached_mem_paths": [],
            "reason": "user requested rewind",
        },
    )

    # Mock agent with get_state method
    mock_agent = Mock()
    mock_agent.get_state.return_value.values = {
        "dmail_checkpoints": [
            {"id": "checkpoint-1", "name": "cp1"},
            {"id": "checkpoint-2", "name": "cp2"},
        ],
        "messages": [dmail_message],
    }

    # Create token tracker
    token_tracker = TokenTracker()

    # Create agent directory
    agent_dir = tmp_path / "test-agent"
    agent_dir.mkdir()

    # Handle /dmail log command
    result = handle_command("/dmail log", mock_agent, token_tracker, agent_dir, enable_dmail=True)

    # Should return True (command was handled)
    assert result is True

    # Verify output contains timeline info
    captured = capsys.readouterr()
    assert "Timeline" in captured.out or "cp1" in captured.out or "cp2" in captured.out


def test_dmail_config_show(tmp_path, capsys):
    """Test /dmail config displays current settings."""
    # Mock agent
    mock_agent = Mock()
    mock_agent.get_state.return_value.values = {
        "dmail_checkpoints": [],
        "messages": [],
    }

    # Create token tracker
    token_tracker = TokenTracker()

    # Create agent directory with config
    agent_dir = tmp_path / "test-agent"
    agent_dir.mkdir()

    # Create config file
    config_file = agent_dir / "config.toml"
    config_file.write_text(
        """
[dmail]
enabled = true
auto_checkpoints = false
max_auto_before_per_run = 7
max_auto_after_per_run = 9
before_every_tool = false
after_every_tool = true
after_agent_response = true
before_first_user_message = false
"""
    )

    # Handle /dmail config command
    result = handle_command(
        "/dmail config", mock_agent, token_tracker, agent_dir, enable_dmail=True
    )

    # Should return True (command was handled)
    assert result is True

    # Verify output contains config values
    captured = capsys.readouterr()
    assert "max_auto_before_per_run" in captured.out


def test_dmail_disabled_message(tmp_path, capsys):
    """Test /dmail shows helpful message when D-Mail is disabled."""
    # Mock agent
    mock_agent = Mock()
    mock_agent.get_state.return_value.values = {
        "dmail_checkpoints": [],
        "messages": [],
    }

    # Create token tracker
    token_tracker = TokenTracker()

    # Create agent directory
    agent_dir = tmp_path / "test-agent"
    agent_dir.mkdir()

    # Handle /dmail command with D-Mail disabled
    result = handle_command("/dmail", mock_agent, token_tracker, agent_dir, enable_dmail=False)

    # Should return True (command was handled)
    assert result is True

    # The render_dmail_audit function should show a message about D-Mail being disabled
    # The test verifies the command runs without errors
    _ = capsys.readouterr()  # Consume output


def test_help_command(tmp_path):
    """Test /help command executes without errors."""
    # Mock agent
    mock_agent = Mock()

    # Create token tracker
    token_tracker = TokenTracker()

    # Create agent directory
    agent_dir = tmp_path / "test-agent"
    agent_dir.mkdir()

    # Handle /help command
    result = handle_command("/help", mock_agent, token_tracker, agent_dir)

    # Should return True (command was handled)
    assert result is True


def test_unknown_command(tmp_path, capsys):
    """Test unknown slash commands show error message."""
    # Mock agent
    mock_agent = Mock()

    # Create token tracker
    token_tracker = TokenTracker()

    # Create agent directory
    agent_dir = tmp_path / "test-agent"
    agent_dir.mkdir()

    # Handle unknown command
    result = handle_command("/unknown", mock_agent, token_tracker, agent_dir)

    # Should return True (command was handled, even if unknown)
    assert result is True

    # Verify output contains error message
    captured = capsys.readouterr()
    assert "Unknown command" in captured.out

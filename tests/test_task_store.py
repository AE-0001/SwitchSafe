from pathlib import Path

import pytest

from switchsafe.task_store import TaskStore, TaskTools


def test_task_tools_persist_real_state(tmp_path: Path):
    store = TaskStore(tmp_path / "tasks.db")
    tools = TaskTools(store)
    created = tools.execute("create_task", {"title": "Submit application"}, approved=True)
    assert tools.execute("list_tasks", {})[0]["title"] == "Submit application"
    tools.execute("complete_task", {"task_id": created["id"]}, approved=True)
    assert tools.execute("list_tasks", {})[0]["status"] == "completed"


def test_mutation_requires_confirmation(tmp_path: Path):
    tools = TaskTools(TaskStore(tmp_path / "tasks.db"))
    with pytest.raises(PermissionError):
        tools.execute("save_note", {"text": "private note"})


def test_unknown_tool_is_blocked(tmp_path: Path):
    tools = TaskTools(TaskStore(tmp_path / "tasks.db"))
    with pytest.raises(ValueError, match="not allowlisted"):
        tools.execute("send_money", {"amount": 100})

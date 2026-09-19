from pathlib import Path

from switchsafe.task_pipeline import VoiceTaskPipeline
from switchsafe.task_store import TaskStore, TaskTools


class FakePlanner:
    def __init__(self, actions):
        self.actions = actions

    def task_actions(self, query: str):
        return {"actions": self.actions, "schema_valid": True, "latency_ms": 1.0}


def pipeline(tmp_path: Path, actions):
    store = TaskStore(tmp_path / "tasks.db")
    return VoiceTaskPipeline(FakePlanner(actions), TaskTools(store)), store


def test_unapproved_task_is_not_written(tmp_path: Path):
    agent, store = pipeline(tmp_path, [{"tool": "create_task", "arguments": {"title": "Apply"}}])
    result = agent.run_text("Add Apply to my tasks")
    assert result["status"] == "awaiting_confirmation"
    assert store.list_tasks() == []


def test_approved_task_is_persisted_and_verified(tmp_path: Path):
    agent, store = pipeline(tmp_path, [{"tool": "create_task", "arguments": {"title": "Apply"}}])
    result = agent.run_text("Add Apply to my tasks", approved=True)
    assert result["status"] == "completed"
    assert store.list_tasks()[0]["title"] == "Apply"
    assert "verified action" in result["answer"]


def test_read_only_action_needs_no_confirmation(tmp_path: Path):
    agent, _ = pipeline(tmp_path, [{"tool": "list_tasks", "arguments": {}}])
    assert agent.run_text("Show my tasks")["status"] == "completed"

from __future__ import annotations

import sqlite3
from pathlib import Path


class TaskStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS tasks (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              title TEXT NOT NULL,
              status TEXT NOT NULL DEFAULT 'pending',
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS notes (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              body TEXT NOT NULL,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

    def create_task(self, title: str) -> dict:
        title = title.strip()
        if not title:
            raise ValueError("task title cannot be empty")
        cursor = self.connection.execute("INSERT INTO tasks(title) VALUES (?)", (title,))
        self.connection.commit()
        return {"id": cursor.lastrowid, "title": title, "status": "pending"}

    def list_tasks(self) -> list[dict]:
        rows = self.connection.execute(
            "SELECT id, title, status, created_at FROM tasks ORDER BY id"
        ).fetchall()
        return [dict(row) for row in rows]

    def complete_task(self, task_id: int) -> dict:
        cursor = self.connection.execute(
            "UPDATE tasks SET status='completed' WHERE id=?", (task_id,)
        )
        self.connection.commit()
        if cursor.rowcount != 1:
            raise ValueError(f"task {task_id} was not found")
        return {"id": task_id, "status": "completed"}

    def save_note(self, text: str) -> dict:
        text = text.strip()
        if not text:
            raise ValueError("note cannot be empty")
        cursor = self.connection.execute("INSERT INTO notes(body) VALUES (?)", (text,))
        self.connection.commit()
        return {"id": cursor.lastrowid, "body": text}

    def search_notes(self, query: str) -> list[dict]:
        rows = self.connection.execute(
            "SELECT id, body, created_at FROM notes WHERE body LIKE ? ORDER BY id",
            (f"%{query.strip()}%",),
        ).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        self.connection.close()


class TaskTools:
    MUTATING = {"create_task", "complete_task", "save_note"}
    ALLOWED = MUTATING | {"list_tasks", "search_notes"}

    def __init__(self, store: TaskStore):
        self.store = store

    def execute(self, tool: str, arguments: dict, approved: bool = False):
        if tool not in self.ALLOWED:
            raise ValueError(f"tool not allowlisted: {tool}")
        if tool in self.MUTATING and not approved:
            raise PermissionError(f"confirmation required for {tool}")
        if tool == "create_task":
            return self.store.create_task(str(arguments.get("title", "")))
        if tool == "list_tasks":
            return self.store.list_tasks()
        if tool == "complete_task":
            return self.store.complete_task(int(arguments["task_id"]))
        if tool == "save_note":
            return self.store.save_note(str(arguments.get("text", "")))
        return self.store.search_notes(str(arguments.get("query", "")))

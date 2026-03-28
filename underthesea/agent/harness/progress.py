"""Progress tracking for multi-session agent tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path


class SubtaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Subtask:
    id: int
    description: str
    status: SubtaskStatus = SubtaskStatus.PENDING
    result_summary: str | None = None
    started_at: str | None = None
    completed_at: str | None = None


@dataclass
class ProgressData:
    task: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    subtasks: list[Subtask] = field(default_factory=list)


class ProgressTracker:
    """Track progress of a multi-step task via JSON file."""

    def __init__(self, progress_file: str | Path):
        self._path = Path(progress_file)
        self._data: ProgressData | None = None

    def create(self, task: str, subtasks: list[str]) -> ProgressData:
        """Create a new progress file with subtask list."""
        self._data = ProgressData(
            task=task,
            subtasks=[
                Subtask(id=i + 1, description=desc)
                for i, desc in enumerate(subtasks)
            ],
        )
        self._save()
        return self._data

    def load(self) -> ProgressData:
        """Load progress from file."""
        with open(self._path) as f:
            raw = json.load(f)
        subtasks = [
            Subtask(
                id=s["id"],
                description=s["description"],
                status=SubtaskStatus(s["status"]),
                result_summary=s.get("result_summary"),
                started_at=s.get("started_at"),
                completed_at=s.get("completed_at"),
            )
            for s in raw["subtasks"]
        ]
        self._data = ProgressData(
            task=raw["task"],
            created_at=raw["created_at"],
            updated_at=raw["updated_at"],
            subtasks=subtasks,
        )
        return self._data

    def update_subtask(
        self,
        subtask_id: int,
        status: SubtaskStatus,
        result_summary: str | None = None,
    ) -> None:
        """Update a subtask's status and optional result."""
        if self._data is None:
            self.load()
        for st in self._data.subtasks:
            if st.id == subtask_id:
                st.status = status
                if result_summary:
                    st.result_summary = result_summary
                if status == SubtaskStatus.IN_PROGRESS:
                    st.started_at = datetime.now().isoformat()
                elif status in (SubtaskStatus.COMPLETED, SubtaskStatus.FAILED):
                    st.completed_at = datetime.now().isoformat()
                break
        self._data.updated_at = datetime.now().isoformat()
        self._save()

    def next_pending(self) -> Subtask | None:
        """Get the next pending subtask."""
        if self._data is None:
            self.load()
        for st in self._data.subtasks:
            if st.status == SubtaskStatus.PENDING:
                return st
        return None

    def is_complete(self) -> bool:
        """Check if all subtasks are completed or skipped."""
        if self._data is None:
            self.load()
        return all(
            st.status in (SubtaskStatus.COMPLETED, SubtaskStatus.SKIPPED)
            for st in self._data.subtasks
        )

    def _save(self) -> None:
        """Persist progress data to JSON file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "task": self._data.task,
            "created_at": self._data.created_at,
            "updated_at": self._data.updated_at,
            "subtasks": [
                {
                    "id": st.id,
                    "description": st.description,
                    "status": st.status.value,
                    "result_summary": st.result_summary,
                    "started_at": st.started_at,
                    "completed_at": st.completed_at,
                }
                for st in self._data.subtasks
            ],
        }
        with open(self._path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

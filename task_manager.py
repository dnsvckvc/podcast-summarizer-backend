import time
import threading
from enum import Enum
from typing import Dict, Optional
from dataclasses import dataclass, asdict


class TaskStatus(Enum):
    FAILED = "failed"
    PENDING = "pending"
    COMPLETED = "completed"
    DOWNLOADING = "downloading"
    SUMMARIZING = "summarizing"
    TRANSCRIBING = "transcribing"


@dataclass
class TaskInfo:
    task_id: str
    status: TaskStatus
    progress: float = 0.0
    message: str = ""
    created_at: float = None
    updated_at: float = None
    result: Optional[Dict] = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.updated_at is None:
            self.updated_at = time.time()

    def update(
        self,
        status: TaskStatus = None,
        progress: float = None,
        message: str = None,
        result: Dict = None,
        error: str = None,
    ):
        if status is not None:
            self.status = status
        if progress is not None:
            self.progress = progress
        if message is not None:
            self.message = message
        if result is not None:
            self.result = result
        if error is not None:
            self.error = error
        self.updated_at = time.time()


class TaskManager:
    def __init__(self):
        self.tasks: Dict[str, TaskInfo] = {}
        self.lock = threading.Lock()

    def create_task(self, task_id: str) -> TaskInfo:
        with self.lock:
            task = TaskInfo(
                task_id=task_id, status=TaskStatus.PENDING, message="Task created"
            )
            self.tasks[task_id] = task
            return task

    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        with self.lock:
            return self.tasks.get(task_id)

    def update_task(self, task_id: str, **kwargs) -> Optional[TaskInfo]:
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id].update(**kwargs)
                return self.tasks[task_id]
            return None

    def get_task_dict(self, task_id: str) -> Optional[Dict]:
        task = self.get_task(task_id)
        if task:
            task_dict = asdict(task)
            task_dict["status"] = task.status.value
            return task_dict
        return None

    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """Remove tasks older than max_age_hours"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        with self.lock:
            expired_tasks = [
                task_id
                for task_id, task in self.tasks.items()
                if current_time - task.created_at > max_age_seconds
            ]
            for task_id in expired_tasks:
                del self.tasks[task_id]

"""Task model definitions."""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class TaskEnvironmentMode(str, Enum):
    """Controls how much environment information is generated for a task."""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    RANDOM = "random"


@dataclass
class Task:
    """A task description, its environment, and its observation metrics."""

    desc: str
    env: list[Any]
    metrics: list[Any]

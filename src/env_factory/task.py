"""Task model definitions."""

from dataclasses import dataclass
from typing import Any


@dataclass
class Task:
    """A task description, its environment, and its observation metrics."""

    desc: str
    env: list[Any]
    metrics: list[Any]

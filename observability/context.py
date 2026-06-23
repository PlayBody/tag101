"""Per-request context for correlating API spend with validator tasks."""

from __future__ import annotations

import contextvars
from dataclasses import dataclass


@dataclass(frozen=True)
class TaskContext:
    task_id: str
    task_kind: str
    validator_hotkey: str | None
    validator_uid: int | None
    netuid: int
    miner_uid: int | None
    tweet_id: str = ""


task_context: contextvars.ContextVar[TaskContext | None] = contextvars.ContextVar(
    "tag101_task_context",
    default=None,
)


def get_task_context() -> TaskContext | None:
    return task_context.get()

"""Miner observability: validator task logs and API cost tracking."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .context import TaskContext, get_task_context, task_context
from .cost_tracker import CostTracker, daily_cost_summary
from .pricing import estimate_call_cost_usd, model_pricing
from .task_logger import TaskLogger, iter_task_events

_observability: "MinerObservability | None" = None


class MinerObservability:
    def __init__(
        self,
        log_dir: Path,
        *,
        enable_task_log: bool = True,
        enable_cost_log: bool = True,
    ) -> None:
        root = Path(log_dir)
        self.task_logger = TaskLogger(root / "validator_tasks") if enable_task_log else None
        self.cost_tracker = CostTracker(root / "api_cost") if enable_cost_log else None


def configure_observability(
    log_dir: str | Path,
    *,
    enable_task_log: bool = True,
    enable_cost_log: bool = True,
) -> MinerObservability:
    global _observability
    _observability = MinerObservability(
        Path(log_dir),
        enable_task_log=enable_task_log,
        enable_cost_log=enable_cost_log,
    )
    return _observability


def get_observability() -> MinerObservability | None:
    return _observability


@contextmanager
def task_scope(
    *,
    task_id: str,
    task_kind: str,
    validator_hotkey: str | None,
    validator_uid: int | None,
    netuid: int,
    miner_uid: int | None,
    tweet_id: str = "",
) -> Iterator[TaskContext]:
    ctx = TaskContext(
        task_id=task_id,
        task_kind=task_kind,
        validator_hotkey=validator_hotkey,
        validator_uid=validator_uid,
        netuid=netuid,
        miner_uid=miner_uid,
        tweet_id=tweet_id,
    )
    token = task_context.set(ctx)
    try:
        yield ctx
    finally:
        task_context.reset(token)


def log_validator_task(
    *,
    task_id: str,
    task_kind: str,
    validator_hotkey: str | None,
    validator_uid: int | None,
    netuid: int,
    miner_uid: int | None,
    post_text: str,
    tweet_id: str,
    tags: list[str],
    elapsed_sec: float,
    success: bool,
    error: str | None = None,
) -> None:
    obs = get_observability()
    if obs is None or obs.task_logger is None:
        return
    obs.task_logger.record_validator_task(
        task_id=task_id,
        task_kind=task_kind,
        validator_hotkey=validator_hotkey,
        validator_uid=validator_uid,
        netuid=netuid,
        miner_uid=miner_uid,
        post_text=post_text,
        tweet_id=tweet_id,
        tags=tags,
        elapsed_sec=elapsed_sec,
        success=success,
        error=error,
    )


def record_api_cost(
    *,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    task_id: str | None = None,
) -> dict[str, Any] | None:
    obs = get_observability()
    if obs is None or obs.cost_tracker is None:
        return None
    ctx = get_task_context()
    return obs.cost_tracker.record(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        task_id=task_id or (ctx.task_id if ctx else None),
    )

"""Append-only JSONL logs for validator→miner task events."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


class TaskLogger:
    def __init__(self, log_dir: Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.log_dir / "events.jsonl"
        self._lock = threading.Lock()

    def record_validator_task(
        self,
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
    ) -> dict[str, Any]:
        now = time.time()
        event = {
            "event": "validator_task",
            "timestamp": now,
            "date": _utc_date(now),
            "task_id": task_id,
            "task_kind": task_kind,
            "validator_hotkey": validator_hotkey or "",
            "validator_uid": validator_uid,
            "netuid": int(netuid),
            "miner_uid": miner_uid,
            "tweet_id": tweet_id or "",
            "post_text": post_text,
            "tags": tags,
            "elapsed_sec": round(float(elapsed_sec), 6),
            "success": bool(success),
            "error": error or "",
        }
        with self._lock:
            self._append_jsonl(self.events_path, event)
        return event

    @staticmethod
    def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def iter_task_events(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        return iter(())
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") == "validator_task":
            yield event


def _utc_date(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")

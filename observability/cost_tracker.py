"""Track OpenAI API usage and daily spend."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .pricing import estimate_call_cost_usd


class CostTracker:
    def __init__(self, log_dir: Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.log_dir / "events.jsonl"
        self._lock = threading.Lock()
        self._day_totals: dict[str, float] = {}

    def record(
        self,
        *,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        date_key = _utc_date(now)
        total_tokens = int(prompt_tokens) + int(completion_tokens)
        cost_usd = estimate_call_cost_usd(
            model=model,
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
        )
        with self._lock:
            day_total = self._day_totals.get(date_key, 0.0) + cost_usd
            self._day_totals[date_key] = day_total
            event = {
                "event": "api_cost",
                "timestamp": now,
                "date": date_key,
                "task_id": task_id or "",
                "model": model,
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": total_tokens,
                "cost_usd": cost_usd,
                "cost_usd_day_total": round(day_total, 8),
            }
            self._append_jsonl(self.events_path, event)
            self._write_daily_summary(date_key)
        return event

    def _write_daily_summary(self, date_key: str) -> None:
        summary = daily_cost_summary(self.events_path, date=date_key)
        daily_path = self.log_dir / "daily" / f"{date_key}.json"
        daily_path.parent.mkdir(parents=True, exist_ok=True)
        daily_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    @staticmethod
    def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def daily_cost_summary(events_path: Path, *, date: str | None = None) -> dict[str, Any]:
    target_date = date or _utc_date(time.time())
    totals_by_model: dict[str, dict[str, float | int]] = {}
    total_cost = 0.0
    total_calls = 0
    total_tokens = 0

    if not events_path.is_file():
        return {
            "date": target_date,
            "total_cost_usd": 0.0,
            "total_calls": 0,
            "total_tokens": 0,
            "by_model": {},
        }

    for event in _iter_jsonl(events_path):
        if event.get("event") != "api_cost":
            continue
        if str(event.get("date", "")) != target_date:
            continue
        model = str(event.get("model", "unknown"))
        bucket = totals_by_model.setdefault(
            model,
            {"calls": 0, "tokens": 0, "cost_usd": 0.0},
        )
        bucket["calls"] = int(bucket["calls"]) + 1
        bucket["tokens"] = int(bucket["tokens"]) + int(event.get("total_tokens", 0))
        cost = float(event.get("cost_usd", 0.0))
        bucket["cost_usd"] = round(float(bucket["cost_usd"]) + cost, 8)
        total_cost += cost
        total_calls += 1
        total_tokens += int(event.get("total_tokens", 0))

    return {
        "date": target_date,
        "total_cost_usd": round(total_cost, 8),
        "total_calls": total_calls,
        "total_tokens": total_tokens,
        "by_model": totals_by_model,
    }


def _iter_jsonl(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def _utc_date(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")

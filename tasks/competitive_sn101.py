"""SN101 task handler using the competitive (scoring-aware) miner."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..chain.runtime import ChainRuntime
from ..protocol import TaskEnvelope
from .competitive_sn101_scoring import score_answers as shared_score_answers
from .framework import ScoreBreakdown, TaskHandler
from .sn101 import KIND, SN101_MAX_TAGS, SN101_TIME_LIMIT, SPEC_VERSION
from .sn101_reference.core.competitive_miner import CompetitiveMiner


def solve_problem(envelope: TaskEnvelope, chain_runtime: ChainRuntime) -> dict[str, Any]:
    _ = chain_runtime
    task_payload = dict(envelope.payload)
    post = str(task_payload.get("text", ""))
    miner = CompetitiveMiner(n_tags=SN101_MAX_TAGS, timeout_sec=int(SN101_TIME_LIMIT))
    tags = miner.generate_tags(post)
    return {"tags": tags}


def score_answers(
    payload: Mapping[str, Any],
    scoring: Mapping[str, Any],
    answers: Sequence[Mapping[str, Any]],
) -> ScoreBreakdown:
    return shared_score_answers(payload, scoring, answers)


def handler() -> TaskHandler:
    return TaskHandler(
        kind=KIND,
        spec_version=SPEC_VERSION,
        solve_problem=solve_problem,
        score_answers=score_answers,
        description="SN101 semantic tagging with scoring-aware competitive miner.",
    )

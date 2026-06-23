"""Shared scoring helpers for SN101 task modules."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from .framework import ScoreBreakdown
from .sn101 import REFERENCE_MODEL_NAME, SN101_MAX_TAGS

_SCORER_CACHE: dict[tuple[str, int, float], Any] = {}


def score_answers(
    payload: Mapping[str, Any],
    scoring: Mapping[str, Any],
    answers: Sequence[Mapping[str, Any]],
) -> ScoreBreakdown:
    post = str(payload.get("text", ""))
    responses = [_tags_from_answer(answer) for answer in answers]
    result = _score_with_reference(
        post=post,
        responses=responses,
        n_tags_per_miner=SN101_MAX_TAGS,
        model_name=str(scoring.get("model_name", REFERENCE_MODEL_NAME)),
        proximity_rank_decay=float(scoring.get("proximity_rank_decay", 1.0)),
    )

    rewards = [float(value) for value in result.get("miner_scores", [])]
    metrics = _miner_metrics(result, len(answers))
    return ScoreBreakdown(
        rewards=rewards,
        metrics=metrics,
        details={
            "method": scoring.get(
                "method",
                "tag101.tasks.sn101_reference.core.scoring.TagScorer",
            ),
            "tweet_id": payload.get("tweet_id", ""),
            "clusters": result.get("clusters", []),
            "spans": result.get("spans", []),
        },
    )


def _score_with_reference(
    *,
    post: str,
    responses: list[list[str]],
    n_tags_per_miner: int,
    model_name: str,
    proximity_rank_decay: float,
) -> dict[str, Any]:
    from .sn101_reference.core.scoring import TagScorer

    cache_key = (model_name, n_tags_per_miner, proximity_rank_decay)
    scorer = _SCORER_CACHE.get(cache_key)
    if scorer is None:
        scorer = TagScorer(
            model_name=model_name,
            n_tags_per_miner=n_tags_per_miner,
            proximity_rank_decay=proximity_rank_decay,
        )
        _SCORER_CACHE[cache_key] = scorer
    return scorer.score(post, responses)


def _miner_metrics(result: Mapping[str, Any], miner_count: int) -> list[dict[str, Any]]:
    per_miner_keys = (
        "normalized_responses",
        "tag_scores",
        "consensus_scores",
        "validity_scores",
        "diversity_scores",
        "validity_details",
        "diversity_details",
    )
    metrics: list[dict[str, Any]] = []
    for index in range(miner_count):
        item: dict[str, Any] = {}
        for key in per_miner_keys:
            values = result.get(key, [])
            if isinstance(values, list) and index < len(values):
                item[key] = values[index]
        metrics.append(item)
    return metrics


def _tags_from_answer(answer: Mapping[str, Any]) -> list[str]:
    raw = answer.get("tags") or answer.get("labels") or answer.get("keywords") or []
    if isinstance(raw, str):
        values: Sequence[Any] = re.split(r"[,;\n|]+", raw)
    elif isinstance(raw, Sequence):
        values = raw
    else:
        values = []

    tags: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        tag = value
        if tag:
            tags.append(tag)
    return tags

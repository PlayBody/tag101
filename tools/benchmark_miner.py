"""Local benchmark and replay tools for Tag101 miners."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Sequence

from tag101.observability.cost_tracker import daily_cost_summary
from tag101.observability.pricing import compare_models_for_typical_tagging_call
from tag101.observability.task_logger import iter_task_events
from tag101.tasks.competitive_sn101_scoring import score_answers
from tag101.tasks.sn101 import SN101_TEST_FALLBACK_TAGS
from tag101.tasks.sn101_reference.core.competitive_miner import CompetitiveMiner
from tag101.tasks.sn101_reference.core.miner import Miner as ReferenceMiner

SAMPLES_PATH = Path(__file__).with_name("sample_posts.json")


def _load_posts(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return [
            {
                "text": (
                    "OpenAI released a new reasoning model today. "
                    "Developers on X are debating latency vs accuracy tradeoffs."
                )
            },
        ]
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    return data.get("posts", [])


def _fallback_tags(n: int = 3) -> list[str]:
    picks = random.sample(list(SN101_TEST_FALLBACK_TAGS), k=min(n, len(SN101_TEST_FALLBACK_TAGS)))
    return picks


def _simulate_crowd_tags(post: str, crowd_size: int, seed: int) -> list[list[str]]:
    rng = random.Random(seed)
    tags_pool = [
        "openai",
        "anthropic",
        "ai model",
        "llm",
        "machine learning",
        "gpu",
        "inference",
        "fine tuning",
        "agents",
        "startup",
        "regulation",
        "benchmark",
    ]
    responses: list[list[str]] = []
    for _ in range(crowd_size):
        if rng.random() < 0.35:
            responses.append(_fallback_tags(3))
            continue
        responses.append(rng.sample(tags_pool, k=3))
    return responses


def _simulate_span_crowd(post: str, crowd_size: int, seed: int) -> list[list[str]]:
    """Proxy crowd: miners pick from the same spaCy spans validators use for validity."""
    from tag101.tasks.sn101_reference.core.scoring.preprocessing import build_spans, normalize_tag

    rng = random.Random(seed)
    spans: list[str] = []
    seen: set[str] = set()
    for raw in build_spans(post):
        tag = normalize_tag(raw)
        if not tag or tag in seen:
            continue
        if not 1 <= len(tag.split()) <= 5:
            continue
        seen.add(tag)
        spans.append(tag)

    responses: list[list[str]] = []
    for index in range(crowd_size):
        if len(spans) >= 3:
            rng.shuffle(spans)
            responses.append(spans[:3])
            continue
        responses.append(_fallback_tags(3))
    return responses


def _crowd_responses(post: str, crowd_size: int, seed: int, *, mode: str) -> list[list[str]]:
    if mode == "span":
        return _simulate_span_crowd(post, crowd_size, seed)
    return _simulate_crowd_tags(post, crowd_size, seed)


def score_batch(
    *,
    post: str,
    labels: Sequence[str],
    responses: Sequence[list[str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    answers = [{"tags": tags} for tags in responses]
    breakdown = score_answers({"text": post}, {}, answers)
    rows: list[dict[str, Any]] = []
    for index, (label, tags, reward) in enumerate(
        zip(labels, responses, breakdown.rewards, strict=False)
    ):
        metrics = breakdown.metrics[index] if index < len(breakdown.metrics) else {}
        rows.append(
            {
                "rank": 0,
                "label": label,
                "score": float(reward),
                "tags": tags,
                "consensus": metrics.get("consensus_scores"),
                "validity": metrics.get("validity_scores"),
                "diversity": metrics.get("diversity_scores"),
                "tag_scores": metrics.get("tag_scores"),
            }
        )
    rows.sort(key=lambda row: row["score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    details = breakdown.details if isinstance(breakdown.details, dict) else {}
    return rows, details


def _print_rows(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        print(
            f"#{row['rank']} {row['label']:16s} score={row['score']:.4f} tags={row['tags']}"
        )
        if row.get("consensus"):
            print(
                f"    consensus={row['consensus']} "
                f"validity={row['validity']} diversity={row['diversity']}"
            )
            if row.get("tag_scores"):
                print(f"    tag_scores={row['tag_scores']}")


def _print_clusters(details: dict[str, Any], *, focus_label: str | None = None) -> None:
    clusters = details.get("clusters") or []
    if not clusters:
        print("  (no cluster data)")
        return
    ranked = sorted(clusters, key=lambda item: float(item.get("support", 0.0)), reverse=True)
    print("  Consensus clusters (support = share of miners in cluster):")
    for cluster in ranked[:8]:
        support = float(cluster.get("support", 0.0))
        tags = cluster.get("tags") or []
        unique_tags = sorted(set(str(tag) for tag in tags))
        preview = unique_tags[:8]
        suffix = "..." if len(unique_tags) > 8 else ""
        print(
            f"    cluster={cluster.get('cluster_id')} support={support:.3f} "
            f"tags={preview}{suffix}"
        )
    if focus_label:
        print(f"  Tip: align '{focus_label}' tags with high-support cluster centers.")


def benchmark_post(
    post: str,
    *,
    crowd_size: int,
    seed: int,
    include_reference: bool,
    crowd_mode: str = "random",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels: list[str] = ["competitive"]
    responses: list[list[str]] = [CompetitiveMiner().generate_tags(post)]

    if include_reference:
        labels.append("reference")
        ref = ReferenceMiner()
        try:
            responses.append(ref.generate_tags(post))
        except RuntimeError:
            responses.append(_fallback_tags(3))

    labels.extend(f"crowd_{index}" for index in range(crowd_size))
    responses.extend(_crowd_responses(post, crowd_size, seed, mode=crowd_mode))
    return score_batch(post=post, labels=labels, responses=responses)


def replay_logged_tasks(
    log_path: Path,
    *,
    crowd_size: int,
    seed: int,
    improve: bool,
    limit: int,
    show_clusters: bool,
    crowd_mode: str = "random",
) -> int:
    if not log_path.is_file():
        print(f"Log file not found: {log_path}", file=sys.stderr)
        return 1

    events = list(iter_task_events(log_path))
    if limit > 0:
        events = events[-limit:]

    if not events:
        print(f"No validator_task events in {log_path}", file=sys.stderr)
        return 1

    improved_wins = 0
    for index, event in enumerate(events, start=1):
        post = str(event.get("post_text", "")).strip()
        if not post:
            continue
        old_tags = [str(tag) for tag in (event.get("tags") or []) if isinstance(tag, str)]
        task_id = str(event.get("task_id", ""))
        print(f"\n=== Replay {index} task={task_id} ===")
        print(f"validator_uid={event.get('validator_uid')} date={event.get('date')}")
        print(post[:240] + ("..." if len(post) > 240 else ""))
        print(f"logged_tags={old_tags}")

        labels = ["logged"]
        responses: list[list[str]] = [old_tags]
        if improve:
            labels.append("improved")
            responses.append(CompetitiveMiner().generate_tags(post))

        labels.extend(f"crowd_{i}" for i in range(crowd_size))
        responses.extend(_crowd_responses(post, crowd_size, seed + index, mode=crowd_mode))

        rows, details = score_batch(post=post, labels=labels, responses=responses)
        _print_rows(rows)
        if show_clusters:
            _print_clusters(details, focus_label="improved" if improve else "logged")

        if improve and len(rows) >= 2:
            improved = next((row for row in rows if row["label"] == "improved"), None)
            logged = next((row for row in rows if row["label"] == "logged"), None)
            if improved and logged and improved["score"] > logged["score"]:
                improved_wins += 1

    if improve:
        print(
            f"\nReplay summary: improved beat logged on "
            f"{improved_wins}/{len(events)} events."
        )
    return 0


def print_cost_report(cost_log: Path, *, date: str | None) -> int:
    events_path = cost_log / "events.jsonl" if cost_log.is_dir() else cost_log
    if events_path.is_dir():
        events_path = events_path / "events.jsonl"
    summary = daily_cost_summary(events_path, date=date)
    print(json.dumps(summary, indent=2))
    comparison = compare_models_for_typical_tagging_call()
    print("\nModel cost comparison (typical tagging call, ~96 tasks/day):")
    print(json.dumps(comparison, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark Tag101 miners locally with the validator TagScorer."
    )
    parser.add_argument("--post", type=str, default="", help="Single post text to score.")
    parser.add_argument(
        "--samples",
        type=Path,
        default=SAMPLES_PATH,
        help="JSON file with {\"posts\": [{\"text\": \"...\"}, ...]}.",
    )
    parser.add_argument(
        "--crowd-size",
        type=int,
        default=20,
        help="Number of simulated competing miners (consensus proxy).",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for crowd simulation.")
    parser.add_argument(
        "--include-reference",
        action="store_true",
        help="Also score the stock reference OpenAI miner (needs OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--show-clusters",
        action="store_true",
        help="Print consensus cluster breakdown after each scored post.",
    )
    parser.add_argument(
        "--crowd-mode",
        choices=("random", "span"),
        default="random",
        help="Crowd simulation: random generic tags (default) or span-based (closer to live miners).",
    )
    parser.add_argument(
        "--replay-log",
        type=Path,
        default=None,
        help="Replay validator task JSONL (miner log validator_tasks/events.jsonl).",
    )
    parser.add_argument(
        "--improve",
        action="store_true",
        help="With --replay-log, also score a fresh competitive-miner answer.",
    )
    parser.add_argument(
        "--replay-limit",
        type=int,
        default=0,
        help="Replay only the last N events (0 = all).",
    )
    parser.add_argument(
        "--cost-log",
        type=Path,
        default=None,
        help="Print daily API cost summary from api_cost/events.jsonl or its directory.",
    )
    parser.add_argument(
        "--cost-date",
        type=str,
        default="",
        help="UTC date YYYY-MM-DD for --cost-log (default: today).",
    )
    args = parser.parse_args(argv)

    if args.cost_log is not None:
        return print_cost_report(args.cost_log, date=args.cost_date or None)

    if args.replay_log is not None:
        return replay_logged_tasks(
            args.replay_log,
            crowd_size=max(0, args.crowd_size),
            seed=args.seed,
            improve=bool(args.improve),
            limit=max(0, args.replay_limit),
            show_clusters=bool(args.show_clusters),
            crowd_mode=args.crowd_mode,
        )

    posts = [{"text": args.post}] if args.post.strip() else _load_posts(args.samples)
    if not posts:
        print("No posts to benchmark.", file=sys.stderr)
        return 1

    competitive_wins = 0
    for index, item in enumerate(posts, start=1):
        post = str(item.get("text", "")).strip()
        if not post:
            continue
        print(f"\n=== Post {index} ===")
        print(post[:240] + ("..." if len(post) > 240 else ""))
        rows, details = benchmark_post(
            post,
            crowd_size=max(0, args.crowd_size),
            seed=args.seed + index,
            include_reference=args.include_reference,
            crowd_mode=args.crowd_mode,
        )
        _print_rows(rows)
        if args.show_clusters:
            _print_clusters(details, focus_label="competitive")
        if rows and rows[0]["label"] == "competitive":
            competitive_wins += 1

    print(
        f"\nSummary: competitive ranked #1 on {competitive_wins}/{len(posts)} posts "
        f"(crowd_size={args.crowd_size})."
    )
    print(
        "Note: live subnet rank also depends on 24h rolling softmax, uptime, "
        "and the real miner crowd — not just single-post scores."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

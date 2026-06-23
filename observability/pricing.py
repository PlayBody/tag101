"""OpenAI model pricing helpers for miner cost tracking.

Prices are USD per 1M tokens. Update when OpenAI changes list prices.
Source baseline: OpenAI API pricing (verify before budgeting).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    input_per_million: float
    output_per_million: float


# USD per 1M tokens
MODEL_PRICING: dict[str, ModelPricing] = {
    "gpt-4o-mini": ModelPricing(input_per_million=0.15, output_per_million=0.60),
    "gpt-4o": ModelPricing(input_per_million=2.50, output_per_million=10.00),
    "gpt-4o-2024-07-18": ModelPricing(input_per_million=2.50, output_per_million=10.00),
    "gpt-4.1-mini": ModelPricing(input_per_million=0.40, output_per_million=1.60),
    "gpt-4.1": ModelPricing(input_per_million=2.00, output_per_million=8.00),
}


def model_pricing(model: str) -> ModelPricing:
    normalized = model.strip().lower()
    if normalized in MODEL_PRICING:
        return MODEL_PRICING[normalized]
    for key, pricing in MODEL_PRICING.items():
        if normalized.startswith(key):
            return pricing
    # Conservative default for unknown chat models
    return ModelPricing(input_per_million=2.50, output_per_million=10.00)


def estimate_call_cost_usd(
    *,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    pricing = model_pricing(model)
    input_cost = (max(prompt_tokens, 0) / 1_000_000.0) * pricing.input_per_million
    output_cost = (max(completion_tokens, 0) / 1_000_000.0) * pricing.output_per_million
    return round(input_cost + output_cost, 8)


def compare_models_for_typical_tagging_call(
    *,
    prompt_tokens: int = 500,
    completion_tokens: int = 30,
    calls_per_day: int = 96,
) -> dict[str, dict[str, float]]:
    """Compare common models for one tagging call and a full day (~15 min cadence)."""
    out: dict[str, dict[str, float]] = {}
    for model in ("gpt-4o-mini", "gpt-4o"):
        per_call = estimate_call_cost_usd(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        out[model] = {
            "per_call_usd": per_call,
            "per_day_usd": round(per_call * calls_per_day, 6),
            "per_month_usd": round(per_call * calls_per_day * 30, 4),
        }
    mini = out["gpt-4o-mini"]["per_call_usd"]
    full = out["gpt-4o"]["per_call_usd"]
    ratio = (full / mini) if mini > 0 else 0.0
    out["ratio_4o_vs_mini"] = {"per_call_multiplier": round(ratio, 2)}
    return out

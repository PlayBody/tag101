"""Live registration profit/loss probability for SN101 (or any netuid).

Estimates whether registering a new hotkey is likely to recoup the
registration fee before displacement, using current metagraph data.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from tag101.observability.pricing import compare_models_for_typical_tagging_call, estimate_call_cost_usd


BLOCKS_PER_DAY = 7200.0  # 12s blocks


@dataclass(frozen=True)
class RegistrationDecision:
    netuid: int
    block: int
    registration_fee_tao: float
    alpha_price_tao: float
    tempo_steps_per_day: float
    alpha_per_step_per_incentive: float
    tao_per_day_per_incentive: float
    immunity_days: float
    registrations_per_day: float
    active_miners: int
    slot_capacity: int
    target_incentive: float
    break_even_days: float
    break_even_days_with_opex: float
    survival_at_break_even: float
    profit_probability: float
    min_probability: float
    recommend_register: bool
    api_model: str
    api_cost_tao_per_day: float
    opex_tao_per_day: float
    median_miner_incentive: float
    p10_miner_incentive: float
    notes: list[str]


def _tao(value: Any) -> float:
    if hasattr(value, "tao"):
        return float(value.tao)
    return float(value)


def _reg_block(raw: Any) -> int:
    if isinstance(raw, (tuple, list, np.ndarray)):
        return int(raw[0])
    return int(raw)


def fetch_metagraph_stats(*, network: str, netuid: int) -> dict[str, Any]:
    import bittensor as bt

    sub = bt.Subtensor(network=network)
    mg = sub.metagraph(netuid=netuid, lite=False)
    hp = sub.get_subnet_hyperparameters(netuid)
    sn = sub.subnet(netuid=netuid)

    current_block = int(mg.block.item())
    reg_blocks = np.array([_reg_block(x) for x in mg.block_at_registration])
    age_days = (current_block - reg_blocks) / BLOCKS_PER_DAY

    inc = np.array([float(x) for x in mg.incentive])
    em = np.array([_tao(x) for x in mg.emission])
    vperm = np.array([bool(x) for x in mg.validator_permit])
    miner_mask = (inc > 0) & (~vperm)
    miner_inc = inc[miner_mask]

    if miner_inc.size == 0:
        raise RuntimeError(f"netuid {netuid}: no active miners with incentive > 0")

    k = float((em[miner_mask] / inc[miner_mask]).mean())
    tempo = int(hp.tempo)
    steps_day = BLOCKS_PER_DAY / float(tempo)
    price = _tao(sn.price)
    reg_fee = _tao(sub.recycle(netuid=netuid))
    immunity_days = float(hp.immunity_period) / BLOCKS_PER_DAY
    tao_per_day_per_incentive = k * steps_day * price

    regs_7d = int((age_days <= 7.0).sum())
    regs_per_day = regs_7d / 7.0

    return {
        "netuid": netuid,
        "block": current_block,
        "registration_fee_tao": reg_fee,
        "alpha_price_tao": price,
        "tempo_steps_per_day": steps_day,
        "alpha_per_step_per_incentive": k,
        "tao_per_day_per_incentive": tao_per_day_per_incentive,
        "immunity_days": immunity_days,
        "registrations_per_day": regs_per_day,
        "active_miners": int(miner_mask.sum()),
        "slot_capacity": int(mg.n),
        "miner_incentives": miner_inc,
        "age_days": age_days,
        "miner_mask": miner_mask,
        "inc": inc,
    }


def survival_rate(age_days: np.ndarray, miner_mask: np.ndarray, *, horizon_days: float) -> float:
    cohort = age_days >= horizon_days
    if not np.any(cohort):
        return 0.0
    return float((cohort & miner_mask).sum() / cohort.sum())


def incentive_recoup_rate(incentives: np.ndarray, *, threshold: float) -> float:
    if incentives.size == 0:
        return 0.0
    return float((incentives >= threshold).mean())


def estimate_profit_probability(
    stats: dict[str, Any],
    *,
    target_incentive: float | None = None,
    api_model: str = "gpt-4o-mini",
    validator_rounds_per_day: int = 96,
    server_usd_per_day: float = 0.43,
    tao_usd: float = 228.0,
) -> RegistrationDecision:
    miner_inc = stats["miner_incentives"]
    median_i = float(np.median(miner_inc))
    p10_i = float(np.percentile(miner_inc, 10))
    incentive = float(target_incentive if target_incentive is not None else median_i)

    reg_fee = float(stats["registration_fee_tao"])
    rate = float(stats["tao_per_day_per_incentive"])
    if incentive <= 0 or rate <= 0:
        raise ValueError("incentive and emission rate must be positive")

    per_call = estimate_call_cost_usd(model=api_model, prompt_tokens=500, completion_tokens=30)
    api_usd_day = per_call * validator_rounds_per_day
    api_tao_day = api_usd_day / tao_usd if tao_usd > 0 else 0.0
    server_tao_day = server_usd_per_day / tao_usd if tao_usd > 0 else 0.0
    opex_tao_day = api_tao_day + server_tao_day

    break_even_days = reg_fee / (incentive * rate)
    net_rate = incentive * rate - opex_tao_day
    break_even_days_with_opex = reg_fee / net_rate if net_rate > 0 else float("inf")

    horizon = break_even_days_with_opex if np.isfinite(break_even_days_with_opex) else break_even_days
    horizon = max(horizon, stats["immunity_days"])

    i_threshold = reg_fee / (horizon * rate) if horizon > 0 else float("inf")
    p_incentive = incentive_recoup_rate(miner_inc, threshold=i_threshold)
    p_survival = survival_rate(stats["age_days"], stats["miner_mask"], horizon_days=horizon)

    # Conservative joint estimate: must still be earning at horizon and earn enough.
    profit_probability = min(p_survival, p_incentive) if p_incentive > 0 else 0.0
    if incentive >= i_threshold:
        profit_probability = p_survival
    else:
        profit_probability = p_survival * (incentive / i_threshold)

    notes = [
        f"Break-even horizon ≈ {horizon:.2f} days at I={incentive:.6f}.",
        f"Immunity period is {stats['immunity_days']:.2f} days — recoup inside immunity is not expected.",
        f"Subnet turnover ≈ {stats['slot_capacity'] / max(stats['registrations_per_day'], 1e-9):.1f} days at current registration rate.",
    ]
    if stats["registrations_per_day"] > 20:
        notes.append(
            f"High churn: ~{stats['registrations_per_day']:.0f} registrations/day for {stats['slot_capacity']} slots."
        )

    return RegistrationDecision(
        netuid=int(stats["netuid"]),
        block=int(stats["block"]),
        registration_fee_tao=reg_fee,
        alpha_price_tao=float(stats["alpha_price_tao"]),
        tempo_steps_per_day=float(stats["tempo_steps_per_day"]),
        alpha_per_step_per_incentive=float(stats["alpha_per_step_per_incentive"]),
        tao_per_day_per_incentive=rate,
        immunity_days=float(stats["immunity_days"]),
        registrations_per_day=float(stats["registrations_per_day"]),
        active_miners=int(stats["active_miners"]),
        slot_capacity=int(stats["slot_capacity"]),
        target_incentive=incentive,
        break_even_days=break_even_days,
        break_even_days_with_opex=break_even_days_with_opex,
        survival_at_break_even=p_survival,
        profit_probability=profit_probability,
        min_probability=0.0,
        recommend_register=False,
        api_model=api_model,
        api_cost_tao_per_day=api_tao_day,
        opex_tao_per_day=opex_tao_day,
        median_miner_incentive=median_i,
        p10_miner_incentive=p10_i,
        notes=notes,
    )


def evaluate_registration(
    *,
    network: str = "finney",
    netuid: int = 101,
    min_probability: float = 0.70,
    target_incentive: float | None = None,
    api_model: str = "gpt-4o-mini",
    tao_usd: float = 228.0,
) -> RegistrationDecision:
    stats = fetch_metagraph_stats(network=network, netuid=netuid)
    decision = estimate_profit_probability(
        stats,
        target_incentive=target_incentive,
        api_model=api_model,
        tao_usd=tao_usd,
    )
    recommend = decision.profit_probability >= min_probability
    return RegistrationDecision(
        **{
            **asdict(decision),
            "min_probability": min_probability,
            "recommend_register": recommend,
        }
    )


def _print_decision(decision: RegistrationDecision) -> None:
    print(f"SN{decision.netuid} registration decision (block {decision.block})")
    print(f"  registration fee R        : {decision.registration_fee_tao:.4f} TAO")
    print(f"  target incentive I        : {decision.target_incentive:.6f} (median={decision.median_miner_incentive:.6f}, p10={decision.p10_miner_incentive:.6f})")
    print(f"  revenue rate μ            : {decision.tao_per_day_per_incentive:.6f} TAO/day per I")
    print(f"  break-even days D*        : {decision.break_even_days:.2f} (opex-adjusted {decision.break_even_days_with_opex:.2f})")
    print(f"  survival at D*            : {decision.survival_at_break_even * 100:.1f}%")
    print(f"  profit probability        : {decision.profit_probability * 100:.1f}%")
    print(f"  your threshold            : {decision.min_probability * 100:.0f}%")
    print(f"  registrations/day         : {decision.registrations_per_day:.1f}")
    print(f"  immunity period           : {decision.immunity_days:.2f} days")
    print(f"  opex (API+server)         : {decision.opex_tao_per_day:.6f} TAO/day ({decision.api_model})")
    print()
    if decision.recommend_register:
        print("  RECOMMENDATION: REGISTER (estimated profit probability meets threshold)")
    else:
        print("  RECOMMENDATION: DO NOT REGISTER YET (profit probability below threshold)")
    print()
    for note in decision.notes:
        print(f"  - {note}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Estimate profit probability before registering on Tag101/SN101."
    )
    parser.add_argument("--network", default="finney")
    parser.add_argument("--netuid", type=int, default=101)
    parser.add_argument(
        "--min-probability",
        type=float,
        default=0.70,
        help="Minimum profit probability required to recommend registration (default 0.70).",
    )
    parser.add_argument(
        "--target-incentive",
        type=float,
        default=None,
        help="Expected incentive after registering (default: current metagraph median miner incentive).",
    )
    parser.add_argument("--api-model", default="gpt-4o-mini")
    parser.add_argument("--tao-usd", type=float, default=228.0)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of human summary.")
    args = parser.parse_args(argv)

    decision = evaluate_registration(
        network=args.network,
        netuid=args.netuid,
        min_probability=args.min_probability,
        target_incentive=args.target_incentive,
        api_model=args.api_model,
        tao_usd=args.tao_usd,
    )

    if args.json:
        print(json.dumps(asdict(decision), indent=2))
    else:
        _print_decision(decision)
        compare = compare_models_for_typical_tagging_call()
        print("\nAPI cost reference (typical tagging load):")
        print(json.dumps(compare, indent=2))

    return 0 if decision.recommend_register else 1


if __name__ == "__main__":
    raise SystemExit(main())

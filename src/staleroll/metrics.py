from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence

from .types import Trajectory


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(q * (len(ordered) - 1))))
    return ordered[idx]


def wilson_interval(successes: int, total: int, z: float = 1.96) -> List[float]:
    if total == 0:
        return [0.0, 0.0]
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return [max(0.0, center - margin), min(1.0, center + margin)]


def summarize(trajectories: Sequence[Trajectory], ticks: int, workers: int) -> Dict[str, object]:
    total = len(trajectories)
    verified = sum(t.verified for t in trajectories)
    accepted = sum(t.accepted for t in trajectories)
    failures = sum(t.failed for t in trajectories)
    proxy_hacks = sum(t.proxy_pass_only for t in trajectories)
    stale_rejected = sum(1 for t in trajectories if not t.accepted and t.reject_reason in {"lag_limit", "kl_limit", "stale_for_sync"})
    lags = [float(t.lag_steps) for t in trajectories]
    kls = [float(t.kl_divergence) for t in trajectories]
    weights = [t.importance_weight for t in trajectories if t.accepted]
    verified_per_tick = verified / max(1, ticks)
    solved = max(verified, 1)
    quality_by_lag: Dict[str, Dict[str, float]] = {}
    buckets: Dict[int, List[Trajectory]] = defaultdict(list)
    for trajectory in trajectories:
        buckets[min(trajectory.lag_steps, 10)].append(trajectory)
    for lag, group in sorted(buckets.items()):
        quality_by_lag[str(lag)] = {
            "count": len(group),
            "verified_rate": _mean([float(t.verified) for t in group]),
            "accept_rate": _mean([float(t.accepted) for t in group]),
        }
    return {
        "logical_ticks": ticks,
        "trajectories": total,
        "verified": verified,
        "verified_pass_rate": verified / total if total else 0.0,
        "verified_pass_rate_ci95": wilson_interval(verified, total),
        "proxy_hacking_rate": proxy_hacks / total if total else 0.0,
        "accepted": accepted,
        "accept_rate": accepted / total if total else 0.0,
        "worker_failures": failures,
        "stale_rejected": stale_rejected,
        "rollouts_per_tick": total / max(1, ticks),
        "verified_rollouts_per_tick": verified_per_tick,
        "compute_seconds_per_solved_task": (ticks * max(1, workers)) / solved,
        "mean_lag": _mean(lags),
        "p95_lag": _quantile(lags, 0.95),
        "mean_kl": _mean(kls),
        "p95_kl": _quantile(kls, 0.95),
        "mean_importance_weight": _mean(weights),
        "quality_by_lag": quality_by_lag,
    }


def aggregate_summaries(summaries: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not summaries:
        return {}
    keys = [
        "verified_pass_rate", "proxy_hacking_rate", "accept_rate", "rollouts_per_tick",
        "verified_rollouts_per_tick", "compute_seconds_per_solved_task", "mean_lag", "p95_lag",
        "mean_kl", "p95_kl", "mean_importance_weight",
    ]
    result: Dict[str, object] = {"seeds": len(summaries)}
    for key in keys:
        values = [float(s[key]) for s in summaries]
        result[key] = _mean(values)
        result[key + "_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    result["verified_pass_rate_seed_values"] = [float(s["verified_pass_rate"]) for s in summaries]
    return result

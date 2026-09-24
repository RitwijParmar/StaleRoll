"""Optional integration seam for a real prime-rl / Verifiers run.

This module intentionally has no hard dependency on prime-rl. It provides a
serializable record and the same controller decision used by the CPU simulator,
so a future pinned integration can map its native rollout object into this
contract without changing the experiment protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Mapping

from .controller import StalenessController
from .types import PolicySnapshot, RunConfig, Task, Trajectory


@dataclass
class PrimeRolloutRecord:
    task_id: str
    worker_id: int
    behavior_version: int
    started_at: float
    finished_at: float
    action: int
    behavior_logprob: float
    reward: float
    verified: bool
    verifier_result: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def map_native_rollout(record: Mapping[str, Any], task: Task) -> Trajectory:
    """Map a native rollout dict into the StaleRoll decision schema.

    The adapter accepts common names (`policy_version`/`behavior_version`,
    `logprob`/`behavior_logprob`) but does not guess missing reward fields.
    """
    version = int(record.get("behavior_version", record.get("policy_version", 0)))
    if "behavior_prob" in record or "probability" in record:
        behavior_prob = float(record.get("behavior_prob", record.get("probability")))
    elif "behavior_logprob" in record or "logprob" in record:
        import math

        behavior_prob = math.exp(float(record.get("behavior_logprob", record.get("logprob"))))
    else:
        raise ValueError("native rollout must provide behavior_prob or behavior_logprob")
    action = int(record["action"])
    return Trajectory(
        task_id=str(record.get("task_id", task.task_id)),
        worker_id=int(record.get("worker_id", 0)),
        action=action,
        candidate=task.candidates[action],
        behavior_version=version,
        started_tick=int(record.get("started_tick", 0)),
        finished_tick=int(record.get("finished_tick", 0)),
        reward=float(record["reward"]),
        proxy_reward=float(record.get("proxy_reward", record["reward"])),
        verified=bool(record.get("verified", record["reward"] > 0)),
        proxy_pass_only=bool(record.get("proxy_pass_only", False)),
        failed=bool(record.get("failed", False)),
        failure_reason=str(record.get("failure_reason", "")),
        behavior_prob=behavior_prob,
    )


def controller_decision(
    trajectory: Trajectory,
    current_snapshot: PolicySnapshot,
    task: Task,
    config: RunConfig,
) -> Dict[str, Any]:
    """Return a JSON-ready accept/reweight decision for an external trainer."""
    import random

    controller = StalenessController(config, random.Random(config.seed))
    controller.assess(trajectory, current_snapshot, task)
    return {
        "accepted": trajectory.accepted,
        "reject_reason": trajectory.reject_reason,
        "lag_steps": trajectory.lag_steps,
        "kl_divergence": trajectory.kl_divergence,
        "importance_weight": trajectory.importance_weight,
        "update_weight": trajectory.update_weight,
    }

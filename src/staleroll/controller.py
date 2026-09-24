from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence

from .policy import ToyPolicy
from .types import PolicySnapshot, RunConfig, Task, Trajectory


class StalenessController:
    """Accept/reweight off-policy trajectories using age and policy divergence."""

    def __init__(self, config: RunConfig, rng: random.Random, policy=None):
        self.config = config
        self.rng = rng
        self.policy = policy

    def assess(self, trajectory: Trajectory, current: PolicySnapshot, task: Task) -> None:
        if trajectory.failed:
            trajectory.reject_reason = "worker_failure"
            return
        trajectory.lag_steps = max(0, current.version - trajectory.behavior_version)
        if self.policy is None:
            trajectory.current_prob = ToyPolicy.action_probability(current, task, trajectory.action)
        else:
            trajectory.current_prob = self.policy.action_probability(current, task, trajectory.action)
        behavior_prob = max(trajectory.behavior_prob, 1e-9)
        trajectory.kl_divergence = abs(math.log(behavior_prob / max(trajectory.current_prob, 1e-9)))
        ratio = trajectory.current_prob / behavior_prob

        mode = self.config.mode
        if mode == "sync":
            accepted = trajectory.lag_steps == 0
            weight = 1.0
            reason = "stale_for_sync" if not accepted else ""
        elif mode == "naive_async":
            accepted = True
            weight = min(self.config.max_importance_weight, max(0.05, ratio))
            reason = ""
        elif mode == "random_filter":
            accepted = self.rng.random() < 0.72
            weight = min(self.config.max_importance_weight, max(0.05, ratio))
            reason = "random_reject" if not accepted else ""
        else:
            lag_ok = trajectory.lag_steps <= self.config.max_lag
            kl_ok = trajectory.kl_divergence <= self.config.max_kl
            accepted = lag_ok and kl_ok
            weight = min(
                self.config.max_importance_weight,
                max(
                    0.05,
                    ratio
                    * math.exp(-self.config.lag_decay * trajectory.lag_steps)
                    * math.exp(-self.config.kl_decay * trajectory.kl_divergence),
                ),
            )
            reason = "lag_limit" if not lag_ok else ("kl_limit" if not kl_ok else "")

        trajectory.accepted = accepted
        trajectory.reject_reason = reason
        trajectory.importance_weight = weight
        trajectory.update_weight = weight if accepted else 0.0

    @staticmethod
    def assign_group_advantages(batch: Sequence[Trajectory]) -> None:
        groups: Dict[str, List[Trajectory]] = defaultdict(list)
        for trajectory in batch:
            if trajectory.accepted and not trajectory.failed:
                groups[trajectory.task_id].append(trajectory)
        for group in groups.values():
            mean = sum(t.reward for t in group) / len(group)
            variance = sum((t.reward - mean) ** 2 for t in group) / max(1, len(group))
            scale = math.sqrt(variance) or 1.0
            for trajectory in group:
                trajectory.group_advantage = (trajectory.reward - mean) / scale if len(group) > 1 else trajectory.reward - 0.5

from __future__ import annotations

import math
import random
from typing import Iterable, List, Sequence, Tuple

from .tasks import task_features
from .types import PolicySnapshot, Task, Trajectory


FEATURES = 7
ACTIONS = 5


def _softmax(logits: Sequence[float], temperature: float = 1.0) -> List[float]:
    scale = max(temperature, 1e-6)
    shifted = [v / scale for v in logits]
    m = max(shifted)
    exps = [math.exp(v - m) for v in shifted]
    total = sum(exps)
    return [v / total for v in exps]


class ToyPolicy:
    """A small contextual policy standing in for a LoRA policy head.

    It is deliberately transparent: the simulator tests the async controller,
    not the quality of a particular language model.
    """

    def __init__(self, seed: int):
        rng = random.Random(seed)
        self.version = 0
        self.weights = [[rng.uniform(-0.05, 0.05) for _ in range(FEATURES)] for _ in range(ACTIONS)]

    def snapshot(self) -> PolicySnapshot:
        return PolicySnapshot(self.version, [row[:] for row in self.weights])

    @staticmethod
    def probabilities(snapshot: PolicySnapshot, task: Task, temperature: float = 1.0) -> List[float]:
        logits = [sum(w * f for w, f in zip(row, task_features(task, i))) for i, row in enumerate(snapshot.weights)]
        return _softmax(logits, temperature)

    @classmethod
    def sample(cls, snapshot: PolicySnapshot, task: Task, rng: random.Random, temperature: float) -> Tuple[int, float]:
        probs = cls.probabilities(snapshot, task, temperature)
        draw = rng.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if draw <= cumulative:
                return i, p
        return len(probs) - 1, probs[-1]

    @classmethod
    def action_probability(cls, snapshot: PolicySnapshot, task: Task, action: int) -> float:
        return cls.probabilities(snapshot, task)[action]

    def update(self, trajectories: Sequence[Trajectory], tasks_by_id: dict, learning_rate: float) -> None:
        if not trajectories:
            return
        for trajectory in trajectories:
            if not trajectory.accepted or trajectory.failed:
                continue
            task = tasks_by_id[trajectory.task_id]
            probs = self.probabilities(self.snapshot(), task)
            advantage = trajectory.group_advantage
            scale = learning_rate * trajectory.update_weight * advantage
            features = task_features(task, trajectory.action)
            for action in range(ACTIONS):
                indicator = 1.0 if action == trajectory.action else 0.0
                for j, feature in enumerate(features):
                    self.weights[action][j] += scale * (indicator - probs[action]) * feature
        self.version += 1


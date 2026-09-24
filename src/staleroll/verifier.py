from __future__ import annotations

import random
from typing import Tuple

from .tasks import verify_candidate
from .types import Task


class Verifier:
    def __init__(self, noise: float = 0.0, seed: int = 0):
        self.noise = noise
        self.rng = random.Random(seed)

    def evaluate(self, task: Task, action: int) -> Tuple[float, float, bool, bool, str]:
        candidate = task.candidates[action]
        verified, proxy_pass, reason = verify_candidate(task, candidate, self.noise, self.rng)
        return (1.0 if verified else 0.0, 1.0 if proxy_pass else 0.0, verified, proxy_pass and not verified, reason)

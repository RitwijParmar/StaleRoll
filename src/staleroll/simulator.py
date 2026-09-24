from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

from .controller import StalenessController
from .policy import ToyPolicy
from .neural_policy import TransformerPolicy
from .lora_policy import LoRATransformerPolicy
from .vertex_policy import VertexGeminiPolicy
from .tasks import generate_tasks
from .types import PolicySnapshot, RunConfig, Task, Trajectory
from .verifier import Verifier


class AsyncSimulator:
    """Discrete-event simulator for async rollout workers and a trainer."""

    def __init__(self, config: RunConfig):
        self.config = config
        self.rng = random.Random(config.seed)
        self.tasks = generate_tasks(config.tasks, config.seed + 1001, config.task_domain)
        self.tasks_by_id = {task.task_id: task for task in self.tasks}
        self.timing = {"sample_seconds": 0.0, "verifier_seconds": 0.0, "update_seconds": 0.0}
        if config.policy_backend == "transformer":
            self.policy = TransformerPolicy(config.seed + 7, config, self.tasks)
        elif config.policy_backend == "lora":
            self.policy = LoRATransformerPolicy(config.seed + 7, config, self.tasks)
        elif config.policy_backend == "tabular":
            self.policy = ToyPolicy(config.seed + 7)
        elif config.policy_backend in {"vertex", "vertex_frozen"}:
            self.policy = VertexGeminiPolicy(config.seed + 7, config, self.tasks, trainable=not config.vertex_frozen)
        else:
            raise ValueError(f"unknown policy backend: {config.policy_backend}")
        self.verifier = Verifier(config.verifier_noise, config.seed + 9)
        self.controller = StalenessController(config, random.Random(config.seed + 11), self.policy)
        self.trajectories: List[Trajectory] = []
        self._task_cursor = 0
        self.elapsed_ticks = 0

    def _next_task(self) -> Task:
        # Consecutive workers receive the same task, enabling group-relative
        # advantages while preserving a changing stream of contexts.
        task = self.tasks[(self._task_cursor // self.config.workers) % len(self.tasks)]
        self._task_cursor += 1
        return task

    def _start(self, worker_id: int, tick: int, snapshot: PolicySnapshot) -> Tuple[int, dict]:
        task = self._next_task()
        sample_start = time.perf_counter()
        action, behavior_prob = self.policy.sample(snapshot, task, self.rng, self.config.temperature)
        self.timing["sample_seconds"] += time.perf_counter() - sample_start
        delay = 1 + self.rng.randint(0, self.config.max_delay)
        return tick + delay, {
            "task": task,
            "worker_id": worker_id,
            "action": action,
            "behavior_prob": behavior_prob,
            "snapshot": snapshot,
            "started_tick": tick,
        }

    def _finish(self, payload: dict, ready_tick: int) -> Trajectory:
        task: Task = payload["task"]
        action = payload["action"]
        verify_start = time.perf_counter()
        if self.rng.random() < self.config.failure_rate:
            reward, proxy, verified, proxy_only, reason = 0.0, 0.0, False, False, "worker_failed"
            failed = True
        else:
            reward, proxy, verified, proxy_only, reason = self.verifier.evaluate(task, action)
            failed = False
        self.timing["verifier_seconds"] += time.perf_counter() - verify_start
        snapshot: PolicySnapshot = payload["snapshot"]
        return Trajectory(
            task_id=task.task_id,
            worker_id=payload["worker_id"],
            action=action,
            candidate=task.candidates[action],
            behavior_version=snapshot.version,
            started_tick=payload["started_tick"],
            finished_tick=ready_tick,
            reward=reward,
            proxy_reward=proxy,
            verified=verified,
            proxy_pass_only=proxy_only,
            failed=failed,
            failure_reason=reason,
            behavior_prob=payload["behavior_prob"],
        )

    def _process_batch(self, ready: List[Trajectory]) -> None:
        if not ready:
            return
        batch = ready[: self.config.batch_size]
        del ready[: len(batch)]
        current = self.policy.snapshot()
        for trajectory in batch:
            self.controller.assess(trajectory, current, self.tasks_by_id[trajectory.task_id])
        self.controller.assign_group_advantages(batch)
        update_start = time.perf_counter()
        self.policy.update(batch, self.tasks_by_id, self.config.learning_rate)
        self.timing["update_seconds"] += time.perf_counter() - update_start
        self.trajectories.extend(batch)

    def run(self) -> List[Trajectory]:
        if self.config.mode == "sync":
            return self._run_sync()
        return self._run_async()

    def _run_sync(self) -> List[Trajectory]:
        # Barrier semantics: every worker samples from one snapshot; the
        # trainer updates only after all workers finish the round.
        rounds = max(1, self.config.ticks // 2)
        logical_tick = 0
        for _ in range(rounds):
            snapshot = self.policy.snapshot()
            round_batch: List[Trajectory] = []
            payloads: List[Tuple[int, dict]] = []
            for worker_id in range(self.config.workers):
                payloads.append(self._start(worker_id, logical_tick, snapshot))
            # A synchronous trainer waits for the slowest rollout. This is the
            # throughput cost that async workers are meant to avoid.
            barrier_tick = max(ready_tick for ready_tick, _ in payloads)
            for _, payload in payloads:
                round_batch.append(self._finish(payload, barrier_tick))
            logical_tick = barrier_tick
            current = self.policy.snapshot()
            for trajectory in round_batch:
                self.controller.assess(trajectory, current, self.tasks_by_id[trajectory.task_id])
            self.controller.assign_group_advantages(round_batch)
            update_start = time.perf_counter()
            self.policy.update(round_batch, self.tasks_by_id, self.config.learning_rate)
            self.timing["update_seconds"] += time.perf_counter() - update_start
            self.trajectories.extend(round_batch)
        self.elapsed_ticks = max(1, logical_tick)
        return self.trajectories

    def _run_async(self) -> List[Trajectory]:
        in_flight: Dict[int, Tuple[int, dict]] = {}
        ready: List[Trajectory] = []
        for tick in range(self.config.ticks):
            for worker_id, (ready_tick, payload) in list(in_flight.items()):
                if ready_tick <= tick:
                    ready.append(self._finish(payload, tick))
                    del in_flight[worker_id]
            while len(ready) >= self.config.batch_size:
                self._process_batch(ready)
            for worker_id in range(self.config.workers):
                if worker_id not in in_flight:
                    in_flight[worker_id] = self._start(worker_id, tick, self.policy.snapshot())
        # Drain delayed workers. This keeps the evaluation deterministic while
        # avoiding a hidden truncation of the last trajectories.
        drain_tick = self.config.ticks
        while in_flight or ready:
            for worker_id, (ready_tick, payload) in list(in_flight.items()):
                if ready_tick <= drain_tick:
                    ready.append(self._finish(payload, drain_tick))
                    del in_flight[worker_id]
            while len(ready) >= self.config.batch_size:
                self._process_batch(ready)
            drain_tick += 1
            if drain_tick > self.config.ticks + self.config.max_delay + 3:
                break
        if ready:
            self._process_batch(ready)
        self.elapsed_ticks = max(1, drain_tick)
        return self.trajectories


def run_once(config: RunConfig) -> Dict[str, object]:
    wall_start = time.perf_counter()
    simulator = AsyncSimulator(config)
    trajectories = simulator.run()
    finalize = getattr(simulator.policy, "finalize", None)
    if callable(finalize):
        finalize()
    timing = dict(simulator.timing)
    timing["wall_clock_seconds"] = time.perf_counter() - wall_start
    result = {
        "config": asdict(config),
        "trajectories": trajectories,
        "final_policy_version": simulator.policy.version,
        "elapsed_ticks": simulator.elapsed_ticks,
        "timing": timing,
    }
    usage = getattr(simulator.policy, "usage", None)
    if callable(usage):
        result["policy_usage"] = usage()
    return result

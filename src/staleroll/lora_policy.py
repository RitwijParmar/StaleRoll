from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch import Tensor, nn

from .neural_policy import _TransformerScorer
from .types import PolicySnapshot, RunConfig, Task, Trajectory


class _LoRAHead(nn.Module):
    """Low-rank residual head over a frozen task/candidate representation."""

    def __init__(self, dim: int, rank: int, alpha: float):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, 1, bias=False)
        self.scale = alpha / max(1, rank)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, features: Tensor) -> Tensor:
        return self.up(self.down(features)).squeeze(-1) * self.scale


class LoRATransformerPolicy:
    """Frozen Transformer plus a trainable task-conditioned LoRA adapter.

    The base encoder is warm-started once from the known training tasks and is
    frozen thereafter. Only the low-rank residual head is updated by verifier-
    weighted policy gradients. Checkpoints and oracle evaluation snapshots are
    written at configured update intervals.
    """

    def __init__(self, seed: int, config: RunConfig, tasks: Sequence[Task]):
        torch.manual_seed(seed)
        random.seed(seed)
        self.version = 0
        self.update_count = 0
        self.config = config
        self.tasks = list(tasks)
        self.started_at = time.perf_counter()
        self.base_model = _TransformerScorer(config.model_dim, config.model_layers, config.model_heads)
        self.view_base = _TransformerScorer(config.model_dim, config.model_layers, config.model_heads)
        self.adapter = _LoRAHead(config.model_dim, config.lora_rank, config.lora_alpha)
        self.view_adapter = _LoRAHead(config.model_dim, config.lora_rank, config.lora_alpha)
        self.base_optimizer = torch.optim.AdamW(
            self.base_model.parameters(), lr=config.learning_rate * 0.01, weight_decay=1e-4
        )
        self.optimizer = None
        self.cache: Dict[str, Tuple[Tensor, Tensor]] = {}
        self.probability_cache: Dict[Tuple[int, str, float], List[float]] = {}
        self.eval_history: List[Dict[str, float]] = []
        self.last_loss = 0.0
        self.checkpoint_dir = Path(config.checkpoint_dir) if config.checkpoint_dir else None
        if config.model_warmup_epochs > 0:
            self._warm_start(self.tasks, config.model_warmup_epochs)
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        self.base_model.eval()
        self.view_base.load_state_dict(self.base_model.state_dict())
        self.optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=config.learning_rate * 0.02, weight_decay=1e-4)
        if config.lora_warmup_epochs > 0:
            self._adapter_warm_start(self.tasks, config.lora_warmup_epochs)
        self._evaluate_and_checkpoint(force=True)

    @staticmethod
    def _text(task: Task, candidate: str) -> str:
        label = "candidate implementation" if task.domain == "code" else "candidate expression"
        return f"{label}: {candidate}\nTask: {task.prompt}\nAnswer with this candidate."

    @staticmethod
    def _encode(text: str, max_length: int = 224) -> Tuple[List[int], List[int]]:
        values = [min(127, ord(char)) for char in text][:max_length]
        return values, [1] * len(values)

    def _candidate_batch(self, task: Task) -> Tuple[Tensor, Tensor]:
        if task.task_id in self.cache:
            return self.cache[task.task_id]
        encoded = [self._encode(self._text(task, candidate)) for candidate in task.candidates]
        length = max(len(ids) for ids, _ in encoded)
        ids = torch.zeros((len(encoded), length), dtype=torch.long)
        attention = torch.zeros((len(encoded), length), dtype=torch.bool)
        for row, (values, mask) in enumerate(encoded):
            ids[row, : len(values)] = torch.tensor(values, dtype=torch.long)
            attention[row, : len(mask)] = True
        self.cache[task.task_id] = (ids, attention)
        return ids, attention

    def _features(self, model: _TransformerScorer, task: Task) -> Tensor:
        ids, attention = self._candidate_batch(task)
        return model.features(ids, attention)

    def _logits(self, base: _TransformerScorer, adapter: _LoRAHead, task: Task) -> Tensor:
        features = self._features(base, task)
        return base.score(features).squeeze(-1) + adapter(features)

    def snapshot(self) -> PolicySnapshot:
        state = {key: value.detach().cpu().clone() for key, value in self.adapter.state_dict().items()}
        return PolicySnapshot(self.version, [], {"adapter": state}, "lora-transformer")

    def probabilities(self, snapshot: PolicySnapshot, task: Task, temperature: float = 1.0) -> List[float]:
        cache_key = (snapshot.version, task.task_id, float(temperature))
        if cache_key in self.probability_cache:
            return self.probability_cache[cache_key][:]
        self.view_adapter.load_state_dict(snapshot.state["adapter"])
        self.view_base.eval()
        self.view_adapter.eval()
        with torch.no_grad():
            logits = self._logits(self.view_base, self.view_adapter, task) / max(temperature, 1e-5)
            probabilities = torch.softmax(logits, dim=0).tolist()
        self.probability_cache[cache_key] = probabilities
        return probabilities[:]

    def sample(self, snapshot: PolicySnapshot, task: Task, rng: random.Random, temperature: float) -> Tuple[int, float]:
        probabilities = self.probabilities(snapshot, task, temperature)
        draw = rng.random()
        cumulative = 0.0
        for index, probability in enumerate(probabilities):
            cumulative += probability
            if draw <= cumulative:
                return index, probability
        return len(probabilities) - 1, probabilities[-1]

    def action_probability(self, snapshot: PolicySnapshot, task: Task, action: int) -> float:
        return self.probabilities(snapshot, task, 1.0)[action]

    def _warm_start(self, tasks: Sequence[Task], epochs: int) -> None:
        self.base_model.train()
        usable = list(tasks)
        for _ in range(epochs):
            random.shuffle(usable)
            losses: List[Tensor] = []
            for task in usable:
                logits = self._logits(self.base_model, _LoRAHead(self.config.model_dim, self.config.lora_rank, self.config.lora_alpha), task)
                # The temporary adapter is zero-initialized; warm-start the
                # base scorer only, keeping this stage independent of LoRA.
                target = torch.tensor([task.correct_index], dtype=torch.long)
                losses.append(nn.functional.cross_entropy(logits.unsqueeze(0), target))
                if len(losses) >= 16:
                    self._base_step(torch.stack(losses).mean())
                    losses = []
            if losses:
                self._base_step(torch.stack(losses).mean())
        self.base_model.eval()

    def _base_step(self, loss: Tensor) -> None:
        self.base_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.base_model.parameters(), 1.0)
        self.base_optimizer.step()

    def _adapter_step(self, loss: Tensor) -> None:
        assert self.optimizer is not None
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.adapter.parameters(), 1.0)
        self.optimizer.step()
        self.last_loss = float(loss.detach().cpu())

    def _adapter_warm_start(self, tasks: Sequence[Task], epochs: int) -> None:
        """Fit only the low-rank head to the known certified task labels."""
        self.adapter.train()
        usable = list(tasks)
        for _ in range(epochs):
            random.shuffle(usable)
            losses: List[Tensor] = []
            for task in usable:
                logits = self._logits(self.base_model, self.adapter, task)
                target = torch.tensor([task.correct_index], dtype=torch.long)
                losses.append(nn.functional.cross_entropy(logits.unsqueeze(0), target))
                if len(losses) >= 16:
                    self._adapter_step(torch.stack(losses).mean())
                    losses = []
            if losses:
                self._adapter_step(torch.stack(losses).mean())
        self.adapter.eval()

    def update(self, trajectories: Sequence[Trajectory], tasks_by_id: Dict[str, Task], learning_rate: float) -> None:
        usable = [t for t in trajectories if t.accepted and not t.failed]
        if usable:
            losses: List[Tensor] = []
            self.adapter.train()
            for trajectory in usable:
                logits = self._logits(self.base_model, self.adapter, tasks_by_id[trajectory.task_id])
                log_prob = torch.log_softmax(logits, dim=0)[trajectory.action]
                coefficient = float(trajectory.group_advantage * trajectory.update_weight)
                losses.append(-coefficient * log_prob)
            regularizer = 1e-4 * sum((parameter.square().sum() for parameter in self.adapter.parameters()))
            self._adapter_step(torch.stack(losses).mean() + regularizer)
            self.adapter.eval()
        self.version += 1
        self.update_count += 1
        if self.update_count % max(1, self.config.eval_interval) == 0:
            self._evaluate_and_checkpoint()

    def _evaluate(self) -> Dict[str, float]:
        snapshot = self.snapshot()
        correct = 0
        entropy = 0.0
        for task in self.tasks:
            probabilities = self.probabilities(snapshot, task, 1.0)
            correct += int(max(range(len(probabilities)), key=probabilities.__getitem__) == task.correct_index)
            entropy -= sum(p * torch.log(torch.tensor(max(p, 1e-8))).item() for p in probabilities)
        result = {
            "update": float(self.update_count),
            "oracle_accuracy": correct / max(1, len(self.tasks)),
            "mean_entropy": entropy / max(1, len(self.tasks)),
            "loss": self.last_loss,
            "wall_seconds": time.perf_counter() - self.started_at,
        }
        self.eval_history.append(result)
        return result

    def _evaluate_and_checkpoint(self, force: bool = False) -> None:
        evaluation = self._evaluate()
        if self.checkpoint_dir is None:
            return
        if not force and self.update_count % max(1, self.config.checkpoint_interval) != 0:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = self.checkpoint_dir / f"checkpoint-update-{self.update_count:05d}.pt"
        torch.save(
            {
                "backend": "lora-transformer",
                "version": self.version,
                "update": self.update_count,
                "base_model": self.base_model.state_dict(),
                "adapter": self.adapter.state_dict(),
                "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
                "evaluation": evaluation,
            },
            checkpoint,
        )
        with (self.checkpoint_dir / "evaluation.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(evaluation) + "\n")

    def finalize(self) -> None:
        """Flush a final evaluation and checkpoint after the simulator drains.

        Async runs can finish before the next configured interval.  A forced
        final snapshot makes the saved training history complete and gives
        every run a directly comparable end-of-training measurement.
        """
        if not self.eval_history or self.eval_history[-1]["update"] != float(self.update_count):
            self._evaluate_and_checkpoint(force=True)
        elif self.checkpoint_dir is not None:
            checkpoint = self.checkpoint_dir / f"checkpoint-update-{self.update_count:05d}.pt"
            if not checkpoint.exists():
                self._evaluate_and_checkpoint(force=True)

    def usage(self) -> Dict[str, object]:
        return {
            "lora_rank": self.config.lora_rank,
            "lora_alpha": self.config.lora_alpha,
            "lora_trainable_parameters": sum(parameter.numel() for parameter in self.adapter.parameters()),
            "lora_updates": self.update_count,
            "lora_eval_history": self.eval_history,
            "lora_checkpoint_dir": str(self.checkpoint_dir) if self.checkpoint_dir else "",
        }

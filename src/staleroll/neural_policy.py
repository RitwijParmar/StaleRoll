from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence, Tuple

import torch
from torch import Tensor, nn

from .types import PolicySnapshot, RunConfig, Task, Trajectory


class _TransformerScorer(nn.Module):
    """Compact character-token Transformer used for the local experiment."""

    def __init__(self, dim: int, layers: int, heads: int, max_length: int = 224):
        super().__init__()
        self.token = nn.Embedding(128, dim)
        self.position = nn.Embedding(max_length, dim)
        block = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.norm = nn.LayerNorm(dim)
        self.score = nn.Linear(dim, 1)
        self.max_length = max_length

    def features(self, input_ids: Tensor, attention: Tensor) -> Tensor:
        length = input_ids.shape[1]
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0)
        hidden = self.token(input_ids) + self.position(positions)
        hidden = self.encoder(hidden, src_key_padding_mask=~attention.bool())
        mask = attention.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.norm(pooled)

    def forward(self, input_ids: Tensor, attention: Tensor) -> Tensor:
        return self.score(self.features(input_ids, attention)).squeeze(-1)


class TransformerPolicy:
    """A real neural policy for the async controller experiment.

    The policy scores each candidate with a character-token Transformer. It is
    warm-started with a small supervised objective, then updated with a
    verifier-weighted policy-gradient loss. Only PyTorch is required; no model
    download or cloud service is involved.
    """

    def __init__(self, seed: int, config: RunConfig, tasks: Sequence[Task]):
        torch.manual_seed(seed)
        random.seed(seed)
        self.version = 0
        self.config = config
        self.model = _TransformerScorer(config.model_dim, config.model_layers, config.model_heads)
        self.view_model = _TransformerScorer(config.model_dim, config.model_layers, config.model_heads)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate * 0.01, weight_decay=1e-4)
        self.cache: Dict[str, Tuple[Tensor, Tensor]] = {}
        self.probability_cache: Dict[Tuple[int, str, float], List[float]] = {}
        if config.model_warmup_epochs > 0:
            self._warm_start(tasks, config.model_warmup_epochs)

    @staticmethod
    def _text(task: Task, candidate: str) -> str:
        # Put the candidate first so long code-repair prompts cannot truncate
        # the implementation out of the fixed context window.
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

    def _logits(self, model: nn.Module, task: Task) -> Tensor:
        ids, attention = self._candidate_batch(task)
        return model(ids, attention)

    def snapshot(self) -> PolicySnapshot:
        state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
        return PolicySnapshot(self.version, [], state, "transformer")

    def probabilities(self, snapshot: PolicySnapshot, task: Task, temperature: float = 1.0) -> List[float]:
        cache_key = (snapshot.version, task.task_id, float(temperature))
        if cache_key in self.probability_cache:
            return self.probability_cache[cache_key][:]
        self.view_model.load_state_dict(snapshot.state)
        self.view_model.eval()
        with torch.no_grad():
            logits = self._logits(self.view_model, task) / max(temperature, 1e-5)
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
        self.model.train()
        usable = list(tasks)
        for _ in range(epochs):
            random.shuffle(usable)
            losses: List[Tensor] = []
            for task in usable:
                logits = self._logits(self.model, task).unsqueeze(0)
                target = torch.tensor([task.correct_index], dtype=torch.long)
                losses.append(nn.functional.cross_entropy(logits, target))
                if len(losses) >= 16:
                    self._step(torch.stack(losses).mean())
                    losses = []
            if losses:
                self._step(torch.stack(losses).mean())
        self.model.eval()

    def _step(self, loss: Tensor) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

    def update(self, trajectories: Sequence[Trajectory], tasks_by_id: Dict[str, Task], learning_rate: float) -> None:
        usable = [t for t in trajectories if t.accepted and not t.failed]
        if usable:
            losses: List[Tensor] = []
            self.model.train()
            for trajectory in usable:
                logits = self._logits(self.model, tasks_by_id[trajectory.task_id])
                log_prob = torch.log_softmax(logits, dim=0)[trajectory.action]
                coefficient = float(trajectory.group_advantage * trajectory.update_weight)
                losses.append(-coefficient * log_prob)
            self._step(torch.stack(losses).mean())
            self.model.eval()
        self.version += 1

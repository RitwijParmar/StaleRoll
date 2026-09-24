from __future__ import annotations

import json
import math
import random
import subprocess
from typing import Dict, List, Sequence, Tuple

import requests

from .types import PolicySnapshot, RunConfig, Task, Trajectory


class VertexGeminiPolicy:
    """Gemini-backed candidate policy with an explicit local spend guard.

    Gemini supplies the base distribution over candidates. A small online
    policy-gradient bias is updated from verifier rewards, so the RL loop can
    change the action distribution without pretending to fine-tune Gemini.
    Every request is bounded by both a request count and a conservative local
    estimated-cost ceiling.
    """

    def __init__(self, seed: int, config: RunConfig, tasks: Sequence[Task], trainable: bool = True):
        self.version = 0
        self.config = config
        self.rng = random.Random(seed)
        self.bias = [0.0] * 5
        self.base_cache: Dict[str, List[float]] = {}
        self.probability_cache: Dict[Tuple[int, str, float], List[float]] = {}
        self.requests_used = 0
        self.estimated_cost_usd = 0.0
        self.input_tokens = 0
        self.output_tokens = 0
        self.model_version = ""
        self._cached_access_token = None
        self._session = requests.Session()
        self.project = config.vertex_project
        self.location = config.vertex_location
        self.model_id = config.vertex_model
        self.account = config.vertex_account
        self.max_requests = config.vertex_max_requests
        self.max_budget_usd = config.vertex_budget_usd
        self.cost_per_request_estimate = config.vertex_cost_per_request_estimate
        self.trainable = trainable

    def snapshot(self) -> PolicySnapshot:
        return PolicySnapshot(
            self.version,
            [],
            {"bias": self.bias[:]},
            "vertex-gemini",
        )

    def _access_token(self) -> str:
        if self._cached_access_token:
            return self._cached_access_token
        if not self.account:
            raise RuntimeError("Vertex runs require an explicit --vertex-account; no cloud account is selected")
        command = ["gcloud", "auth", "print-access-token"]
        command.append(f"--account={self.account}")
        self._cached_access_token = subprocess.check_output(command, text=True).strip()
        return self._cached_access_token

    def _base_distribution(self, task: Task) -> List[float]:
        if task.task_id in self.base_cache:
            return self.base_cache[task.task_id][:]
        if self.requests_used >= self.max_requests:
            raise RuntimeError(f"Vertex request guard reached {self.max_requests} requests")
        next_estimate = self.estimated_cost_usd + self.cost_per_request_estimate
        if next_estimate > self.max_budget_usd:
            raise RuntimeError(f"Vertex estimated-cost guard reached ${self.max_budget_usd:.2f}")
        candidates = "\n\n".join(f"CANDIDATE {index}:\n{text}" for index, text in enumerate(task.candidates))
        if task.domain == "code":
            instruction = (
                "Select the complete Python implementation that is most likely to pass both the visible tests "
                "and unseen tests. Return JSON only with a chosen_index and five probabilities that sum to 1."
            )
        else:
            instruction = (
                "Select the expression that matches the hidden arithmetic rule. Use the visible examples, and "
                "return JSON only with a chosen_index and five probabilities that sum to 1."
            )
        prompt = (
            f"{instruction} Do not add markdown.\n\n{task.prompt}\nCandidates:\n{candidates}\n"
            '{"chosen_index": 0, "probabilities": [0.2, 0.2, 0.2, 0.2, 0.2]}'
        )
        endpoint = (
            f"https://{self.location}-aiplatform.googleapis.com/v1/projects/"
            f"{self.project}/locations/{self.location}/publishers/google/models/"
            f"{self.model_id}:generateContent"
        )
        # Count the request before opening the socket so a timeout also consumes
        # one guard unit. This prevents an unreachable endpoint from bypassing
        # the local ceiling through repeated retries.
        self.requests_used += 1
        self.estimated_cost_usd = next_estimate
        response = self._session.post(
            endpoint,
            headers={"Authorization": f"Bearer {self._access_token()}", "Content-Type": "application/json"},
            json={
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.0,
                    "maxOutputTokens": 128,
                    "responseMimeType": "application/json",
                },
            },
            timeout=(10, 90),
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Vertex request failed ({response.status_code}): {response.text[:500]}")
        payload = response.json()
        self.model_version = str(payload.get("modelVersion", self.model_version))
        usage = payload.get("usageMetadata", {})
        self.input_tokens += int(usage.get("promptTokenCount", 0) or 0)
        self.output_tokens += int(usage.get("candidatesTokenCount", 0) or 0)
        text = payload["candidates"][0]["content"]["parts"][0]["text"]
        try:
            start, end = text.find("{"), text.rfind("}")
            parsed = json.loads(text[start : end + 1])
            probabilities = [float(value) for value in parsed["probabilities"]]
            if len(probabilities) != 5 or any(not math.isfinite(value) or value < 0 for value in probabilities):
                raise ValueError("invalid probability vector")
            total = sum(probabilities)
            if total <= 0:
                raise ValueError("empty probability vector")
            probabilities = [max(1e-4, value / total) for value in probabilities]
            normalizer = sum(probabilities)
            probabilities = [value / normalizer for value in probabilities]
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Gemini returned invalid candidate JSON: {text!r}") from exc
        self.base_cache[task.task_id] = probabilities
        return probabilities[:]

    def probabilities(self, snapshot: PolicySnapshot, task: Task, temperature: float = 1.0) -> List[float]:
        key = (snapshot.version, task.task_id, float(temperature))
        if key in self.probability_cache:
            return self.probability_cache[key][:]
        base = self._base_distribution(task)
        bias = snapshot.state["bias"]
        logits = [(math.log(max(value, 1e-8)) + bias[index]) / max(temperature, 1e-5) for index, value in enumerate(base)]
        maximum = max(logits)
        exponentials = [math.exp(value - maximum) for value in logits]
        total = sum(exponentials)
        probabilities = [value / total for value in exponentials]
        self.probability_cache[key] = probabilities
        return probabilities[:]

    def sample(self, snapshot: PolicySnapshot, task: Task, rng: random.Random, temperature: float) -> Tuple[int, float]:
        probabilities = self.probabilities(snapshot, task, temperature)
        draw = rng.random()
        cumulative = 0.0
        for index, probability in enumerate(probabilities):
            cumulative += probability
            if draw <= cumulative:
                return index, probability
        return 4, probabilities[4]

    def action_probability(self, snapshot: PolicySnapshot, task: Task, action: int) -> float:
        return self.probabilities(snapshot, task, 1.0)[action]

    def update(self, trajectories: Sequence[Trajectory], tasks_by_id: Dict[str, Task], learning_rate: float) -> None:
        if not self.trainable:
            self.version += 1
            return
        for trajectory in trajectories:
            if not trajectory.accepted or trajectory.failed:
                continue
            task = tasks_by_id[trajectory.task_id]
            probabilities = self.probabilities(self.snapshot(), task, 1.0)
            coefficient = learning_rate * trajectory.update_weight * trajectory.group_advantage
            for index, probability in enumerate(probabilities):
                self.bias[index] += coefficient * ((1.0 if index == trajectory.action else 0.0) - probability)
        self.version += 1

    def usage(self) -> Dict[str, object]:
        return {
            "vertex_project": self.project,
            "vertex_location": self.location,
            "vertex_model": self.model_id,
            "vertex_model_version": self.model_version,
            "vertex_requests": self.requests_used,
            "vertex_estimated_cost_usd": self.estimated_cost_usd,
            "vertex_budget_guard_usd": self.max_budget_usd,
            "vertex_trainable": self.trainable,
            "vertex_input_tokens": self.input_tokens,
            "vertex_output_tokens": self.output_tokens,
        }

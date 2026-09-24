from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Task:
    task_id: str
    prompt: str
    candidates: List[str]
    visible_cases: List[tuple]
    hidden_cases: List[tuple]
    target_description: str
    correct_index: int = 0
    domain: str = "arithmetic"
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class Trajectory:
    task_id: str
    worker_id: int
    action: int
    candidate: str
    behavior_version: int
    started_tick: int
    finished_tick: int
    reward: float
    proxy_reward: float
    verified: bool
    proxy_pass_only: bool
    failed: bool
    failure_reason: str = ""
    behavior_prob: float = 0.0
    current_prob: float = 0.0
    lag_steps: int = 0
    kl_divergence: float = 0.0
    importance_weight: float = 1.0
    accepted: bool = False
    reject_reason: str = ""
    update_weight: float = 0.0
    group_advantage: float = 0.0
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class PolicySnapshot:
    version: int
    weights: List[List[float]]
    state: object = None
    backend: str = "tabular"


@dataclass
class RunConfig:
    seed: int = 0
    mode: str = "stale_filter"
    tasks: int = 80
    workers: int = 4
    ticks: int = 120
    batch_size: int = 4
    learning_rate: float = 0.12
    max_lag: int = 8
    max_kl: float = 1.25
    lag_decay: float = 0.12
    kl_decay: float = 0.40
    max_importance_weight: float = 2.0
    failure_rate: float = 0.04
    verifier_noise: float = 0.0
    max_delay: int = 8
    temperature: float = 0.9
    task_domain: str = "arithmetic"
    policy_backend: str = "transformer"
    model_dim: int = 128
    model_layers: int = 2
    model_heads: int = 4
    model_warmup_epochs: int = 2
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_warmup_epochs: int = 1
    checkpoint_dir: str = ""
    checkpoint_interval: int = 10
    eval_interval: int = 10
    # Vertex Gemini backend. These defaults point at the already-open project
    # used for the bounded smoke run; the client still enforces request and
    # estimated-cost guards before making a call.
    vertex_project: str = "gen-lang-client-0576163520"
    vertex_location: str = "us-central1"
    vertex_model: str = "gemini-2.5-flash-lite"
    # Leave account selection empty so a cloud run cannot silently bind to a
    # personal account. Pass --vertex-account explicitly only after choosing a
    # different, authorized account.
    vertex_account: str = ""
    vertex_max_requests: int = 120
    vertex_budget_usd: float = 180.0
    vertex_cost_per_request_estimate: float = 0.01
    vertex_frozen: bool = False

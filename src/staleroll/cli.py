from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from .metrics import aggregate_summaries, summarize
from .reporting import build_dashboard, build_report, write_json, write_jsonl
from .simulator import run_once
from .types import RunConfig


MODES = ["sync", "naive_async", "stale_filter", "random_filter"]


def _run(args: argparse.Namespace) -> int:
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    comparison: Dict[str, object] = {
        "experiment": "staleroll",
        "policy_backend": args.policy,
        "task_domain": args.task_domain,
        "modes": {},
        "runs": [],
    }
    for mode in MODES:
        seed_summaries: List[Dict[str, object]] = []
        for seed in range(args.seeds):
            config = RunConfig(
                seed=seed,
                mode=mode,
                tasks=args.tasks,
                workers=args.workers,
                ticks=args.ticks,
                batch_size=args.batch_size,
                failure_rate=args.failure_rate,
                verifier_noise=args.verifier_noise,
                max_delay=args.max_delay,
                max_lag=args.max_lag,
                max_kl=args.max_kl,
                lag_decay=args.lag_decay,
                kl_decay=args.kl_decay,
                learning_rate=args.learning_rate,
                temperature=args.temperature,
                policy_backend=args.policy,
                model_dim=args.model_dim,
                model_layers=args.model_layers,
                model_heads=args.model_heads,
                model_warmup_epochs=args.model_warmup_epochs,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_warmup_epochs=args.lora_warmup_epochs,
                checkpoint_dir=str(root / "checkpoints" / mode / f"seed-{seed:02d}") if args.policy == "lora" else "",
                checkpoint_interval=args.checkpoint_interval,
                eval_interval=args.eval_interval,
                task_domain=args.task_domain,
                vertex_project=args.vertex_project,
                vertex_location=args.vertex_location,
                vertex_model=args.vertex_model,
                vertex_account=args.vertex_account,
                vertex_max_requests=args.vertex_max_requests,
                vertex_budget_usd=args.vertex_budget_usd,
                vertex_cost_per_request_estimate=args.vertex_cost_per_request,
                vertex_frozen=args.policy == "vertex_frozen",
            )
            result = run_once(config)
            if "run_config" not in comparison:
                comparison["run_config"] = asdict(config)
            summary = summarize(result["trajectories"], result["elapsed_ticks"], args.workers)
            seed_summaries.append(summary)
            run_dir = root / "runs" / mode / f"seed-{seed:02d}"
            write_json(run_dir / "config.json", asdict(config))
            write_json(run_dir / "summary.json", summary)
            write_json(run_dir / "profile.json", result["timing"])
            if "policy_usage" in result:
                write_json(run_dir / "policy_usage.json", result["policy_usage"])
            write_jsonl(run_dir / "trajectories.jsonl", [asdict(t) for t in result["trajectories"]])
            run_record = {"mode": mode, "seed": seed, "summary": summary}
            if "policy_usage" in result:
                run_record["policy_usage"] = result["policy_usage"]
            run_record["timing"] = result["timing"]
            comparison["runs"].append(run_record)
        comparison["modes"][mode] = aggregate_summaries(seed_summaries)
    sync = comparison["modes"]["sync"]
    for mode, values in comparison["modes"].items():
        values["quality_retention_vs_sync"] = values["verified_pass_rate"] / max(sync["verified_pass_rate"], 1e-9)
        values["verified_throughput_multiplier_vs_sync"] = values["verified_rollouts_per_tick"] / max(sync["verified_rollouts_per_tick"], 1e-9)
    write_json(root / "comparison.json", comparison)
    build_dashboard(root / "dashboard.html", comparison)
    build_report(root / "report.md", comparison)
    print(json.dumps({"output": str(root), "modes": comparison["modes"]}, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="StaleRoll CPU-only async RL simulator")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run all baselines and produce a report")
    run.add_argument("--seeds", type=int, default=3)
    run.add_argument("--tasks", type=int, default=80)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--ticks", type=int, default=120)
    run.add_argument("--batch-size", type=int, default=4)
    run.add_argument("--failure-rate", type=float, default=0.04)
    run.add_argument("--verifier-noise", type=float, default=0.0)
    run.add_argument("--max-delay", type=int, default=8)
    run.add_argument("--max-lag", type=int, default=8)
    run.add_argument("--max-kl", type=float, default=1.25)
    run.add_argument("--lag-decay", type=float, default=0.12)
    run.add_argument("--kl-decay", type=float, default=0.40)
    run.add_argument("--learning-rate", type=float, default=0.12)
    run.add_argument("--temperature", type=float, default=0.9)
    run.add_argument("--output", default="artifacts/latest")
    run.add_argument("--policy", choices=["transformer", "lora", "tabular", "vertex", "vertex_frozen"], default="transformer")
    run.add_argument("--task-domain", choices=["arithmetic", "code"], default="arithmetic")
    run.add_argument("--model-dim", type=int, default=128)
    run.add_argument("--model-layers", type=int, default=2)
    run.add_argument("--model-heads", type=int, default=4)
    run.add_argument("--model-warmup-epochs", type=int, default=1)
    run.add_argument("--lora-rank", type=int, default=8)
    run.add_argument("--lora-alpha", type=float, default=16.0)
    run.add_argument("--lora-warmup-epochs", type=int, default=1)
    run.add_argument("--checkpoint-interval", type=int, default=10)
    run.add_argument("--eval-interval", type=int, default=10)
    run.add_argument("--vertex-project", default="gen-lang-client-0576163520")
    run.add_argument("--vertex-location", default="us-central1")
    run.add_argument("--vertex-model", default="gemini-2.5-flash-lite")
    run.add_argument("--vertex-account", default="ritwij.aryan.parmar@gmail.com")
    run.add_argument("--vertex-max-requests", type=int, default=120)
    run.add_argument("--vertex-budget-usd", type=float, default=180.0)
    run.add_argument("--vertex-cost-per-request", type=float, default=0.01)
    run.set_defaults(func=_run)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

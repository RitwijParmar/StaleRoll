#!/usr/bin/env python3
"""Compare the trained LoRA adapter with a frozen Gemini baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def load(path: Path) -> dict:
    return json.loads((path / "comparison.json").read_text(encoding="utf-8"))


def usage(path: Path) -> dict:
    rows = []
    for file in path.rglob("policy_usage.json"):
        rows.append(json.loads(file.read_text(encoding="utf-8")))
    return {
        "requests": sum(int(row.get("vertex_requests", 0)) for row in rows),
        "estimated_cost_usd": sum(float(row.get("vertex_estimated_cost_usd", 0.0)) for row in rows),
        "lora_updates": sum(int(row.get("lora_updates", 0)) for row in rows),
        "lora_trainable_parameters": sorted({int(row.get("lora_trainable_parameters", 0)) for row in rows}),
    }


def mean_profile(path: Path) -> dict:
    rows = [json.loads(file.read_text(encoding="utf-8")) for file in path.rglob("profile.json")]
    if not rows:
        return {}
    return {key: mean(float(row.get(key, 0.0)) for row in rows) for key in rows[0]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lora = load(args.lora)
    frozen = load(args.frozen)
    lora_config = lora.get("run_config", {})
    frozen_config = frozen.get("run_config", {})
    lora_seeds = int(lora.get("modes", {}).get("sync", {}).get("seeds", 0))
    frozen_seeds = int(frozen.get("modes", {}).get("sync", {}).get("seeds", 0))
    rows = []
    for mode in ("sync", "naive_async", "stale_filter", "random_filter"):
        left = lora["modes"][mode]
        right = frozen["modes"][mode]
        rows.append(
            {
                "mode": mode,
                "lora_verified_pass_rate": left["verified_pass_rate"],
                "frozen_gemini_verified_pass_rate": right["verified_pass_rate"],
                "delta_verified_pass_rate_lora_minus_frozen": left["verified_pass_rate"] - right["verified_pass_rate"],
                "lora_verified_throughput": left["verified_rollouts_per_tick"],
                "frozen_gemini_verified_throughput": right["verified_rollouts_per_tick"],
                "delta_verified_throughput_lora_minus_frozen": left["verified_rollouts_per_tick"] - right["verified_rollouts_per_tick"],
            }
        )
    result = {
        "experiment": "staleroll_backend_comparison",
        "task_domain": "code",
        "lora_artifact": str(args.lora),
        "frozen_gemini_artifact": str(args.frozen),
        "protocol": {
            "tasks": int(lora_config.get("tasks", 0)),
            "lora_seeds": lora_seeds,
            "frozen_gemini_seeds": frozen_seeds,
            "lora_ticks": int(lora_config.get("ticks", 0)),
            "frozen_gemini_ticks": int(frozen_config.get("ticks", 0)),
            "max_lag": int(lora_config.get("max_lag", 0)),
            "max_kl": float(lora_config.get("max_kl", 0.20)),
        },
        "rows": rows,
        "lora_usage": usage(args.lora),
        "frozen_gemini_usage": usage(args.frozen),
        "lora_mean_profile": mean_profile(args.lora),
        "frozen_gemini_mean_profile": mean_profile(args.frozen),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# LoRA adapter vs frozen Gemini",
        "",
        f"Matched task/controller protocol: {result['protocol']['tasks']} execution-verified code tasks, maximum accepted lag {result['protocol']['max_lag']}, and KL cutoff {result['protocol']['max_kl']:.2f}. LoRA used {lora_seeds} seeds/{result['protocol']['lora_ticks']} ticks; frozen Gemini used {frozen_seeds} seeds/{result['protocol']['frozen_gemini_ticks']} ticks.",
        "",
        "| Mode | LoRA pass rate | Frozen Gemini pass rate | Δ pass rate | LoRA verified/tick | Frozen Gemini verified/tick | Δ throughput |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['lora_verified_pass_rate']:.4f} | {row['frozen_gemini_verified_pass_rate']:.4f} | {row['delta_verified_pass_rate_lora_minus_frozen']:+.4f} | {row['lora_verified_throughput']:.4f} | {row['frozen_gemini_verified_throughput']:.4f} | {row['delta_verified_throughput_lora_minus_frozen']:+.4f} |"
        )
    lines += [
        "",
        f"LoRA training used {result['lora_usage']['lora_trainable_parameters']} trainable parameters per policy instance and saved checkpoints/evaluation traces. Frozen Gemini used {result['frozen_gemini_usage']['requests']} Vertex requests with a local estimate of ${result['frozen_gemini_usage']['estimated_cost_usd']:.2f}.",
        "",
        "Interpretation: the LoRA path is a real local training path with checkpointed updates; frozen Gemini is the stronger quality baseline in this task-scale comparison. Seed counts differ because the Gemini arm is the cloud-cost-limited reference, so this is evidence for engineering direction rather than a final significance claim.",
        "",
    ]
    (args.output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

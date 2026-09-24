from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def build_dashboard(path: Path, comparison: Dict[str, object]) -> None:
    rows = comparison.get("modes", {})
    payload = json.dumps(rows, sort_keys=True)
    backend = comparison.get("policy_backend", "transformer")
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>StaleRoll dashboard</title>
<style>body{{font:15px system-ui;margin:32px;color:#17202a}} table{{border-collapse:collapse;margin-top:16px}} th,td{{border:1px solid #ccd;padding:8px 12px;text-align:right}} th:first-child,td:first-child{{text-align:left}} .bar{{height:14px;background:#3478c9}}</style></head>
<body><h1>StaleRoll experiment dashboard</h1>
<p>Discrete-event simulation using the <strong>{backend}</strong> policy backend. Values are means across requested seeds.</p>
<div id="app"></div>
<script>
const data={payload};
const metrics=[['verified_pass_rate','Verified pass rate'],['quality_retention_vs_sync','Retention vs sync'],['verified_rollouts_per_tick','Verified / logical tick'],['verified_throughput_multiplier_vs_sync','Throughput vs sync'],['mean_lag','Mean policy lag'],['p95_kl','P95 KL'],['proxy_hacking_rate','Proxy-only reward rate']];
let h='<table><tr><th>Mode</th>'+metrics.map(x=>'<th>'+x[1]+'</th>').join('')+'</tr>';
for(const [mode,v] of Object.entries(data)){{h+='<tr><td>'+mode+'</td>'+metrics.map(([k])=>'<td>'+Number(v[k]||0).toFixed(4)+'</td>').join('')+'</tr>';}}
h+='</table><h2>Verified pass rate</h2>';
for(const [mode,v] of Object.entries(data)){{const w=Math.round((v.verified_pass_rate||0)*500);h+='<p>'+mode+' '+(v.verified_pass_rate||0).toFixed(3)+'</p><div class="bar" style="width:'+w+'px"></div>';}}
document.getElementById('app').innerHTML=h;
</script></body></html>"""
    path.write_text(html, encoding="utf-8")


def build_report(path: Path, comparison: Dict[str, object]) -> None:
    modes = comparison.get("modes", {})
    backend = str(comparison.get("policy_backend", "transformer"))
    domain = str(comparison.get("task_domain", "arithmetic"))
    runs = comparison.get("runs", [])
    run_config = comparison.get("run_config", {})
    seed_count = max((int(run.get("seed", 0)) for run in runs), default=0) + 1
    if backend in {"vertex", "vertex_frozen"}:
        frozen_label = "frozen Gemini" if backend == "vertex_frozen" else "Gemini with a local policy-gradient bias"
        opening = f"This report uses {frozen_label} through Vertex AI for the policy's base action distribution. The verifier and controller remain local."
        reproduce = (
            f"python3 -m staleroll.cli run --policy {backend} --task-domain {domain} --seeds {seed_count} "
            f"--tasks {run_config.get('tasks', 10)} --workers {run_config.get('workers', 2)} "
            f"--ticks {run_config.get('ticks', 24)} --batch-size {run_config.get('batch_size', 2)} "
            f"--max-delay {run_config.get('max_delay', 4)} --max-lag {run_config.get('max_lag', 8)} "
            f"--max-kl {run_config.get('max_kl', 1.25)} --lag-decay {run_config.get('lag_decay', 0.12)} "
            f"--kl-decay {run_config.get('kl_decay', 0.4)} --vertex-max-requests {run_config.get('vertex_max_requests', 40)} "
            f"--vertex-budget-usd {run_config.get('vertex_budget_usd', 180)} --output artifacts/gcp_code_benchmark_tuned"
        )
    elif backend == "lora":
        opening = "This report uses a frozen character Transformer encoder with a trainable task-conditioned low-rank (LoRA) action head. Checkpoints and oracle evaluations are saved during training."
        checkpoint_dir = str(run_config.get("checkpoint_dir", ""))
        artifact_root = "artifacts/lora_code_training"
        if checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir)
            if len(checkpoint_path.parents) >= 3:
                artifact_root = str(checkpoint_path.parents[2])
        reproduce = (
            f"python3 -m staleroll.cli run --policy lora --task-domain {domain} --seeds {seed_count} "
            f"--tasks {run_config.get('tasks', 100)} --workers {run_config.get('workers', 2)} "
            f"--ticks {run_config.get('ticks', 96)} --batch-size {run_config.get('batch_size', 4)} "
            f"--max-delay {run_config.get('max_delay', 8)} --max-lag {run_config.get('max_lag', 0)} "
            f"--max-kl {run_config.get('max_kl', 0.2)} --lora-rank {run_config.get('lora_rank', 8)} "
            f"--lora-alpha {run_config.get('lora_alpha', 16.0)} --lora-warmup-epochs {run_config.get('lora_warmup_epochs', 1)} "
            f"--checkpoint-interval {run_config.get('checkpoint_interval', 10)} "
            f"--eval-interval {run_config.get('eval_interval', 10)} --output {artifact_root}"
        )
    elif backend == "tabular":
        opening = "This report uses the retained tabular policy ablation with the local verifier and controller."
        reproduce = "python3 -m staleroll.cli run --policy tabular --seeds 3 --tasks 80 --workers 4 --batch-size 4 --max-delay 8 --ticks 120"
    else:
        opening = "This report uses the local character-token Transformer policy. It tests the rollout controller and measurement protocol; it is not a claim about a Qwen-scale checkpoint."
        reproduce = "python3 -m staleroll.cli run --policy transformer --seeds 3 --tasks 80 --workers 4 --batch-size 4 --max-delay 8 --ticks 120"
    lines = [
        "# StaleRoll experiment report",
        "",
        opening,
        "",
        "## Research question",
        "",
        "> Can asynchronous RL with stale rollout workers preserve verified quality while increasing rollout throughput?",
        "",
        "## Protocol",
        "",
        ("Each task asks the policy to select a complete Python repair. The proxy evaluator runs visible unit tests, while the verifier executes hidden unit tests. A candidate that passes only the visible tests is counted as proxy-reward hacking." if domain == "code" else "Each task asks the policy to select a restricted arithmetic expression. The proxy evaluator sees two examples, while the verifier checks four hidden examples. A candidate that fits only the visible examples is counted as proxy-reward hacking."),
        "",
        "Compared modes: synchronous barrier training, naive asynchronous training, staleness-filtered asynchronous training, and a random acceptance control.",
        "",
        "## Results",
        "",
        "| Mode | Verified pass rate | Retention vs sync | Verified/tick | Throughput vs sync | Mean lag | P95 KL | Proxy-only rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, values in modes.items():
        lines.append(
            f"| {mode} | {values.get('verified_pass_rate', 0):.4f} | {values.get('quality_retention_vs_sync', 0):.3f} | {values.get('verified_rollouts_per_tick', 0):.3f} | {values.get('verified_throughput_multiplier_vs_sync', 0):.3f} | {values.get('mean_lag', 0):.3f} | {values.get('p95_kl', 0):.3f} | {values.get('proxy_hacking_rate', 0):.4f} |"
        )
    if backend in {"vertex", "vertex_frozen"}:
        usage_rows = [run.get("policy_usage", {}) for run in runs if run.get("policy_usage")]
        request_count = sum(int(row.get("vertex_requests", 0)) for row in usage_rows)
        estimated_cost = sum(float(row.get("vertex_estimated_cost_usd", 0.0)) for row in usage_rows)
        input_tokens = sum(int(row.get("vertex_input_tokens", 0)) for row in usage_rows)
        output_tokens = sum(int(row.get("vertex_output_tokens", 0)) for row in usage_rows)
        model_versions = sorted({str(row.get("vertex_model_version", "")) for row in usage_rows if row.get("vertex_model_version")})
        model_label = ", ".join(model_versions) if model_versions else "recorded in policy_usage.json"
        lines += [
            "",
            "## Vertex usage",
            "",
            f"The run used model version **{model_label}** and made **{request_count}** uncached Vertex requests ({input_tokens} input tokens and {output_tokens} output tokens). The local conservative estimate was **${estimated_cost:.2f}** against the configured **$180.00** guard. This estimate is an experiment guard, not a Cloud Billing invoice.",
            "",
        ]
    if backend == "lora":
        usage_rows = [run.get("policy_usage", {}) for run in runs if run.get("policy_usage")]
        updates = sum(int(row.get("lora_updates", 0)) for row in usage_rows)
        trainable = sorted({int(row.get("lora_trainable_parameters", 0)) for row in usage_rows})
        lines += [
            "## LoRA training",
            "",
            f"The run performed {updates} local adapter updates. The trainable low-rank head contained {trainable[0] if trainable else 0} parameters per policy instance; checkpoint and evaluation paths are recorded in each run's `policy_usage.json`.",
            "",
        ]
    timing_rows = [run.get("timing", {}) for run in runs if run.get("timing")]
    if timing_rows:
        wall = sum(float(row.get("wall_clock_seconds", 0.0)) for row in timing_rows) / len(timing_rows)
        sample = sum(float(row.get("sample_seconds", 0.0)) for row in timing_rows) / len(timing_rows)
        update = sum(float(row.get("update_seconds", 0.0)) for row in timing_rows) / len(timing_rows)
        verifier = sum(float(row.get("verifier_seconds", 0.0)) for row in timing_rows) / len(timing_rows)
        lines += [
            "## Wall-clock profile",
            "",
            f"Mean per-run wall time was **{wall:.3f}s**: sampling/inference {sample:.3f}s, local policy updates {update:.3f}s, and verification {verifier:.3f}s. Raw profiles are stored beside each run summary.",
            "",
        ]
    if backend in {"vertex", "vertex_frozen", "lora"} and run_config:
        lines += [
            "## Training configuration",
            "",
            f"The run used {run_config.get('ticks')} logical ticks, batch size {run_config.get('batch_size')}, maximum delay {run_config.get('max_delay')}, maximum accepted lag {run_config.get('max_lag')}, and KL cutoff {run_config.get('max_kl')}.",
            "",
        ]
    if len(runs) > 4:
        stale_values = modes.get("stale_filter", {})
        naive_values = modes.get("naive_async", {})
        stale_retention = float(stale_values.get("quality_retention_vs_sync", 0.0))
        lines += [
            "## Interpretation",
            "",
            f"Across {seed_count} matched seeds, naive asynchronous updates reached {float(naive_values.get('verified_pass_rate', 0.0)):.4f} verified pass rate at {float(naive_values.get('verified_throughput_multiplier_vs_sync', 0.0)):.3f}x synchronous verified throughput. The staleness filter retained {stale_retention:.3f} of synchronous verified quality, so the preregistered 0.95 retention target was {'met' if stale_retention >= 0.95 else 'not met'} in this small run. These are descriptive results, not a production generalization.",
            "",
        ]
    lines += [
        "",
        "## Claim discipline",
        "",
        "The experiment should only support a throughput/quality claim if the measured confidence intervals and repeated seeds meet a pre-registered threshold. The simulator intentionally reports the evidence without declaring a win automatically.",
        "",
        "## Reproduce",
        "",
        "```bash",
        reproduce,
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")

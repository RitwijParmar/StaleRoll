# StaleRoll

**Risk-aware asynchronous RL with verifiable rewards**

StaleRoll is a small, reproducible research project for one precise question:

> Can asynchronous rollout workers use stale policy versions without losing verified quality, and can a controller turn the quality/throughput tradeoff into a measurable operating point?

It implements a Prime Intellect-style split between rollout workers, a verifier, and a trainer. The default experiment uses a local PyTorch Transformer. An explicit `vertex` backend can ask Gemini 2.5 Flash-Lite for a base action distribution while keeping the policy-gradient controller, verifier, and staleness logic local.

## What is implemented

- Four matched baselines: `sync`, `naive_async`, `stale_filter`, and `random_filter`.
- Discrete-event asynchronous workers with versioned policy snapshots, delays, failures, and verifier noise.
- A staleness controller that records policy lag and KL divergence, rejects unsafe samples, and clips importance weights.
- Group-relative updates with a task-conditioned low-rank (LoRA) action adapter over a frozen Transformer encoder.
- A restricted arithmetic-expression verifier: visible examples create a proxy signal, hidden examples provide the true reward, and proxy-only successes expose reward hacking.
- An execution-verified code-repair domain (`--task-domain code`) with 100 deterministic task variants from compact Python repair families, visible and hidden unit tests, candidate shuffling, and a restricted execution namespace.
- A bounded Vertex Gemini backend (`--policy vertex`) with per-request caching, a request-count guard, and a local estimated-cost guard. The cloud model supplies action priors; it is not fine-tuned or silently billed by the default local run.
- JSONL trajectory logs, confidence intervals, quality-vs-lag buckets, a comparison report, and a standalone dashboard.
- Checkpointed LoRA training with interval oracle evaluations, final evaluation flushes, and wall-clock profiles for sampling, verification, and updates.
- A frozen-Gemini comparison at the same 100-task scale, including request counts and local cost estimates.
- An adapter contract and configuration notes for a later `prime-rl` / Verifiers integration.

## Run it

```bash
cd StaleRoll
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[model,cloud]'
python3 -m staleroll.cli run --policy transformer --seeds 3 --tasks 80 --workers 4 --batch-size 4 --max-delay 8 --ticks 120
```

The command writes `artifacts/latest/comparison.json`, `report.md`, `dashboard.html`, and one `trajectories.jsonl` per mode and seed. Open `dashboard.html` locally to inspect the result.

The same reproduction is available as `make reproduce`. Increase `--tasks` and `--ticks` for a larger local run.

## Bounded Vertex smoke run

The configured GCP project is `gen-lang-client-0576163520`, using
`gemini-2.5-flash-lite` in `us-central1`. The following command uses one seed,
10 tasks, and at most 40 model requests across the four controller arms. The
client stops before its local request or estimated-cost guard is exceeded:

```bash
PYTHONPATH=src python3 -m staleroll.cli run \
  --policy vertex --seeds 1 --tasks 10 --workers 2 --ticks 24 \
  --task-domain code \
  --batch-size 2 --max-delay 4 --vertex-max-requests 40 \
  --vertex-budget-usd 180 --output artifacts/gcp_smoke
```

Each run writes `policy_usage.json` with the project, model, model version,
token counts, request count, and local estimated cost. The Cloud Billing budget
is an alert budget; it is not a hard shutdown. Keep the request guard enabled
for every cloud run.

The completed code-domain smoke artifact is in
`artifacts/gcp_code_smoke_final/`. With one seed and 10 tasks, synchronous
training reached a 0.875 verified pass rate; the asynchronous arms reached
0.9375 and 1.319x verified rollouts per logical tick. This is an integration
smoke result, not a multi-seed benchmark claim. It used 34 Vertex requests and
the local conservative estimate was $0.34.

The three-seed continuation is reproducible with `make gcp-code-benchmark` and
is saved under `artifacts/gcp_code_benchmark_tuned/`. With 96 logical ticks,
the tuned staleness filter retained 1.061x of synchronous verified quality and
delivered 1.169x verified throughput.

## Local LoRA training and frozen-Gemini comparison

The completed local training run is in
`artifacts/lora_code_training_final2/`. It uses 100 execution-verified repair
tasks, three seeds, 96 logical ticks, a rank-8/alpha-16 task-conditioned LoRA
head, three supervised warm-start epochs, verifier-weighted policy-gradient
updates, evaluations every five updates, and checkpoints every ten updates.
It produced 48 checkpoint files and 324 adapter updates across the four
controller arms. The mean run profile was 10.474 seconds wall-clock, with
0.271 seconds in sampling, 3.840 seconds in adapter updates, and 0.015 seconds
in verification.

The frozen Gemini reference at the same 100-task scale is in
`artifacts/vertex_frozen_code_final100/`; the matched analysis is
`artifacts/backend_comparison_final/`. In the stale-filter arm, LoRA reached
0.7552 verified pass rate and 0.5916 verified rollouts per tick; frozen Gemini
reached 0.7622 and 0.5888. The synchronous LoRA arm was 0.7951 versus 0.7839
for frozen Gemini. The comparison uses three LoRA seeds and two frozen-Gemini
seeds because the cloud arm is request-limited; it is an engineering comparison,
not a final significance claim.

Re-run the local training with `make lora-code-training`, the cloud reference
with `make gcp-frozen-code`, and regenerate the comparison with
`make compare-backends`.

## Demo

The narrated demo is a real MP4 asset, not a placeholder: [watch the StaleRoll demo](demo/staleroll_demo.mp4). It uses a conversational voiceover and a pen-and-marker metaphor to walk through the verifier, the LoRA adapter, the staleness controller, and the measured 100-task comparison. The poster is [here](demo/staleroll_demo_poster.png), and the narration/build source is in [`demo/`](demo/).

The accumulated per-run request audit is in
`artifacts/cloud_usage_summary.json`; it records the earlier 572 requests and
$5.72 local guard estimate, plus the two frozen-Gemini comparison artifacts for
927 saved-artifact requests and a $9.27 local guard estimate in total. This is
not a Cloud Billing invoice and excludes any request from an interrupted run
that did not produce a saved usage artifact.

Fast smoke run:

```bash
PYTHONHASHSEED=0 python3 -m staleroll.cli run --seeds 1 --tasks 40 --workers 3 --ticks 60 --output artifacts/smoke
```

## Research claim standard

Do not claim that stale filtering wins from one run. Pre-register a threshold, repeat at least three seeds, report confidence intervals, and compare verified quality at a fixed throughput budget. The report intentionally prints evidence and leaves the claim decision to the researcher.

## Spend boundary

The default local command makes no cloud calls. The `vertex` backend is opt-in
and calls only the configured Vertex endpoint. It applies a local request-count
and estimated-cost stop before each uncached task request. Google Cloud Billing
also has a project budget with alerts at 50%, 80%, 90%, and 100%; budgets are
alerts rather than a guaranteed hard cap, so the local stop remains enabled.

## Project map

- `src/staleroll/simulator.py` — discrete-event rollout/trainer simulation.
- `src/staleroll/controller.py` — lag/KL filtering and clipped weights.
- `src/staleroll/tasks.py` — safe task generation and restricted verifier.
- `src/staleroll/neural_policy.py` — character-token Transformer policy, supervised warm-start, and verifier-weighted policy-gradient updates.
- `src/staleroll/lora_policy.py` — frozen Transformer encoder with trainable task-conditioned low-rank adapter, checkpointing, evaluations, and adapter profiling.
- `src/staleroll/vertex_policy.py` — Gemini-backed action priors with caching, online bias updates, and local spend guards.
- `src/staleroll/policy.py` — retained tabular baseline for an explicit ablation (`--policy tabular`).
- `src/staleroll/metrics.py` — pass rates, confidence intervals, lag buckets.
- `docs/protocol.md` — frozen evaluation protocol.
- `docs/prime_rl_integration.md` — path to a real Verifiers/prime-rl run.
- `docs/gcp.md` — the Vertex AI billing gate and the current account diagnosis.
- `src/staleroll/prime_rl_adapter.py` — dependency-free mapping/decision seam for native rollout records.
- `demo/` — narrated MP4 demo, poster, narration text, and reproducible video builder.

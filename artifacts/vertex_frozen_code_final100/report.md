# StaleRoll experiment report

This report uses frozen Gemini through Vertex AI for the policy's base action distribution. The verifier and controller remain local.

## Research question

> Can asynchronous RL with stale rollout workers preserve verified quality while increasing rollout throughput?

## Protocol

Each task asks the policy to select a complete Python repair. The proxy evaluator runs visible unit tests, while the verifier executes hidden unit tests. A candidate that passes only the visible tests is counted as proxy-reward hacking.

Compared modes: synchronous barrier training, naive asynchronous training, staleness-filtered asynchronous training, and a random acceptance control.

## Results

| Mode | Verified pass rate | Retention vs sync | Verified/tick | Throughput vs sync | Mean lag | P95 KL | Proxy-only rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| naive_async | 0.7747 | 0.988 | 0.599 | 1.445 | 0.610 | 0.254 | 0.0635 |
| random_filter | 0.7622 | 0.972 | 0.589 | 1.421 | 0.610 | 0.254 | 0.0697 |
| stale_filter | 0.7622 | 0.972 | 0.589 | 1.421 | 0.610 | 0.254 | 0.0697 |
| sync | 0.7839 | 1.000 | 0.414 | 1.000 | 0.000 | 0.254 | 0.0286 |

## Vertex usage

The run used model version **gemini-2.5-flash-lite** and made **213** uncached Vertex requests (63259 input tokens and 8381 output tokens). The local conservative estimate was **$2.13** against the configured **$180.00** guard. This estimate is an experiment guard, not a Cloud Billing invoice.

## Wall-clock profile

Mean per-run wall time was **35.293s**: sampling/inference 35.190s, local policy updates 0.000s, and verification 0.051s. Raw profiles are stored beside each run summary.

## Training configuration

The run used 96 logical ticks, batch size 4, maximum delay 8, maximum accepted lag 0, and KL cutoff 0.2.

## Interpretation

Across 2 matched seeds, naive asynchronous updates reached 0.7747 verified pass rate at 1.445x synchronous verified throughput. The staleness filter retained 0.972 of synchronous verified quality, so the preregistered 0.95 retention target was met in this small run. These are descriptive results, not a production generalization.


## Claim discipline

The experiment should only support a throughput/quality claim if the measured confidence intervals and repeated seeds meet a pre-registered threshold. The simulator intentionally reports the evidence without declaring a win automatically.

## Reproduce

```bash
python3 -m staleroll.cli run --policy vertex_frozen --task-domain code --seeds 2 --tasks 100 --workers 4 --ticks 96 --batch-size 4 --max-delay 8 --max-lag 0 --max-kl 0.2 --lag-decay 0.25 --kl-decay 0.8 --vertex-max-requests 320 --vertex-budget-usd 180.0 --output artifacts/gcp_code_benchmark_tuned
```

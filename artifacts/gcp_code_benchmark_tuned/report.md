# StaleRoll experiment report

This report uses Gemini through Vertex AI for the policy's base action distribution. The verifier, controller, and online policy-gradient bias remain local.

## Research question

> Can asynchronous RL with stale rollout workers preserve verified quality while increasing rollout throughput?

## Protocol

Each task asks the policy to select a complete Python repair. The proxy evaluator runs visible unit tests, while the verifier executes hidden unit tests. A candidate that passes only the visible tests is counted as proxy-reward hacking.

Compared modes: synchronous barrier training, naive asynchronous training, staleness-filtered asynchronous training, and a random acceptance control.

## Results

| Mode | Verified pass rate | Retention vs sync | Verified/tick | Throughput vs sync | Mean lag | P95 KL | Proxy-only rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| naive_async | 0.8851 | 1.085 | 0.296 | 1.194 | 0.210 | 0.176 | 0.0291 |
| random_filter | 0.8756 | 1.073 | 0.293 | 1.181 | 0.210 | 0.197 | 0.0291 |
| stale_filter | 0.8661 | 1.061 | 0.290 | 1.169 | 0.210 | 0.221 | 0.0291 |
| sync | 0.8160 | 1.000 | 0.248 | 1.000 | 0.000 | 0.249 | 0.0451 |

## Vertex usage

The run used model version **gemini-2.5-flash-lite** and made **120** uncached Vertex requests (36696 input tokens and 4719 output tokens). The local conservative estimate was **$1.20** against the configured **$180.00** guard. This estimate is an experiment guard, not a Cloud Billing invoice.

## Training configuration

The run used 96 logical ticks, batch size 4, maximum delay 8, maximum accepted lag 0, and KL cutoff 0.2.

## Interpretation

Across 3 matched seeds, naive asynchronous updates reached 0.8851 verified pass rate at 1.194x synchronous verified throughput. The staleness filter retained 1.061 of synchronous verified quality, so the preregistered 0.95 retention target was met in this small run. These are descriptive results, not a production generalization.


## Claim discipline

The experiment should only support a throughput/quality claim if the measured confidence intervals and repeated seeds meet a pre-registered threshold. The simulator intentionally reports the evidence without declaring a win automatically.

## Reproduce

```bash
python3 -m staleroll.cli run --policy vertex --task-domain code --seeds 3 --tasks 10 --workers 2 --ticks 96 --batch-size 4 --max-delay 8 --max-lag 0 --max-kl 0.2 --lag-decay 0.25 --kl-decay 0.8 --vertex-max-requests 40 --vertex-budget-usd 180.0 --output artifacts/gcp_code_benchmark_tuned
```

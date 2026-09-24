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
| naive_async | 0.9375 | 1.071 | 0.577 | 1.319 | 0.312 | 0.055 | 0.0000 |
| random_filter | 0.9375 | 1.071 | 0.577 | 1.319 | 0.312 | 0.062 | 0.0000 |
| stale_filter | 0.9375 | 1.071 | 0.577 | 1.319 | 0.312 | 0.055 | 0.0000 |
| sync | 0.8750 | 1.000 | 0.438 | 1.000 | 0.000 | 0.210 | 0.0000 |

## Vertex usage

The run used model version **gemini-2.5-flash-lite** and made **34** uncached Vertex requests (10198 input tokens and 1322 output tokens). The local conservative estimate was **$0.34** against the configured **$180.00** guard. This estimate is an experiment guard, not a Cloud Billing invoice.


## Claim discipline

The experiment should only support a throughput/quality claim if the measured confidence intervals and repeated seeds meet a pre-registered threshold. The simulator intentionally reports the evidence without declaring a win automatically.

## Reproduce

```bash
python3 -m staleroll.cli run --policy vertex --task-domain code --seeds 1 --tasks 10 --workers 2 --ticks 24 --batch-size 2 --max-delay 4 --vertex-max-requests 40 --vertex-budget-usd 180 --output artifacts/gcp_code_smoke_final
```

# StaleRoll experiment report

This report uses Gemini through Vertex AI for the policy's base action distribution. The verifier, controller, and online policy-gradient bias remain local.

## Research question

> Can asynchronous RL with stale rollout workers preserve verified quality while increasing rollout throughput?

## Protocol

Each task asks the policy to select a restricted arithmetic expression. The proxy evaluator sees two examples, while the verifier checks four hidden examples. A candidate that fits only the visible examples is counted as proxy-reward hacking.

Compared modes: synchronous barrier training, naive asynchronous training, staleness-filtered asynchronous training, and a random acceptance control.

## Results

| Mode | Verified pass rate | Retention vs sync | Verified/tick | Throughput vs sync | Mean lag | P95 KL | Proxy-only rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| naive_async | 0.8750 | 1.235 | 0.538 | 1.520 | 0.312 | 0.075 | 0.1250 |
| random_filter | 0.8750 | 1.235 | 0.538 | 1.520 | 0.312 | 0.075 | 0.1250 |
| stale_filter | 0.8750 | 1.235 | 0.538 | 1.520 | 0.312 | 0.075 | 0.1250 |
| sync | 0.7083 | 1.000 | 0.354 | 1.000 | 0.000 | 0.075 | 0.1667 |

## Vertex usage

The run made **34** uncached Vertex requests (5743 input tokens and 1308 output tokens). The local conservative estimate was **$0.34** against the configured **$180.00** guard. This estimate is an experiment guard, not a Cloud Billing invoice.


## Claim discipline

The experiment should only support a throughput/quality claim if the measured confidence intervals and repeated seeds meet a pre-registered threshold. The simulator intentionally reports the evidence without declaring a win automatically.

## Reproduce

```bash
python3 -m staleroll.cli run --policy vertex --seeds 1 --tasks 10 --workers 2 --ticks 24 --batch-size 2 --max-delay 4 --vertex-max-requests 40 --vertex-budget-usd 180 --output artifacts/gcp_smoke
```

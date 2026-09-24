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
| naive_async | 0.9000 | 1.013 | 0.581 | 1.229 | 0.325 | 0.153 | 0.0208 |
| random_filter | 0.9000 | 1.013 | 0.581 | 1.229 | 0.325 | 0.153 | 0.0208 |
| stale_filter | 0.8375 | 0.942 | 0.545 | 1.154 | 0.325 | 0.181 | 0.0208 |
| sync | 0.8889 | 1.000 | 0.472 | 1.000 | 0.000 | 0.127 | 0.0278 |

## Vertex usage

The run used model version **gemini-2.5-flash-lite** and made **108** uncached Vertex requests (32628 input tokens and 4252 output tokens). The local conservative estimate was **$1.08** against the configured **$180.00** guard. This estimate is an experiment guard, not a Cloud Billing invoice.

## Interpretation

Across 3 matched seeds, naive asynchronous updates reached 0.9000 verified pass rate at 1.229x synchronous verified throughput. The staleness filter retained 0.942 of synchronous verified quality, so the preregistered 0.95 retention target was not met in this small run. These are descriptive results, not a production generalization.


## Claim discipline

The experiment should only support a throughput/quality claim if the measured confidence intervals and repeated seeds meet a pre-registered threshold. The simulator intentionally reports the evidence without declaring a win automatically.

## Reproduce

```bash
python3 -m staleroll.cli run --policy vertex --task-domain code --seeds 3 --tasks 10 --workers 2 --ticks 24 --batch-size 2 --max-delay 4 --vertex-max-requests 40 --vertex-budget-usd 180 --output artifacts/gcp_code_benchmark_final
```

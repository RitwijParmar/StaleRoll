# StaleRoll experiment report

This report is produced by a CPU-only discrete-event run using the local character-token Transformer policy. It tests the rollout controller and measurement protocol; it is not a claim about a Qwen-scale checkpoint.

## Research question

> Can asynchronous RL with stale rollout workers preserve verified quality while increasing rollout throughput?

## Protocol

Each task asks the policy to select a restricted arithmetic expression. The proxy evaluator sees two examples, while the verifier checks four hidden examples. A candidate that fits only the visible examples is counted as proxy-reward hacking.

Compared modes: synchronous barrier training, naive asynchronous training, staleness-filtered asynchronous training, and a random acceptance control.

## Results

| Mode | Verified pass rate | Retention vs sync | Verified/tick | Throughput vs sync | Mean lag | P95 KL | Proxy-only rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| sync | 0.3000 | 1.000 | 0.148 | 1.000 | 0.000 | 0.032 | 0.1333 |
| naive_async | 0.1500 | 0.500 | 0.088 | 0.598 | 0.350 | 0.042 | 0.2000 |
| stale_filter | 0.1500 | 0.500 | 0.088 | 0.598 | 0.350 | 0.041 | 0.2000 |
| random_filter | 0.1500 | 0.500 | 0.088 | 0.598 | 0.350 | 0.038 | 0.2000 |

## Claim discipline

The experiment should only support a throughput/quality claim if the measured confidence intervals and repeated seeds meet a pre-registered threshold. The simulator intentionally reports the evidence without declaring a win automatically.

## Reproduce

```bash
python3 -m staleroll.cli run --seeds 3 --tasks 160 --workers 4 --batch-size 4 --max-delay 8 --ticks 240
```

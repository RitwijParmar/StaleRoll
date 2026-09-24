# StaleRoll experiment report

This report is produced by the CPU-only discrete-event simulator. It tests the rollout controller and measurement protocol; it does not claim to train a language model.

## Research question

> Can asynchronous RL with stale rollout workers preserve verified quality while increasing rollout throughput?

## Protocol

Each task asks the policy to select a restricted arithmetic expression. The proxy evaluator sees two examples, while the verifier checks four hidden examples. A candidate that fits only the visible examples is counted as proxy-reward hacking.

Compared modes: synchronous barrier training, naive asynchronous training, staleness-filtered asynchronous training, and a random acceptance control.

## Results

| Mode | Verified pass rate | Rollouts/tick | Verified/tick | Mean lag | P95 KL | Proxy-only rate |
|---|---:|---:|---:|---:|---:|---:|
| sync | 0.9000 | 1.500 | 1.350 | 0.000 | 0.071 | 0.0111 |
| naive_async | 0.3478 | 0.767 | 0.267 | 0.130 | 0.356 | 0.0870 |
| stale_filter | 0.3478 | 0.767 | 0.267 | 0.130 | 0.358 | 0.0870 |
| random_filter | 0.3478 | 0.767 | 0.267 | 0.130 | 0.303 | 0.1304 |

## Claim discipline

The experiment should only support a throughput/quality claim if the measured confidence intervals and repeated seeds meet a pre-registered threshold. The simulator intentionally reports the evidence without declaring a win automatically.

## Reproduce

```bash
python -m staleroll.cli run --seeds 3 --tasks 160 --workers 3 --ticks 240
```

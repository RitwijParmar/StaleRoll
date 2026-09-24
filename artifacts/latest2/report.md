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
| sync | 0.9204 | 0.517 | 0.476 | 0.000 | 0.035 | 0.0130 |
| naive_async | 0.8626 | 0.735 | 0.634 | 0.133 | 0.045 | 0.0200 |
| stale_filter | 0.8626 | 0.735 | 0.634 | 0.133 | 0.046 | 0.0219 |
| random_filter | 0.8571 | 0.735 | 0.630 | 0.133 | 0.065 | 0.0255 |

## Claim discipline

The experiment should only support a throughput/quality claim if the measured confidence intervals and repeated seeds meet a pre-registered threshold. The simulator intentionally reports the evidence without declaring a win automatically.

## Reproduce

```bash
python -m staleroll.cli run --seeds 3 --tasks 160 --workers 3 --ticks 240
```

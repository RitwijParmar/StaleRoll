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
| sync | 0.9213 | 0.413 | 0.380 | 0.000 | 0.032 | 0.0074 |
| naive_async | 0.8870 | 0.562 | 0.499 | 0.426 | 0.069 | 0.0071 |
| stale_filter | 0.8822 | 0.562 | 0.496 | 0.426 | 0.070 | 0.0095 |
| random_filter | 0.8382 | 0.562 | 0.471 | 0.426 | 0.135 | 0.0255 |

## Claim discipline

The experiment should only support a throughput/quality claim if the measured confidence intervals and repeated seeds meet a pre-registered threshold. The simulator intentionally reports the evidence without declaring a win automatically.

## Reproduce

```bash
python -m staleroll.cli run --seeds 3 --tasks 160 --workers 3 --ticks 240
```

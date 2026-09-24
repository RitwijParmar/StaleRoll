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
| sync | 0.9368 | 0.522 | 0.489 | 0.000 | 0.021 | 0.0042 |
| naive_async | 0.9045 | 0.770 | 0.697 | 0.644 | 0.093 | 0.0087 |
| stale_filter | 0.8958 | 0.770 | 0.690 | 0.644 | 0.090 | 0.0122 |
| random_filter | 0.8974 | 0.770 | 0.691 | 0.644 | 0.144 | 0.0140 |

## Claim discipline

The experiment should only support a throughput/quality claim if the measured confidence intervals and repeated seeds meet a pre-registered threshold. The simulator intentionally reports the evidence without declaring a win automatically.

## Reproduce

```bash
python -m staleroll.cli run --seeds 3 --tasks 160 --workers 3 --ticks 240
```

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
| sync | 0.9406 | 0.525 | 0.493 | 0.000 | 0.027 | 0.0063 |
| naive_async | 0.8722 | 0.773 | 0.674 | 1.271 | 0.280 | 0.0301 |
| stale_filter | 0.8647 | 0.773 | 0.669 | 1.271 | 0.336 | 0.0376 |
| random_filter | 0.8647 | 0.773 | 0.669 | 1.271 | 0.266 | 0.0301 |

## Claim discipline

The experiment should only support a throughput/quality claim if the measured confidence intervals and repeated seeds meet a pre-registered threshold. The simulator intentionally reports the evidence without declaring a win automatically.

## Reproduce

```bash
python -m staleroll.cli run --seeds 3 --tasks 160 --workers 3 --ticks 240
```

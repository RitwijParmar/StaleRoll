# StaleRoll experiment report

This report uses the retained tabular policy ablation with the local verifier and controller.

## Research question

> Can asynchronous RL with stale rollout workers preserve verified quality while increasing rollout throughput?

## Protocol

Each task asks the policy to select a complete Python repair. The proxy evaluator runs visible unit tests, while the verifier executes hidden unit tests. A candidate that passes only the visible tests is counted as proxy-reward hacking.

Compared modes: synchronous barrier training, naive asynchronous training, staleness-filtered asynchronous training, and a random acceptance control.

## Results

| Mode | Verified pass rate | Retention vs sync | Verified/tick | Throughput vs sync | Mean lag | P95 KL | Proxy-only rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| sync | 0.4000 | 1.000 | 0.200 | 1.000 | 0.000 | 0.035 | 0.1000 |
| naive_async | 0.3333 | 0.833 | 0.174 | 0.870 | 0.250 | 0.030 | 0.1667 |
| stale_filter | 0.3333 | 0.833 | 0.174 | 0.870 | 0.250 | 0.030 | 0.1667 |
| random_filter | 0.3333 | 0.833 | 0.174 | 0.870 | 0.250 | 0.029 | 0.1667 |

## Claim discipline

The experiment should only support a throughput/quality claim if the measured confidence intervals and repeated seeds meet a pre-registered threshold. The simulator intentionally reports the evidence without declaring a win automatically.

## Reproduce

```bash
python3 -m staleroll.cli run --policy tabular --seeds 3 --tasks 80 --workers 4 --batch-size 4 --max-delay 8 --ticks 120
```

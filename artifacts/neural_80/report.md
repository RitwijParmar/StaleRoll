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
| sync | 0.4375 | 1.000 | 0.235 | 1.000 | 0.000 | 0.092 | 0.0458 |
| naive_async | 0.4490 | 1.026 | 0.333 | 1.416 | 0.663 | 0.345 | 0.0306 |
| stale_filter | 0.4388 | 1.003 | 0.326 | 1.384 | 0.663 | 0.597 | 0.0306 |
| random_filter | 0.3367 | 0.770 | 0.250 | 1.062 | 0.663 | 0.248 | 0.1327 |

## Claim discipline

The experiment should only support a throughput/quality claim if the measured confidence intervals and repeated seeds meet a pre-registered threshold. The simulator intentionally reports the evidence without declaring a win automatically.

## Reproduce

```bash
python3 -m staleroll.cli run --seeds 3 --tasks 160 --workers 4 --batch-size 4 --max-delay 8 --ticks 240
```

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
| sync | 0.3917 | 1.000 | 0.211 | 1.000 | 0.000 | 0.098 | 0.0167 |
| naive_async | 0.3265 | 0.834 | 0.242 | 1.150 | 0.663 | 0.211 | 0.1122 |
| stale_filter | 0.3980 | 1.016 | 0.295 | 1.402 | 0.663 | 0.352 | 0.0102 |
| random_filter | 0.2347 | 0.599 | 0.174 | 0.827 | 0.663 | 0.305 | 0.1327 |

## Claim discipline

The experiment should only support a throughput/quality claim if the measured confidence intervals and repeated seeds meet a pre-registered threshold. The simulator intentionally reports the evidence without declaring a win automatically.

## Reproduce

```bash
python3 -m staleroll.cli run --seeds 3 --tasks 160 --workers 4 --batch-size 4 --max-delay 8 --ticks 240
```

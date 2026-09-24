# StaleRoll experiment report

This report uses a frozen character Transformer encoder with a trainable task-conditioned low-rank (LoRA) action head. Checkpoints and oracle evaluations are saved during training.

## Research question

> Can asynchronous RL with stale rollout workers preserve verified quality while increasing rollout throughput?

## Protocol

Each task asks the policy to select a complete Python repair. The proxy evaluator runs visible unit tests, while the verifier executes hidden unit tests. A candidate that passes only the visible tests is counted as proxy-reward hacking.

Compared modes: synchronous barrier training, naive asynchronous training, staleness-filtered asynchronous training, and a random acceptance control.

## Results

| Mode | Verified pass rate | Retention vs sync | Verified/tick | Throughput vs sync | Mean lag | P95 KL | Proxy-only rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| sync | 0.5312 | 1.000 | 0.156 | 1.000 | 0.000 | 0.071 | 0.1172 |
| naive_async | 0.5443 | 1.024 | 0.181 | 1.156 | 0.176 | 0.061 | 0.0217 |
| stale_filter | 0.5443 | 1.024 | 0.181 | 1.156 | 0.176 | 0.061 | 0.0217 |
| random_filter | 0.5443 | 1.024 | 0.181 | 1.156 | 0.176 | 0.061 | 0.0217 |
## LoRA training

The run performed 103 local adapter updates. The trainable low-rank head contained 1032 parameters per policy instance; checkpoint and evaluation paths are recorded in each run's `policy_usage.json`.

## Wall-clock profile

Mean per-run wall time was **3.107s**: sampling/inference 0.138s, local policy updates 0.971s, and verification 0.005s. Raw profiles are stored beside each run summary.

## Training configuration

The run used 64 logical ticks, batch size 4, maximum delay 8, maximum accepted lag 0, and KL cutoff 0.2.

## Interpretation

Across 2 matched seeds, naive asynchronous updates reached 0.5443 verified pass rate at 1.156x synchronous verified throughput. The staleness filter retained 1.024 of synchronous verified quality, so the preregistered 0.95 retention target was met in this small run. These are descriptive results, not a production generalization.


## Claim discipline

The experiment should only support a throughput/quality claim if the measured confidence intervals and repeated seeds meet a pre-registered threshold. The simulator intentionally reports the evidence without declaring a win automatically.

## Reproduce

```bash
python3 -m staleroll.cli run --policy lora --task-domain code --seeds 2 --tasks 50 --workers 2 --ticks 64 --batch-size 4 --max-delay 8 --max-lag 0 --max-kl 0.2 --lora-rank 8 --lora-alpha 16.0 --checkpoint-interval 5 --eval-interval 5 --output artifacts/lora_code_training
```

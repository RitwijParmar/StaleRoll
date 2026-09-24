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
| sync | 0.5000 | 1.000 | 0.152 | 1.000 | 0.000 | 0.070 | 0.1094 |
| naive_async | 0.6957 | 1.391 | 0.211 | 1.382 | 0.174 | 0.067 | 0.0435 |
| stale_filter | 0.6957 | 1.391 | 0.211 | 1.382 | 0.174 | 0.067 | 0.0435 |
| random_filter | 0.6957 | 1.391 | 0.211 | 1.382 | 0.174 | 0.067 | 0.0435 |
## LoRA training

The run performed 50 local adapter updates. The trainable low-rank head contained 1032 parameters per policy instance; checkpoint and evaluation paths are recorded in each run's `policy_usage.json`.

## Wall-clock profile

Mean per-run wall time was **3.167s**: sampling/inference 0.138s, local policy updates 0.941s, and verification 0.005s. Raw profiles are stored beside each run summary.

## Training configuration

The run used 64 logical ticks, batch size 4, maximum delay 8, maximum accepted lag 0, and KL cutoff 0.2.


## Claim discipline

The experiment should only support a throughput/quality claim if the measured confidence intervals and repeated seeds meet a pre-registered threshold. The simulator intentionally reports the evidence without declaring a win automatically.

## Reproduce

```bash
python3 -m staleroll.cli run --policy lora --task-domain code --seeds 1 --tasks 50 --workers 2 --ticks 64 --batch-size 4 --max-delay 8 --max-lag 0 --max-kl 0.2 --lora-rank 8 --lora-alpha 16.0 --checkpoint-interval 5 --eval-interval 5 --output artifacts/lora_code_training
```

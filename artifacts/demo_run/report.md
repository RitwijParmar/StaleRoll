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
| sync | 0.5000 | 1.000 | 0.250 | 1.000 | 0.000 | 0.066 | 0.1250 |
| naive_async | 0.3750 | 0.750 | 0.231 | 0.923 | 0.125 | 0.065 | 0.1250 |
| stale_filter | 0.3750 | 0.750 | 0.231 | 0.923 | 0.125 | 0.065 | 0.1250 |
| random_filter | 0.3750 | 0.750 | 0.231 | 0.923 | 0.125 | 0.065 | 0.1250 |
## LoRA training

The run performed 24 local adapter updates. The trainable low-rank head contained 1032 parameters per policy instance; checkpoint and evaluation paths are recorded in each run's `policy_usage.json`.

## Wall-clock profile

Mean per-run wall time was **1.769s**: sampling/inference 0.075s, local policy updates 0.310s, and verification 0.003s. Raw profiles are stored beside each run summary.

## Training configuration

The run used 24 logical ticks, batch size 4, maximum delay 4, maximum accepted lag 0, and KL cutoff 0.2.


## Claim discipline

The experiment should only support a throughput/quality claim if the measured confidence intervals and repeated seeds meet a pre-registered threshold. The simulator intentionally reports the evidence without declaring a win automatically.

## Reproduce

```bash
python3 -m staleroll.cli run --policy lora --task-domain code --seeds 1 --tasks 20 --workers 2 --ticks 24 --batch-size 4 --max-delay 4 --max-lag 0 --max-kl 0.2 --lora-rank 8 --lora-alpha 16.0 --lora-warmup-epochs 1 --checkpoint-interval 4 --eval-interval 4 --output /Users/ritwij/Documents/ChatGPT/new project/StaleRoll/artifacts/demo_run
```

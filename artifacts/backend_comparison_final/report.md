# LoRA adapter vs frozen Gemini

Matched task/controller protocol: 100 execution-verified code tasks, maximum accepted lag 0, and KL cutoff 0.20. LoRA used 3 seeds/96 ticks; frozen Gemini used 2 seeds/96 ticks.

| Mode | LoRA pass rate | Frozen Gemini pass rate | Δ pass rate | LoRA verified/tick | Frozen Gemini verified/tick | Δ throughput |
|---|---:|---:|---:|---:|---:|---:|
| sync | 0.7951 | 0.7839 | +0.0113 | 0.4180 | 0.4143 | +0.0036 |
| naive_async | 0.7552 | 0.7747 | -0.0195 | 0.5916 | 0.5986 | -0.0070 |
| stale_filter | 0.7552 | 0.7622 | -0.0070 | 0.5916 | 0.5888 | +0.0028 |
| random_filter | 0.7510 | 0.7622 | -0.0112 | 0.5883 | 0.5888 | -0.0005 |

LoRA training used [1032] trainable parameters per policy instance and saved checkpoints/evaluation traces. Frozen Gemini used 213 Vertex requests with a local estimate of $2.13.

Interpretation: the LoRA path is a real local training path with checkpointed updates; frozen Gemini is the stronger quality baseline in this task-scale comparison. Seed counts differ because the Gemini arm is the cloud-cost-limited reference, so this is evidence for engineering direction rather than a final significance claim.

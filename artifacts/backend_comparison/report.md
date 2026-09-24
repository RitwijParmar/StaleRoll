# LoRA adapter vs frozen Gemini

Matched protocol: 50 execution-verified code tasks, two seeds, 64 logical ticks, maximum accepted lag 0, and KL cutoff 0.20.

| Mode | LoRA pass rate | Frozen Gemini pass rate | Δ pass rate | LoRA verified/tick | Frozen Gemini verified/tick | Δ throughput |
|---|---:|---:|---:|---:|---:|---:|
| sync | 0.5469 | 0.8047 | -0.2578 | 0.1608 | 0.2367 | -0.0759 |
| naive_async | 0.5443 | 0.8773 | -0.3331 | 0.1806 | 0.3031 | -0.1225 |
| stale_filter | 0.5443 | 0.8991 | -0.3548 | 0.1806 | 0.3097 | -0.1291 |
| random_filter | 0.5443 | 0.8773 | -0.3331 | 0.1806 | 0.3031 | -0.1225 |

LoRA training used [1032] trainable parameters per policy instance and saved checkpoints/evaluation traces. Frozen Gemini used 142 Vertex requests with a local estimate of $1.42.

Interpretation: the LoRA path is a real local training path with checkpointed updates; frozen Gemini is the stronger quality baseline in this matched run. The adapter is not presented as a replacement until it closes that gap on larger seeds.

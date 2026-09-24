# Architecture

```text
                 policy snapshot vN
                        |
        +---------------+----------------+
        |               |                |
   rollout worker 0  worker 1        worker 2
        |               |                |
        +------- delayed trajectories--+
                        |
                 verifier / reward
                        |
                staleness controller
             (lag, KL, reject, reweight)
                        |
                      trainer
                        |
                   policy vN+1
```

The simulator uses a deterministic logical clock rather than sleeping. Each rollout records its behavior-policy version, start and finish ticks, probability under the behavior policy, current-policy probability, lag, KL proxy, reward, verifier status, and controller decision. That makes failure analysis possible without needing a GPU or a distributed runtime. The `code` task domain executes candidate Python repairs against visible and hidden unit tests inside a restricted namespace, so the same controller can be exercised on a real model-selection workload instead of only arithmetic expressions.

The default policy is a character-token Transformer encoder with a supervised warm-start and verifier-weighted policy-gradient updates. It is a compact local neural model so the controller can be evaluated without downloading a checkpoint or using a cloud GPU. The tabular policy remains available only as an explicit ablation. The Prime Intellect integration document explains how to replace this local model with a LoRA trainer after the local evidence is strong enough to justify GPU spend.

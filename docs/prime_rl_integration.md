# Prime Intellect integration plan

StaleRoll's environment and controller can be moved into a real async RL run in three steps:

1. Implement the task/verifier as a Verifiers environment whose reward is the hidden-case pass result. Keep the visible-case proxy only as a diagnostic.
2. Add the trajectory metadata fields (`behavior_version`, `lag_steps`, `kl_divergence`, verifier result, and wall-clock age) to the rollout record or sidecar logger.
3. Port `StalenessController.assess` into the trainer's pre-update filter. Keep synchronous, naive async, and random-filter controls in the same config and use the same task seed list.

The current repository does not import `prime-rl` and does not launch a cloud job. A GPU run should use a pinned commit, a small Qwen checkpoint with LoRA, one verifier worker per rollout process, and a short smoke run before scaling. Save the exact config, checkpoint hashes, verifier image, and failed rollouts.

## Suggested first GPU configuration

- Model: a small Qwen checkpoint that fits the available GPU with LoRA.
- One trainer plus two or three rollout workers.
- Short context and capped completion length.
- One gradient step per update, fixed batch size, three seeds.
- Local or already-paid-for GPU first; do not attach a payment method to an unattended run.

## What this project can honestly claim

The CPU experiment can demonstrate that the controller is deterministic, instrumented, and statistically testable. It cannot establish a Qwen or production throughput result. That claim requires the pinned real integration run and its raw logs.

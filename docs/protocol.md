# Frozen experiment protocol

## Hypothesis

With delayed rollout workers, a controller that uses policy lag and behavior/current-policy KL to reject or down-weight stale trajectories will preserve at least 95% of synchronous verified pass rate while improving verified rollouts per logical tick over the synchronous baseline.

This is a target, not a result. The threshold is falsified if the confidence intervals or repeated seeds do not support it.

## Environment

The run selects a task domain with `--task-domain`. The default arithmetic
domain is a deterministic controller gate. The `code` domain expands compact
Python repair families into deterministic input variants; the completed
training run uses 100 tasks. Each candidate is a complete function, visible
tests provide the proxy signal, and hidden tests are executed in a restricted
namespace for the verified reward.

For the arithmetic domain, each task hides a linear function `f(x)=a*x+b`.
Two visible examples are shown to the proxy evaluator. The real verifier
evaluates the selected expression on four hidden inputs. Candidate expressions
are parsed with a restricted AST evaluator: no calls, attributes, imports,
names other than `x`, or side effects are permitted.

## Arms

1. `sync`: barrier after every worker round; all workers sample the current policy snapshot.
2. `naive_async`: accept every completed rollout and apply a clipped importance ratio.
3. `stale_filter`: reject samples beyond `max_lag` or `max_kl`, then apply age/KL decay and importance clipping.
4. `random_filter`: accept 72% of samples at random, controlling for sample rejection without using staleness information.

## Stressors

- Worker failure probability.
- Random worker delay up to `max_delay` logical ticks.
- Optional verifier noise.
- Hidden cases that reveal proxy-reward hacking.

## Measurements

- Verified pass rate with Wilson 95% interval.
- Rollouts and verified rollouts per logical tick.
- Mean/P95 policy lag and KL divergence.
- Acceptance/rejection counts and rejection reasons.
- Proxy-only reward rate.
- Quality by lag bucket.
- Approximate compute seconds per solved task (logical worker-seconds).
- Wall-clock seconds split into policy sampling, verifier execution, and local
  adapter updates.
- LoRA oracle accuracy and entropy at configured evaluation intervals, plus a
  checkpoint at each configured save interval and a final flushed checkpoint.

## Reproducibility

Use the same seed list and task count for every arm. The checked-in default stress profile uses four workers, batch size four, and a maximum delay of eight logical ticks so the lag controller is exercised. Store configs, per-trajectory JSONL, and the generated report.

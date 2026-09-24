# Results placeholder

Run the following to generate the actual report from the current code:

```bash
python3 -m staleroll.cli run --policy transformer --seeds 3 --tasks 80 --workers 4 --batch-size 4 --max-delay 8 --ticks 120
```

The generated `artifacts/latest/report.md` is the authoritative result for that run. This file stays a placeholder so a checked-in document cannot be mistaken for an unexecuted benchmark.

The bounded Vertex code-repair continuation is recorded in
`artifacts/gcp_code_benchmark_tuned/report.md`. It uses three matched seeds,
10 tasks per seed, 96 logical training ticks, and the same four controller
arms.

The completed local adapter run is recorded in
`artifacts/lora_code_training_final2/report.md`. It uses 100 code-repair tasks,
three seeds, interval evaluations, 48 checkpoints, and wall-clock profiles.
The frozen-Gemini reference and the task-scale comparison are recorded in
`artifacts/vertex_frozen_code_final100/report.md` and
`artifacts/backend_comparison_final/report.md`.

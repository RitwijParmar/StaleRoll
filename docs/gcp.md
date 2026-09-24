# Google Cloud / Vertex AI gate

The repository does not call Vertex AI automatically. The local Transformer is
the default. The explicit `--policy vertex` path uses the open, credit-backed
project below and has a local request/cost stop before every uncached request.

Before a cloud run, verify the account state:

```bash
gcloud config get-value project
gcloud billing projects describe "$(gcloud config get-value project)"
gcloud billing accounts list
```

The usable project is:

- project: `gen-lang-client-0576163520`
- billing account: `01B8F9-4880C4-320ABC` (open)
- location: `us-central1`
- model: `gemini-2.5-flash-lite`
- gcloud account: `ritwij.aryan.parmar@gmail.com`

The project has a **$180 USD** budget with alerts at 50%, 80%, 90%, and 100%.
This is an alert budget, not an automatic shutdown mechanism. The Vertex
client adds a second local guard: a maximum request count and a conservative
estimated cost per request. The smoke command uses 40 requests at most.

Google's [spend-cap budget preview](https://docs.cloud.google.com/billing/docs/how-to/budgets-spend-caps)
can pause eligible Vertex usage at 100%, but it must be created as a separate
console budget and requires billing permissions. The existing $180 budget is
alerts-only; do not describe it as a hard cap.

Google documents that free-trial usage is paid from the welcome credit, while a
paid billing account can bill usage after credits are exhausted. Do not attach
another account or payment method for this project. The existing open project
is the one used by the smoke run.

## Smoke run

From `StaleRoll/`:

```bash
pip install -e '.[model,cloud]'
PYTHONPATH=src python3 -m staleroll.cli run \
  --policy vertex --task-domain code --seeds 3 --tasks 10 --workers 2 --ticks 24 \
  --batch-size 2 --max-delay 4 --vertex-max-requests 40 \
  --vertex-budget-usd 180 --output artifacts/gcp_code_benchmark_final
```

Inspect `artifacts/gcp_code_benchmark_tuned/comparison.json` and each
`runs/*/seed-00/policy_usage.json`. The cloud model is used only to produce
candidate priors; the verifier and RL updates remain in this repository.

The completed frozen-Gemini reference uses the same 100-task code-repair scale
as the local LoRA run and is saved in
`artifacts/vertex_frozen_code_final100/`. Its side-by-side analysis with the
trained adapter is `artifacts/backend_comparison_final/report.md`.

The saved-run spend audit is `artifacts/cloud_usage_summary.json`.

Google's budget and the local guard should be treated as separate controls:
the budget gives account-level alerts, while the local guard bounds this
experiment's request volume. Neither should be described as a guarantee of
zero overage for arbitrary future jobs.

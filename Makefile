.PHONY: smoke reproduce gcp-code-benchmark gcp-frozen-code lora-code-training compare-backends

smoke:
	PYTHONPATH=src python3 -m staleroll.cli run --seeds 1 --tasks 40 --workers 3 --ticks 60 --output artifacts/smoke

reproduce:
	PYTHONPATH=src python3 -m staleroll.cli run --policy transformer --seeds 3 --tasks 80 --workers 4 --batch-size 4 --max-delay 8 --ticks 120 --output artifacts/latest

gcp-code-benchmark:
	PYTHONPATH=src python3 -m staleroll.cli run --policy vertex --task-domain code --seeds 3 --tasks 10 --workers 2 --ticks 96 --batch-size 4 --max-delay 8 --max-lag 0 --max-kl 0.20 --lag-decay 0.25 --kl-decay 0.80 --vertex-max-requests 40 --vertex-budget-usd 180 --output artifacts/gcp_code_benchmark_tuned

gcp-frozen-code:
	PYTHONPATH=src python3 -m staleroll.cli run --policy vertex_frozen --task-domain code --seeds 2 --tasks 100 --workers 4 --ticks 96 --batch-size 4 --max-delay 8 --max-lag 0 --max-kl 0.20 --lag-decay 0.25 --kl-decay 0.80 --vertex-max-requests 320 --vertex-budget-usd 180 --output artifacts/vertex_frozen_code_final100

lora-code-training:
	PYTHONPATH=src python3 -m staleroll.cli run --policy lora --task-domain code --seeds 3 --tasks 100 --workers 4 --ticks 96 --batch-size 4 --max-delay 8 --max-lag 0 --max-kl 0.20 --lag-decay 0.25 --kl-decay 0.80 --lora-rank 8 --lora-alpha 16 --lora-warmup-epochs 3 --checkpoint-interval 10 --eval-interval 5 --output artifacts/lora_code_training_final2

compare-backends:
	PYTHONPATH=src python3 scripts/compare_backends.py --lora artifacts/lora_code_training_final2 --frozen artifacts/vertex_frozen_code_final100 --output artifacts/backend_comparison_final

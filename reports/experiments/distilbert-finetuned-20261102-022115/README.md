# DistilBERT fine-tuning experiment

Completed all six configurations from the predefined search.

Selected checkpoint:
- Learning rate: 0.00005
- Batch size: 32
- Epoch: 3
- Validation accuracy: 0.2632398753894081
- Validation macro-F1: 0.2653263592190651

These are validation results. Held-out test evaluation is pending.

The source commit is recorded in run_info.json.
The configuration used is preserved in config.snapshot.yaml.

Model and tokenizer files are stored separately in Google Drive:
My Drive/deep-truth/distilbert-finetuned-20261102-022115/artifacts/distilbert_finetuned/model/

Locally, restore that model directory to:
artifacts/distilbert_finetuned/model/

A repeat run selected identical settings and produced identical validation metrics.

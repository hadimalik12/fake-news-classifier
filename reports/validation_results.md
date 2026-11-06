# Model validation results

Compiled November 6, 2026 from the completed local experiment summaries.

**Status: validation results only. Final held-out test evaluation is pending.**

## Evaluation setup

The models classify statement text into the six LIAR truthfulness labels using
the official training split (10,269 statements) and validation split (1,284
statements). All runs use seed 42. Hyperparameters and neural checkpoints are
selected by validation macro-F1 under the shared [experimental protocol](../docs/approach.md).
The results below come from full searches, excluding smoke tests.

Accuracy is reported as a percentage. Precision, recall, and F1 are unweighted
macro averages across all six classes, on a scale from 0 to 1.

## Validation comparison

| Model | Accuracy | Macro-precision | Macro-recall | Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF + logistic regression | 26.56% | 0.2681 | 0.2578 | 0.2592 |
| BiLSTM | 26.25% | 0.2592 | 0.2615 | 0.2597 |
| Frozen DistilBERT | 24.92% | 0.2719 | 0.2522 | 0.2389 |
| Fine-tuned DistilBERT | **27.41%** | **0.2819** | **0.2684** | **0.2719** |

Fine-tuned DistilBERT has the highest validation accuracy and macro-F1. Its
macro-F1 exceeds TF-IDF by 0.0127, BiLSTM by 0.0122, and frozen DistilBERT by
0.0330. BiLSTM has slightly higher macro-F1 than TF-IDF, while TF-IDF has slightly
higher accuracy.

These scores were used for model selection. They do not determine whether the
project hypotheses are supported; those decisions require the held-out test
results specified in the protocol.

## Selected configurations

| Run | Configurations searched | Selected settings | Selected epoch | Device |
| --- | ---: | --- | ---: | --- |
| `tfidf_logreg` | 24 | n-grams: (1, 2); min_df: 2; max_features: 50,000; C: 10.0 | N/A | CPU |
| `bilstm` | 16 | learning rate: 0.001; hidden size per direction: 64; dropout: 0.3; batch size: 64 | 5 | CPU |
| `distilbert_frozen` | 6 | learning rate: 0.0005; batch size: 16; encoder frozen | 7 | Apple MPS |
| `distilbert_finetuned` | 6 | learning rate: 0.00002; batch size: 32; encoder and head trained | 3 | Apple MPS |

## Source records

The values above are taken from `selected.validation` and `selected.parameters`
in these local files:

- `results/tfidf_logreg/summary.json`
- `results/bilstm/summary.json`
- `results/distilbert_frozen/summary.json`
- `results/distilbert_finetuned/summary.json`

The `results/` directory is excluded from Git. This Markdown file preserves the
comparison in a location that can be committed to GitHub.

The earlier [archived CUDA DistilBERT experiment](experiments/distilbert-finetuned-20261102-022115/README.md)
is a separate run: it reports 26.32% validation accuracy and 0.2653 macro-F1.
This comparison uses the local MPS fine-tuned checkpoint, which reports 27.41%
accuracy and 0.2719 macro-F1.

## Remaining final evaluation

Evaluate the four selected checkpoints on the official held-out test split and
report overall metrics, confusion matrices, statement-length comparisons,
frozen-versus-fine-tuned results, hypothesis decisions, and qualitative error
analysis. Final test results are not yet available in this summary.

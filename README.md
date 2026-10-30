# Misinformation and Fake News Classification

**Deep Truth · CS 7643 Deep Learning · Fall 2026**

This project studies how different text classification methods perform on short fact-checked statements. Using the [LIAR dataset](https://aclanthology.org/P17-2067/), we plan to compare a traditional machine learning baseline, a recurrent neural network, and a pretrained transformer under the same data splits and evaluation protocol.

> **Status:** The approach is frozen in [docs/approach.md](docs/approach.md). Data setup, shared inputs, TF-IDF training, and BiLSTM training are implemented. DistilBERT training and final test benchmark results are not implemented yet.

## Setup and data preparation

From a fresh clone, run:

```bash
./scripts/setup.sh
./scripts/download_data.sh
./scripts/prepare_data.sh
```

`setup.sh` creates `.venv` and installs the project. `download_data.sh` downloads the official LIAR files and GloVe 6B 100-dimensional embeddings, verifies the source archive checksums, extracts only the required files, and validates their structure. Downloads are idempotent and stored under `data/`, which is ignored by Git.

`prepare_data.sh` validates the official LIAR splits, fits statement-length settings using the training split only, confirms the saved settings in [config.yaml](config.yaml), and writes `results/data_setup_report.txt`. It also initializes the pretrained DistilBERT tokenizer used to calculate the training token-length percentile.

The data layout is:

```text
data/
├── raw/liar/
│   ├── liar_dataset.zip
│   ├── train.tsv
│   ├── valid.tsv
│   └── test.tsv
└── external/glove/
    ├── glove.6B.zip
    └── glove.6B.100d.txt
```

Shared paths, source URLs, checksums, labels, hyperparameter grids, and derived settings live in [config.yaml](config.yaml).

## Train the TF-IDF baseline

After data preparation, run:

```bash
.venv/bin/python -m deep_truth.train_tfidf
```

This fits all 24 TF-IDF/logistic regression configurations on training statements and selects the best using validation macro-F1, followed by the configured tie-break rules. Validation scores are compared at four decimal places. The test split is never loaded, and the winning pipeline is not refitted on validation.

The saved pipeline is `artifacts/tfidf_logreg/model.joblib`. Search records (`search.json` and `search.csv`), the selected settings and config snapshot (`summary.json`), and validation predictions (`validation_predictions.csv`) are written to `results/tfidf_logreg/`. Re-running training overwrites these outputs. Convergence warnings are displayed and recorded in search results.

Use `--dry-run` to inspect the grid without fitting. Final held-out test evaluation will be implemented separately after all four winning models are selected.

## Train the BiLSTM

```bash
.venv/bin/python -m deep_truth.train_bilstm --smoke-test
.venv/bin/python -m deep_truth.train_bilstm
```

The smoke test runs one configuration for one epoch and writes only to `artifacts/bilstm_smoke/` and `results/bilstm_smoke/`. It verifies the training path; its scores are not a completed experiment.

The full run searches all 16 configured settings, up to 15 epochs each, with patience 3. It uses trainable GloVe embeddings, a single bidirectional LSTM with packed sequences to exclude padding, dropout, and a six-class head. Training uses cross-entropy, Adam, and gradient clipping. Validation macro-F1 selects checkpoints using the shared tie-break rules; a separate minimum-improvement threshold controls early stopping. The test split is never loaded.

The winning checkpoint at `artifacts/bilstm/model.pt` contains model weights, vocabulary, selected parameters, epoch, validation metrics, and the config snapshot. Epoch histories and search results, a summary, and validation predictions go in `results/bilstm/`. The saved winner is reloaded and its validation metrics verified before export. Re-running overwrites those outputs.

Automatic device selection uses CUDA when available and CPU otherwise (including on Macs). Override with `--device cpu` or `--device cuda`; `--threads` controls CPU threads (default 4). Use `--dry-run` to inspect the grid without training. The full search can take substantially longer than TF-IDF.

## Shared model inputs

The shared data layer constructs model-specific views over the same validated examples:

```python
from deep_truth.config import load_config
from deep_truth.data.loaders import (
    make_loaders,
    prepare_bilstm_data,
    prepare_distilbert_data,
    prepare_tfidf_data,
)

config = load_config()
tfidf_data = prepare_tfidf_data(config)
bilstm_data = prepare_bilstm_data(config)
distilbert_data = prepare_distilbert_data(config)

bilstm_loaders = make_loaders(
    bilstm_data.datasets,
    batch_size=32,
    seed=config.seed,
)
```

Training and validation are loaded by default. The held-out test split must be requested explicitly with `include_test=True`. BiLSTM vocabulary construction and GloVe alignment use training statements only. Each dataset keeps an aligned metadata table containing the statement ID, raw text, word count, length bin, and string label for later prediction export.

Run the data-interface tests with:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Repository layout

```text
scripts/                 Reproducible environment and data commands
src/deep_truth/data/     Downloading, validation, vocabularies, and model inputs
src/deep_truth/          Existing shared configuration, metrics, and run skeletons
docs/                    Frozen experimental approach
data/                    Downloaded source data (not committed)
artifacts/               Future model artifacts (not committed)
results/                 Generated reports and results (not committed)
```

## Research questions

1. **Model comparison:** How does a fine-tuned DistilBERT classifier compare with TF-IDF plus logistic regression and an LSTM or BiLSTM on six class LIAR statement classification?
2. **Statement length:** How does statement length affect each model's performance, and does the relative ranking of the models change across short, medium, and long statements?
3. **Task-specific fine-tuning:** How much does fully fine-tuning DistilBERT improve performance over keeping its encoder frozen and training only the classification head?

Our hypothesis is that fine-tuned DistilBERT will perform best overall because its pretrained representations capture contextual relationships that a bag-of-words model does not explicitly represent and that a recurrent model must learn from a relatively small task dataset.

These questions are operationalized as three testable hypotheses, with the data, model, tuning, and evaluation protocol fixed, in [docs/approach.md](docs/approach.md).

## Dataset and task

LIAR contains **12,836** short English-language statements collected from PolitiFact between 2007 and 2016. We will use its official splits: **10,269 training**, **1,284 validation**, and **1,283 test** examples. Those files are downloaded locally with `./scripts/download_data.sh` and are not committed. Each model will predict one of six truthfulness labels:

`pants-fire` · `false` · `barely-true` · `half-true` · `mostly-true` · `true`

The primary input is the **statement text only**. LIAR also provides speaker, subject, party, context, and fact-checking history metadata, but we will exclude those fields so that all models receive the same information. Model-specific vocabularies and preprocessing settings will be fitted on the training split to avoid test-data leakage.

## Planned methods

| Model | Statement representation | Classifier |
| --- | --- | --- |
| TF-IDF + logistic regression | TF-IDF features with a vocabulary fitted on training text | Multinomial logistic regression baseline |
| LSTM or BiLSTM | Tokenized statements mapped to word indices or embeddings and padded to a consistent length | Recurrent encoder followed by a six class classification layer |
| DistilBERT | Pretrained tokenizer producing token IDs and attention masks | Pretrained encoder with a six class classification head |

The neural models will use multiclass cross-entropy loss. We will select learning rate, batch size, and training duration using validation performance. DistilBERT will be evaluated both with a frozen encoder and with full task-specific fine-tuning.

## Planned evaluation

All models will use the same held-out test split. We will report **accuracy, precision, recall, and F1 score**, along with confusion matrices. The planned analyses are:

1. **Model comparison:** Compare the three model families under the same dataset splits and metrics.
2. **Statement length:** Group test statements into short, medium, and long subsets and compare performance across groups.
3. **Fine-tuning ablation:** Compare a frozen DistilBERT encoder, with only its classification head trained, against full fine-tuning.
4. **Error analysis:** Inspect misclassifications and examples on which the models disagree.

No performance claims are available yet. Results and reproducible run instructions will be added as the project progresses.

## Responsible use and limitations

These models will learn patterns from LIAR's labels and fact-checking process. A prediction is **not** an independent determination of a statement's objective truth, and performance on LIAR may not transfer to other topics, time periods, or sources. False positives may cast doubt on reliable statements, while false negatives may miss misleading ones. We will use multiple evaluation metrics, inspect errors, and keep conclusions within the scope of the benchmark.

## Team

| Member | Proposal responsibilities |
| --- | --- |
| Hadi Malik | Research, model tuning, testing |
| Yameen Ahmed | Approach development, model tuning, testing |
| Farhan Sharif | Research, model tuning, testing |
| Aly Aadil Salewala | Ethical considerations, model tuning, testing |

## References

- Wang, W. Y. (2017). [“Liar, Liar Pants on Fire”: A New Benchmark Dataset for Fake News Detection](https://aclanthology.org/P17-2067/).
- Long, Y., Lu, Q., Xiang, R., Li, M., & Huang, C.-R. (2017). [Fake News Detection Through Multi-Perspective Speaker Profiles](https://aclanthology.org/I17-2043/).
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423/).
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). [DistilBERT, a Distilled Version of BERT: Smaller, Faster, Cheaper and Lighter](https://arxiv.org/abs/1910.01108).

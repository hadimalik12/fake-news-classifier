
## Hypotheses

All three hypotheses use **macro-F1 on the official LIAR test split**. Macro-F1 is the unweighted mean of the six per-class F1 scores. A class with no predicted positives contributes precision 0 and a class with no true positives contributes recall 0 (`zero_division = 0`). Scores are compared at four decimal places. A tie is a failure of that hypothesis: support requires a strictly greater value.

Validation macro-F1 selects checkpoints. It does not decide the hypotheses.

1. **Model comparison.** Fully fine-tuned DistilBERT will have higher test macro-F1 than TF-IDF plus logistic regression and higher test macro-F1 than the BiLSTM. Statement text is the only input.
2. **Statement length.** Fully fine-tuned DistilBERT’s macro-F1 advantage over each baseline will be larger on long statements than on short statements. For a baseline `b` and a length bin `g`:

   ```
   gap(b, g) = macro-F1(DistilBERT full, g) - macro-F1(b, g)
   ```

   The hypothesis is supported when both of these are true:

   ```
   gap(TF-IDF, long) > gap(TF-IDF, short)
   gap(BiLSTM, long) > gap(BiLSTM, short)
   ```

   The medium bin is reported and is not part of the decision.
3. **Fine-tuning.** Fully fine-tuned DistilBERT will have higher test macro-F1 than the same DistilBERT with the encoder frozen and only the classification head trained.

## Data

Use the official LIAR release files:

| Split | File | Examples |
| --- | --- | --- |
| Train | `data/liar/train.tsv` | 10,269 |
| Validation | `data/liar/valid.tsv` | 1,284 |
| Test | `data/liar/test.tsv` | 1,283 |

These TSV files are local data, not repository files. From the repository root, `python -m deep_truth.prepare_data` downloads the official archive URL in `config.yaml` and writes the three splits into `data/liar/`. Run it once on each machine. The length cutoffs already stored in `config.yaml` were fit on the training split, and this command checks that a new download reproduces them.

The files are tab-separated and have no header. Read these columns only:

| Index | Field | Use |
| --- | --- | --- |
| 0 | statement id | stored with predictions |
| 1 | label | target |
| 2 | statement | model input and length |

Ignore subject, speaker, job, state, party, credit-history counts, and context. Do not concatenate them onto the statement.

Label order, and the integer index used by every model:

| Index | Label |
| --- | --- |
| 0 | `pants-fire` |
| 1 | `false` |
| 2 | `barely-true` |
| 3 | `half-true` |
| 4 | `mostly-true` |
| 5 | `true` |

**Word count** for length bins and for BiLSTM padding is `len(statement.split())` on the raw statement string.

**Length bins** are equal-frequency tertiles of training word counts. Compute `q1` and `q2` with NumPy `quantile` at probabilities `1/3` and `2/3` and `method="linear"`. Assign:

- `short`: word count <= `q1`
- `medium`: `q1` < word count <= `q2`
- `long`: word count > `q2`

Write `q1` and `q2` into the shared config during data setup. Apply those saved numbers to validation and test. Do not recompute them on those splits. Length bins use whitespace word counts for every model, including DistilBERT.

**BiLSTM pad length** is the ceiling of the 95th percentile of training word counts, with a floor of 8. Truncate longer sequences by dropping tokens from the end. Pad shorter sequences at the end.

Vocabularies, the TF-IDF vectorizer, pad length, and length cutpoints are fit on training text only.

## Models

Four checkpoints are compared. Names are fixed so prediction files match.

| Run name | Model |
| --- | --- |
| `tfidf_logreg` | TF-IDF plus multinomial logistic regression |
| `bilstm` | BiLSTM with a six-class linear head |
| `distilbert_frozen` | `distilbert-base-uncased`, encoder frozen |
| `distilbert_finetuned` | `distilbert-base-uncased`, encoder and head trained |

### TF-IDF plus logistic regression

`TfidfVectorizer` (lowercase, word analyzer, default token pattern, L2 norm, `use_idf=True`) inside a scikit-learn pipeline, followed by `LogisticRegression(solver="lbfgs", max_iter=2000, random_state=42)`. For six classes, `lbfgs` fits the multinomial objective. Searched settings are listed under Tuning. `class_weight` stays at the default.

### BiLSTM

Tokenizer: lowercase the statement, then split on whitespace. Build the vocabulary on training tokens with count at least 2. Sort the vocabulary by descending count, breaking ties alphabetically.

| Index | Token |
| --- | --- |
| 0 | `<pad>` |
| 1 | `<unk>` |
| 2+ | training vocabulary |

Initialize the embedding with [GloVe 6B 100d](https://nlp.stanford.edu/data/glove.6B.zip) (`glove.6B.100d.txt`) for in-vocabulary words. Draw out-of-vocabulary rows, including `<unk>`, from a normal distribution with mean 0 and standard deviation 0.05, under the global seed. Keep the `<pad>` row at zero (`padding_idx=0`). Embeddings are trainable.

Architecture: embedding (dimension 100), one bidirectional LSTM, dropout, then a linear layer to 6 logits. Hidden size in the search grid is the size in each direction. Optimizer is Adam with weight decay 0. Clip gradients to max norm 1.0.

### DistilBERT

Use `DistilBertForSequenceClassification` with `num_labels=6` and `problem_type="single_label_classification"`. That head reads the first-token hidden state. Set `id2label` and `label2id` from the label table.

`max_length` is chosen once from the training set during data setup: the 99th percentile of `distilbert-base-uncased` tokenized lengths, rounded up, then snapped to 64 if that percentile is at most 64 and to 128 otherwise. Store the chosen value in config. Both DistilBERT runs use it.

The frozen run sets `requires_grad = False` on every parameter in the `distilbert` encoder and trains the classification head. The fine-tuned run trains the encoder and the head. Tokenizer, `max_length`, head architecture, loss, seed, and splits otherwise match.

Optimizer is AdamW with weight decay 0.01. Clip gradients to max norm 1.0. The fine-tuned run uses a linear learning-rate schedule with warmup over the first 10% of steps. The frozen run uses a constant learning rate.

## Loss and seed

Neural models minimize the mean multiclass cross-entropy from the proposal. For a batch of `N` examples and six classes, with `y[i,c]` the true label indicator and `y_hat[i,c]` the predicted probability:

```
L = -(1/N) * sum_i sum_c y[i,c] * log(y_hat[i,c])
```

Implement it as `CrossEntropyLoss` (mean reduction) on logits. Apply softmax inside that loss, not before it.

Global seed is **42** for Python, NumPy, and PyTorch, including CUDA when it is used. Train one seed. Do not average additional seeds into the hypothesis test.

## Tuning

Search only the grids below. Select the checkpoint with the highest validation macro-F1. Break ties in order: higher validation accuracy, smaller learning rate, smaller batch size, then the simpler capacity setting (fewer `max_features`, unigrams before bigrams, smaller `min_df`, smaller `C`, smaller hidden size, smaller dropout).

Early stopping restores the epoch with the best validation macro-F1. Patience counts epochs without an improvement of at least 0.0001 macro-F1.

### `tfidf_logreg`

| Parameter | Values |
| --- | --- |
| `ngram_range` | `(1, 1)`, `(1, 2)` |
| `min_df` | `2`, `5` |
| `max_features` | `10000`, `50000` |
| `C` | `0.1`, `1.0`, `10.0` |

### `bilstm`

| Parameter | Values |
| --- | --- |
| learning rate | `1e-3`, `1e-4` |
| hidden size per direction | `64`, `128` |
| dropout | `0.3`, `0.5` |
| batch size | `32`, `64` |
| max epochs | `15` |
| patience | `3` |

### `distilbert_frozen`

| Parameter | Values |
| --- | --- |
| learning rate | `1e-3`, `5e-4`, `1e-4` |
| batch size | `16`, `32` |
| max epochs | `10` |
| patience | `2` |

### `distilbert_finetuned`

| Parameter | Values |
| --- | --- |
| learning rate | `2e-5`, `3e-5`, `5e-5` |
| batch size | `16`, `32` |
| max epochs | `4` |
| patience | `2` |

Record every cell’s hyperparameters and validation macro-F1. The test split is not loaded during tuning.

## Test evaluation

After the four winning checkpoints are chosen, score each one once on the official test split. Hypothesis decisions use those test scores.

Save one prediction table per run at `results/predictions/<run_name>.csv` with columns:

`id`, `statement`, `word_count`, `length_bin`, `label`, `prediction`

`label` and `prediction` are the label strings, not integer indexes.

From those files, write three tables:

1. **Hypothesis 1.** Accuracy, macro-precision, macro-recall, and macro-F1 for `tfidf_logreg`, `bilstm`, and `distilbert_finetuned`, plus a confusion matrix for each. Matrix rows are true labels and columns are predicted labels, in the label order above.
2. **Hypothesis 2.** Macro-F1 for those three runs on `short`, `medium`, and `long`, with the number of test statements in each bin, and the two gaps defined above.
3. **Hypothesis 3.** The same overall metrics for `distilbert_frozen` and `distilbert_finetuned`, plus per-class F1.

Each results table states the hypothesis as **supported** or **not supported** under the decision rules in this document. Validation scores that selected the checkpoints are reported next to the test scores. Regenerating the tables must be possible from the prediction files alone.

## Error-analysis inputs

Using the same prediction files, prepare counts and a short example list for the report:

- statements whose predicted label differs across `tfidf_logreg`, `bilstm`, and `distilbert_finetuned`
- errors between adjacent labels (`false`/`barely-true`, `barely-true`/`half-true`, `half-true`/`mostly-true`, `mostly-true`/`true`, and `pants-fire`/`false`)
- statements `distilbert_finetuned` misses and `tfidf_logreg` gets right

Break the counts out by length bin. This pack is input to the qualitative discussion and to the ethics write-up. It does not trigger another training run.

## Report order

The final report places Related Work before Approach, as requested on the graded proposal. The Approach section quotes the three hypotheses and the protocol in this file, then the selected hyperparameters, their validation macro-F1, and the three test tables.

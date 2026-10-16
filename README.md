# Misinformation and Fake News Classification

**Deep Truth · CS 7643 Deep Learning · Fall 2026**

This project studies how different text classification methods perform on short fact-checked statements. Using the [LIAR dataset](https://aclanthology.org/P17-2067/), we plan to compare a traditional machine learning baseline, a recurrent neural network, and a pretrained transformer under the same data splits and evaluation protocol.

> **Status:** Project proposal. The models, experiments, and results described below are planned; this repository does not yet contain a completed implementation or benchmark results.

## Research question

Can a fine-tuned DistilBERT model improve six class misinformation classification over TF-IDF with logistic regression and an LSTM or BiLSTM? We will also examine how statement length and transformer fine-tuning affect performance.

Our hypothesis is that fine-tuned DistilBERT will perform best overall because its pretrained representations capture contextual relationships that a bag-of-words model does not explicitly represent and that a recurrent model must learn from a relatively small task dataset.

## Dataset and task

LIAR contains **12,836** short English-language statements collected from PolitiFact between 2007 and 2016. We will use its official splits: **10,269 training**, **1,284 validation**, and **1,283 test** examples. Each model will predict one of six truthfulness labels:

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

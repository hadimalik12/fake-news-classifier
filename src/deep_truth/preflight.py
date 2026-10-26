"""Check that the skeleton matches the frozen approach and that scripts read config.yaml."""

from __future__ import annotations

import math

from deep_truth.config import grid_size, iter_grid, load_config, parse_probability
from deep_truth.data import assign_length_bin, read_split, training_length_statistics, word_count
from deep_truth.metrics import (
    compute_metrics,
    hypothesis_1_supported,
    hypothesis_2_supported,
    hypothesis_3_supported,
)
from deep_truth.train_bilstm import main as bilstm_main
from deep_truth.train_distilbert import main as distilbert_main
from deep_truth.train_tfidf import main as tfidf_main


def check_derived_settings(config) -> None:
    cutpoints = config.length_cutpoints
    if cutpoints is None:
        raise AssertionError("q1 and q2 are empty. Run python -m deep_truth.prepare_data.")
    q1, q2 = cutpoints
    if not q1 < q2:
        raise AssertionError("q1 must be below q2.")
    pad_length = config.data["bilstm"]["pad_length"]
    floor = int(config.data["bilstm"]["pad_length_floor"])
    if not isinstance(pad_length, int) or pad_length < floor:
        raise AssertionError("pad_length must be an integer at least the configured floor.")
    allowed = {
        int(config.data["distilbert"]["max_length_if_percentile_at_most_64"]),
        int(config.data["distilbert"]["max_length_otherwise"]),
    }
    if config.data["distilbert"]["max_length"] not in allowed:
        raise AssertionError(f"DistilBERT max_length must be one of {sorted(allowed)}.")
    train_path = config.split_path("train")
    if not train_path.is_file():
        return
    frame = read_split(config, "train")
    stats = training_length_statistics(frame["word_count"].to_numpy(), config)
    if not math.isclose(stats["q1"], q1, abs_tol=1e-6) or not math.isclose(stats["q2"], q2, abs_tol=1e-6):
        raise AssertionError("Saved length cutpoints do not match the training split.")
    if stats["pad_length"] != pad_length:
        raise AssertionError("Saved BiLSTM pad length does not match the training split.")


def check_config() -> None:
    config = load_config()
    labels = config.labels
    if labels != [
        "pants-fire",
        "false",
        "barely-true",
        "half-true",
        "mostly-true",
        "true",
    ]:
        raise AssertionError(f"Unexpected label order: {labels}")
    if config.label_to_index["true"] != 5:
        raise AssertionError("Label indexes must follow the order in config.yaml.")
    if config.seed != 42:
        raise AssertionError("Seed in config.yaml must stay 42.")
    if [parse_probability(value) for value in config.data["length_bins"]["probabilities"]] != [
        1 / 3,
        2 / 3,
    ]:
        raise AssertionError("Length tertiles must use probabilities 1/3 and 2/3.")
    check_derived_settings(config)

    sizes = {
        "tfidf_logreg": grid_size(config.data["tfidf"]["grid"]),
        "bilstm": grid_size(config.data["bilstm"]["grid"]),
        "distilbert_frozen": grid_size(config.data["distilbert"]["frozen"]["grid"]),
        "distilbert_finetuned": grid_size(config.data["distilbert"]["finetuned"]["grid"]),
    }
    expected = {
        "tfidf_logreg": 24,
        "bilstm": 16,
        "distilbert_frozen": 6,
        "distilbert_finetuned": 6,
    }
    if sizes != expected:
        raise AssertionError(f"Grid sizes {sizes} do not match the approach spec {expected}.")
    tfidf_grid = list(iter_grid(config.data["tfidf"]["grid"]))
    if len(tfidf_grid) != 24 or tfidf_grid[0]["ngram_range"] != (1, 1):
        raise AssertionError("TF-IDF grid did not expand to the 24 settings in the approach spec.")

    for split in ("train", "validation", "test"):
        path = config.split_path(split)
        if path.suffix != ".tsv" or path.parent.name != "liar":
            raise AssertionError(f"Unexpected path for {split}: {path}")


def check_length_and_metrics() -> None:
    if word_count("  one   two three ") != 3:
        raise AssertionError("Word count must use whitespace splitting.")
    if [assign_length_bin(count, 10, 20) for count in (10, 11, 20, 21)] != [
        "short",
        "medium",
        "medium",
        "long",
    ]:
        raise AssertionError("Length bin edges do not match the approach spec.")

    labels = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
    scores = compute_metrics(
        ["pants-fire", "false", "false", "true"],
        ["pants-fire", "false", "true", "true"],
        labels,
    )
    if abs(scores["accuracy"] - 0.75) > 1e-12:
        raise AssertionError(f"Unexpected accuracy {scores['accuracy']}.")
    expected_macro_f1 = (1 + (2 / 3) + 0 + 0 + 0 + (2 / 3)) / 6
    if abs(scores["macro_f1"] - expected_macro_f1) > 1e-12:
        raise AssertionError(f"Unexpected macro-F1 {scores['macro_f1']}.")
    if scores["confusion_matrix"][0][0] != 1 or scores["confusion_matrix"][1][5] != 1:
        raise AssertionError("Confusion matrix is not ordered with true labels on the rows.")

    decimals = 4
    if not hypothesis_1_supported(0.30, 0.20, 0.25, decimals):
        raise AssertionError("Hypothesis 1 should be supported when full fine-tuning is strictly ahead.")
    if hypothesis_1_supported(0.20001, 0.20, 0.10, decimals):
        raise AssertionError("Hypothesis 1 must treat a four-decimal tie as not supported.")
    if not hypothesis_2_supported(
        {"short": 0.20, "long": 0.40},
        {"short": 0.18, "long": 0.25},
        {"short": 0.19, "long": 0.27},
        decimals,
    ):
        raise AssertionError("Hypothesis 2 should be supported when both long gaps are larger.")
    if hypothesis_3_supported(0.25, 0.25, decimals):
        raise AssertionError("Hypothesis 3 must treat an equal score as not supported.")


def main() -> None:
    check_config()
    check_length_and_metrics()
    tfidf_main()
    bilstm_main()
    distilbert_main(["--run", "distilbert_frozen"])
    distilbert_main(["--run", "distilbert_finetuned"])
    print("Skeleton checks passed.")


if __name__ == "__main__":
    main()

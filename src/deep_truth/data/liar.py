"""LIAR loading and length bins.

Column indexes, split paths, and expected sizes come from config.yaml.
q1 and q2 are read back from that file after data setup writes them.
"""

from __future__ import annotations

import csv
import math

import numpy as np
import pandas as pd

from deep_truth.config import Config, parse_probability


def word_count(statement: str) -> int:
    """Return the whitespace-delimited word count used by the experiment."""
    return len(statement.split())


def assign_length_bin(count: int, q1: float, q2: float) -> str:
    if count <= q1:
        return "short"
    if count <= q2:
        return "medium"
    return "long"


def read_split(config: Config, split: str) -> pd.DataFrame:
    """Read one official LIAR split, keeping id, label, and statement."""
    path = config.split_path(split)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. From the repository root, run ./scripts/download_data.sh "
            "and then ./scripts/prepare_data.sh."
        )
    columns = config.data["columns"]
    frame = pd.read_csv(
        path,
        sep="\t",
        header=None,
        dtype=str,
        keep_default_na=False,
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
    )
    needed = [columns["id"], columns["label"], columns["statement"]]
    if frame.shape[1] <= max(needed):
        raise ValueError(f"{path} has {frame.shape[1]} columns; expected the LIAR TSV layout.")
    frame = frame.iloc[:, needed].copy()
    frame.columns = ["id", "label", "statement"]
    frame["word_count"] = frame["statement"].map(word_count)
    return frame


def check_split(config: Config, split: str, frame: pd.DataFrame) -> None:
    expected = config.split_examples(split)
    if len(frame) != expected:
        raise ValueError(f"{split} has {len(frame)} rows; config.yaml expects {expected}.")
    unknown = sorted(set(frame["label"]) - set(config.labels))
    if unknown:
        raise ValueError(f"{split} contains labels that are not in config.yaml: {unknown}")


def training_length_statistics(word_counts: np.ndarray, config: Config) -> dict[str, float | int]:
    """Fit length cutpoints and the BiLSTM pad length on training word counts only."""
    method = config.data["length_bins"]["quantile_method"]
    probabilities = [parse_probability(value) for value in config.data["length_bins"]["probabilities"]]
    q1, q2 = np.quantile(np.asarray(word_counts), probabilities, method=method)
    pad_probability = float(config.data["bilstm"]["pad_percentile"]) / 100
    pad_raw = float(np.quantile(np.asarray(word_counts), pad_probability, method=method))
    pad_length = max(int(config.data["bilstm"]["pad_length_floor"]), math.ceil(pad_raw))
    return {
        "q1": float(q1),
        "q2": float(q2),
        "pad_percentile_value": pad_raw,
        "pad_length": int(pad_length),
    }


def add_length_bins(config: Config, frame: pd.DataFrame) -> pd.DataFrame:
    """Copy a frame and add length_bin from the cutpoints stored in config."""
    q1, q2 = config.require_length_cutpoints()
    binned = frame.copy()
    if "word_count" not in binned.columns:
        binned["word_count"] = binned["statement"].map(word_count)
    binned["length_bin"] = binned["word_count"].map(
        lambda count: assign_length_bin(int(count), q1, q2)
    )
    return binned

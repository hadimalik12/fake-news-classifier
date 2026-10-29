"""Validate downloaded data and write training-only length settings into config.yaml."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from deep_truth.config import load_config
from deep_truth.data.download import ensure_data_files
from deep_truth.data.liar import (
    add_length_bins,
    check_split,
    read_split,
    training_length_statistics,
)


def distilbert_max_length(statements: list[str], config) -> dict[str, float | int]:
    """Choose 64 or 128 from the training-set token-length percentile."""
    model_name = config.data["distilbert"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lengths: list[int] = []
    batch_size = 256
    for start in range(0, len(statements), batch_size):
        encoded = tokenizer(
            statements[start : start + batch_size],
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
        lengths.extend(len(token_ids) for token_ids in encoded["input_ids"])
    method = config.data["length_bins"]["quantile_method"]
    probability = float(config.data["distilbert"]["max_length_percentile"]) / 100
    raw = float(np.quantile(np.asarray(lengths), probability, method=method))
    rounded = math.ceil(raw)
    short_cap = int(config.data["distilbert"]["max_length_if_percentile_at_most_64"])
    long_cap = int(config.data["distilbert"]["max_length_otherwise"])
    chosen = short_cap if rounded <= short_cap else long_cap
    return {"token_percentile": raw, "token_percentile_ceil": int(rounded), "max_length": int(chosen)}


def format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return format(float(value), ".10g")


def write_derived_settings(config_path: Path, q1: float, q2: float, pad_length: int, max_length: int) -> None:
    raw = config_path.read_bytes()
    newline = "\r\n" if b"\r\n" in raw else "\n"
    text = raw.decode("utf-8").replace("\r\n", "\n")
    updates = {
        "q1": format_number(q1),
        "q2": format_number(q2),
        "pad_length": str(int(pad_length)),
        "max_length": str(int(max_length)),
    }
    for key, value in updates.items():
        pattern = re.compile(rf"^(\s*{re.escape(key)}:\s*).*$", re.MULTILINE)
        text, count = pattern.subn(rf"\g<1>{value}", text, count=1)
        if count != 1:
            raise RuntimeError(f"Expected exactly one {key} entry in {config_path}.")
    config_path.write_bytes(text.replace("\n", newline).encode("utf-8"))


def count_labels(frame, labels: list[str]) -> dict[str, int]:
    counts = frame["label"].value_counts()
    return {label: int(counts.get(label, 0)) for label in labels}


def build_report(config, splits: dict, length_stats: dict, token_stats: dict) -> str:
    lines = ["LIAR data setup", "Cutpoints were fit on training statements only.", ""]
    for split, frame in splits.items():
        lines.append(f"{split}: {len(frame)} rows")
    lines.append("")
    lines.append("label counts (train / validation / test)")
    for label in config.labels:
        counts = [str(count_labels(splits[split], config.labels)[label]) for split in ("train", "validation", "test")]
        lines.append(f"  {label}: {' / '.join(counts)}")
    train_counts = splits["train"]["word_count"]
    lines.extend(
        [
            "",
            f"train word count: min {int(train_counts.min())}, median {float(train_counts.median()):.1f}, max {int(train_counts.max())}",
            f"q1: {length_stats['q1']}",
            f"q2: {length_stats['q2']}",
            f"BiLSTM pad percentile value: {length_stats['pad_percentile_value']}",
            f"BiLSTM pad length: {length_stats['pad_length']}",
            f"DistilBERT training token percentile: {token_stats['token_percentile']}",
            f"DistilBERT max_length: {token_stats['max_length']}",
            "",
            "length bins using the saved training cutpoints",
        ]
    )
    for split, frame in splits.items():
        binned = add_length_bins(config, frame)
        counts = binned["length_bin"].value_counts()
        rendered = ", ".join(f"{name} {int(counts.get(name, 0))}" for name in config.data["length_bins"]["names"])
        lines.append(f"  {split}: {rendered}")
    return "\n".join(lines) + "\n"


def main() -> None:
    config = load_config()
    ensure_data_files(config)
    splits = {split: read_split(config, split) for split in ("train", "validation", "test")}
    for split, frame in splits.items():
        check_split(config, split, frame)
    length_stats = training_length_statistics(splits["train"]["word_count"].to_numpy(), config)
    token_stats = distilbert_max_length(splits["train"]["statement"].tolist(), config)
    write_derived_settings(
        config.root / "config.yaml",
        q1=length_stats["q1"],
        q2=length_stats["q2"],
        pad_length=int(length_stats["pad_length"]),
        max_length=int(token_stats["max_length"]),
    )
    config = load_config()
    saved_q1, saved_q2 = config.require_length_cutpoints()
    if not math.isclose(saved_q1, length_stats["q1"], abs_tol=1e-8):
        raise RuntimeError("config.yaml q1 did not round-trip.")
    if not math.isclose(saved_q2, length_stats["q2"], abs_tol=1e-8):
        raise RuntimeError("config.yaml q2 did not round-trip.")
    if config.data["bilstm"]["pad_length"] != length_stats["pad_length"]:
        raise RuntimeError("config.yaml pad_length did not round-trip.")
    if config.data["distilbert"]["max_length"] != token_stats["max_length"]:
        raise RuntimeError("config.yaml max_length did not round-trip.")
    if not all(isinstance(label, str) for label in config.labels):
        raise RuntimeError("Writing config.yaml changed the labels away from strings.")
    report = build_report(config, splits, length_stats, token_stats)
    report_path = config.root / "results" / "data_setup_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()

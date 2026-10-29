from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch

from deep_truth.config import Config, load_config


def write_split(path: Path, rows: list[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{identifier}\t{label}\t{statement}\n" for identifier, label, statement in rows),
        encoding="utf-8",
    )


def make_config(
    root: Path,
    train: list[tuple[str, str, str]],
    validation: list[tuple[str, str, str]],
    test: list[tuple[str, str, str]] | None = None,
) -> Config:
    data = deepcopy(load_config().data)
    rows_by_split = {"train": train, "validation": validation, "test": test or []}
    filenames = {"train": "train.tsv", "validation": "valid.tsv", "test": "test.tsv"}
    for split, rows in rows_by_split.items():
        relative = Path("data") / filenames[split]
        data["splits"][split]["path"] = str(relative)
        data["splits"][split]["examples"] = len(rows)
        write_split(root / relative, rows)
    data["length_bins"]["q1"] = 1
    data["length_bins"]["q2"] = 2
    data["bilstm"]["min_token_count"] = 1
    data["bilstm"]["pad_length"] = 4
    data["bilstm"]["embedding_dim"] = 3
    data["embeddings"]["glove"]["dimensions"] = 3
    data["embeddings"]["glove"]["path"] = "data/glove.txt"
    data["distilbert"]["max_length"] = 5
    return Config(root=root, data=data)


class FakeTokenizer:
    def __call__(
        self,
        statements: list[str],
        padding: str,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        assert padding == "max_length"
        assert truncation is True
        assert return_tensors == "pt"
        all_ids: list[list[int]] = []
        all_masks: list[list[int]] = []
        for statement in statements:
            ids = [101, *(10 + len(token) for token in statement.split()), 102][:max_length]
            mask = [1] * len(ids)
            padding_count = max_length - len(ids)
            all_ids.append(ids + [0] * padding_count)
            all_masks.append(mask + [0] * padding_count)
        return {
            "input_ids": torch.tensor(all_ids, dtype=torch.long),
            "attention_mask": torch.tensor(all_masks, dtype=torch.long),
        }

"""Model-specific views over the same validated LIAR examples."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from deep_truth.config import Config
from deep_truth.data.vocabulary import Vocabulary

METADATA_COLUMNS = ["id", "statement", "word_count", "length_bin", "label"]


def copy_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(METADATA_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Frame is missing metadata columns: {missing}.")
    return frame.loc[:, METADATA_COLUMNS].reset_index(drop=True).copy()


class TfidfDataset:
    """Raw statements and integer labels for scikit-learn estimators."""

    def __init__(self, frame: pd.DataFrame, config: Config) -> None:
        self.metadata = copy_metadata(frame)
        self.statements = self.metadata["statement"].tolist()
        self.labels = np.asarray(
            [config.label_to_index[label] for label in self.metadata["label"]],
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return len(self.metadata)


class BiLSTMDataset(Dataset):
    """Padded word-index tensors backed by aligned example metadata."""

    def __init__(self, frame: pd.DataFrame, config: Config, vocabulary: Vocabulary) -> None:
        self.metadata = copy_metadata(frame)
        self.pad_length = int(config.data["bilstm"]["pad_length"])
        if self.pad_length < 1:
            raise ValueError("bilstm.pad_length must be a positive integer.")
        self.input_ids = torch.full(
            (len(self.metadata), self.pad_length),
            fill_value=vocabulary.pad_index,
            dtype=torch.long,
        )
        lengths: list[int] = []
        for row_index, statement in enumerate(self.metadata["statement"]):
            encoded = vocabulary.encode(statement, max_length=self.pad_length)
            length = len(encoded)
            lengths.append(length)
            if encoded:
                self.input_ids[row_index, :length] = torch.tensor(encoded, dtype=torch.long)
        self.lengths = torch.tensor(lengths, dtype=torch.long)
        self.labels = torch.tensor(
            [config.label_to_index[label] for label in self.metadata["label"]],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "length": self.lengths[index],
            "labels": self.labels[index],
            "example_index": torch.tensor(index, dtype=torch.long),
        }


class DistilBertDataset(Dataset):
    """Fixed-length tokenizer outputs backed by aligned example metadata."""

    def __init__(self, frame: pd.DataFrame, config: Config, tokenizer: Any) -> None:
        self.metadata = copy_metadata(frame)
        self.max_length = int(config.data["distilbert"]["max_length"])
        encoded = tokenizer(
            self.metadata["statement"].tolist(),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        for key in ("input_ids", "attention_mask"):
            if key not in encoded:
                raise ValueError(f"Tokenizer output is missing {key}.")
            value = encoded[key]
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, dtype=torch.long)
            if value.shape != (len(self.metadata), self.max_length):
                raise ValueError(
                    f"Tokenizer {key} shape is {tuple(value.shape)}; expected "
                    f"{(len(self.metadata), self.max_length)}."
                )
            setattr(self, key, value.to(dtype=torch.long))
        self.labels = torch.tensor(
            [config.label_to_index[label] for label in self.metadata["label"]],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "labels": self.labels[index],
            "example_index": torch.tensor(index, dtype=torch.long),
        }

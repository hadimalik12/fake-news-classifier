"""Shared dataset and DataLoader construction for every planned model family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from deep_truth.config import Config
from deep_truth.data.datasets import BiLSTMDataset, DistilBertDataset, TfidfDataset
from deep_truth.data.embeddings import EmbeddingMatrix, load_glove_matrix
from deep_truth.data.liar import add_length_bins, check_split, read_split
from deep_truth.data.vocabulary import Vocabulary

DatasetType = TypeVar("DatasetType")


@dataclass(frozen=True)
class SplitBundle(Generic[DatasetType]):
    train: DatasetType
    validation: DatasetType
    test: DatasetType | None = None


@dataclass(frozen=True)
class LoaderBundle:
    train: DataLoader
    validation: DataLoader
    test: DataLoader | None = None


@dataclass(frozen=True)
class BiLSTMData:
    datasets: SplitBundle[BiLSTMDataset]
    vocabulary: Vocabulary
    embeddings: EmbeddingMatrix


def load_split_frames(config: Config, include_test: bool = False) -> SplitBundle[pd.DataFrame]:
    """Load only requested official splits and attach training-derived length bins."""
    names = ["train", "validation"]
    if include_test:
        names.append("test")
    frames: dict[str, pd.DataFrame] = {}
    for name in names:
        frame = read_split(config, name)
        check_split(config, name, frame)
        frames[name] = add_length_bins(config, frame)
    return SplitBundle(
        train=frames["train"],
        validation=frames["validation"],
        test=frames.get("test"),
    )


def prepare_tfidf_data(config: Config, include_test: bool = False) -> SplitBundle[TfidfDataset]:
    frames = load_split_frames(config, include_test=include_test)
    return SplitBundle(
        train=TfidfDataset(frames.train, config),
        validation=TfidfDataset(frames.validation, config),
        test=TfidfDataset(frames.test, config) if frames.test is not None else None,
    )


def prepare_bilstm_data(config: Config, include_test: bool = False) -> BiLSTMData:
    frames = load_split_frames(config, include_test=include_test)
    settings = config.data["bilstm"]
    vocabulary = Vocabulary.build(
        frames.train["statement"],
        min_count=int(settings["min_token_count"]),
        pad_token=str(settings["pad_token"]),
        unk_token=str(settings["unk_token"]),
    )
    datasets = SplitBundle(
        train=BiLSTMDataset(frames.train, config, vocabulary),
        validation=BiLSTMDataset(frames.validation, config, vocabulary),
        test=(
            BiLSTMDataset(frames.test, config, vocabulary)
            if frames.test is not None
            else None
        ),
    )
    return BiLSTMData(
        datasets=datasets,
        vocabulary=vocabulary,
        embeddings=load_glove_matrix(config, vocabulary),
    )


def prepare_distilbert_data(
    config: Config,
    tokenizer: Any | None = None,
    include_test: bool = False,
) -> SplitBundle[DistilBertDataset]:
    frames = load_split_frames(config, include_test=include_test)
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(config.data["distilbert"]["model_name"])
    return SplitBundle(
        train=DistilBertDataset(frames.train, config, tokenizer),
        validation=DistilBertDataset(frames.validation, config, tokenizer),
        test=(
            DistilBertDataset(frames.test, config, tokenizer)
            if frames.test is not None
            else None
        ),
    )


def make_loaders(
    datasets: SplitBundle[Any],
    batch_size: int,
    seed: int,
    num_workers: int = 0,
) -> LoaderBundle:
    """Create deterministic train shuffling and ordered evaluation loaders."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")
    generator = torch.Generator().manual_seed(seed)
    common = {"batch_size": batch_size, "num_workers": num_workers}
    return LoaderBundle(
        train=DataLoader(datasets.train, shuffle=True, generator=generator, **common),
        validation=DataLoader(datasets.validation, shuffle=False, **common),
        test=(DataLoader(datasets.test, shuffle=False, **common) if datasets.test is not None else None),
    )

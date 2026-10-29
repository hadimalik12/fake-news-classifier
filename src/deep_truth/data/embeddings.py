"""GloVe loading for the BiLSTM vocabulary."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from deep_truth.config import Config
from deep_truth.data.vocabulary import Vocabulary


@dataclass(frozen=True)
class EmbeddingMatrix:
    weights: torch.Tensor
    matched_tokens: int


def load_glove_matrix(config: Config, vocabulary: Vocabulary) -> EmbeddingMatrix:
    """Return deterministic trainable initial weights aligned to ``vocabulary``."""
    glove = config.data["embeddings"]["glove"]
    dimensions = int(glove["dimensions"])
    configured_dimensions = int(config.data["bilstm"]["embedding_dim"])
    if dimensions != configured_dimensions:
        raise ValueError(
            f"GloVe dimensions ({dimensions}) do not match BiLSTM embedding_dim "
            f"({configured_dimensions})."
        )
    path = config.path(glove["path"])
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. From the repository root, run ./scripts/download_data.sh."
        )

    rng = np.random.default_rng(config.seed)
    weights = rng.normal(
        loc=float(config.data["bilstm"]["oov_mean"]),
        scale=float(config.data["bilstm"]["oov_std"]),
        size=(len(vocabulary), dimensions),
    ).astype(np.float32)
    weights[vocabulary.pad_index] = 0.0

    wanted = set(vocabulary.index_to_token[2:])
    matched = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip().split()
            if len(fields) != dimensions + 1:
                raise ValueError(
                    f"{path} line {line_number} has {len(fields) - 1} values; "
                    f"expected {dimensions}."
                )
            token = fields[0]
            if token not in wanted:
                continue
            weights[vocabulary.token_to_index[token]] = np.asarray(fields[1:], dtype=np.float32)
            wanted.remove(token)
            matched += 1
            if not wanted:
                break

    return EmbeddingMatrix(weights=torch.from_numpy(weights), matched_tokens=matched)

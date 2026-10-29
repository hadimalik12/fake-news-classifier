"""Training-only vocabulary construction for the BiLSTM input pipeline."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable


def tokenize_statement(statement: str) -> list[str]:
    """Apply the lowercase, whitespace tokenizer fixed in the approach spec."""
    return statement.lower().split()


@dataclass(frozen=True)
class Vocabulary:
    """Deterministic token-to-index mapping with fixed special-token indexes."""

    index_to_token: tuple[str, ...]
    token_to_index: dict[str, int]
    pad_token: str
    unk_token: str

    @classmethod
    def build(
        cls,
        statements: Iterable[str],
        min_count: int,
        pad_token: str,
        unk_token: str,
    ) -> "Vocabulary":
        if min_count < 1:
            raise ValueError("min_count must be at least 1.")
        if pad_token == unk_token:
            raise ValueError("pad_token and unk_token must be different.")
        counts = Counter(
            token
            for statement in statements
            for token in tokenize_statement(statement)
            if token not in {pad_token, unk_token}
        )
        tokens = sorted(
            (token for token, count in counts.items() if count >= min_count),
            key=lambda token: (-counts[token], token),
        )
        index_to_token = (pad_token, unk_token, *tokens)
        token_to_index = {token: index for index, token in enumerate(index_to_token)}
        return cls(index_to_token, token_to_index, pad_token, unk_token)

    def __len__(self) -> int:
        return len(self.index_to_token)

    @property
    def pad_index(self) -> int:
        return self.token_to_index[self.pad_token]

    @property
    def unk_index(self) -> int:
        return self.token_to_index[self.unk_token]

    def encode(self, statement: str, max_length: int | None = None) -> list[int]:
        indexes = [
            self.token_to_index.get(token, self.unk_index)
            for token in tokenize_statement(statement)
        ]
        if max_length is not None:
            if max_length < 1:
                raise ValueError("max_length must be at least 1.")
            indexes = indexes[:max_length]
        return indexes

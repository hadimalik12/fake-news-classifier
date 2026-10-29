from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from deep_truth.data.embeddings import load_glove_matrix
from deep_truth.data.vocabulary import Vocabulary
from tests.helpers import make_config


class EmbeddingTests(unittest.TestCase):
    def test_glove_alignment_padding_and_seeded_oov_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root, [("1", "false", "known missing")], [])
            glove_path = config.path(config.data["embeddings"]["glove"]["path"])
            glove_path.write_text("known 0.1 0.2 0.3\nother 0.4 0.5 0.6\n", encoding="utf-8")
            vocabulary = Vocabulary.build(
                ["known missing"],
                min_count=1,
                pad_token="<pad>",
                unk_token="<unk>",
            )

            first = load_glove_matrix(config, vocabulary)
            second = load_glove_matrix(config, vocabulary)

            self.assertEqual(first.matched_tokens, 1)
            self.assertTrue(torch.equal(first.weights, second.weights))
            self.assertTrue(torch.equal(first.weights[vocabulary.pad_index], torch.zeros(3)))
            self.assertTrue(
                torch.allclose(
                    first.weights[vocabulary.token_to_index["known"]],
                    torch.tensor([0.1, 0.2, 0.3]),
                )
            )
            self.assertFalse(
                torch.equal(
                    first.weights[vocabulary.token_to_index["missing"]],
                    torch.zeros(3),
                )
            )


if __name__ == "__main__":
    unittest.main()

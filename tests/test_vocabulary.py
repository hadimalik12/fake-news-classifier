from __future__ import annotations

import unittest

from deep_truth.data.vocabulary import Vocabulary, tokenize_statement


class VocabularyTests(unittest.TestCase):
    def test_training_vocabulary_is_frequency_sorted_with_alphabetical_ties(self) -> None:
        vocabulary = Vocabulary.build(
            ["Beta alpha beta", "gamma alpha delta"],
            min_count=1,
            pad_token="<pad>",
            unk_token="<unk>",
        )
        self.assertEqual(
            vocabulary.index_to_token,
            ("<pad>", "<unk>", "alpha", "beta", "delta", "gamma"),
        )
        self.assertEqual(vocabulary.pad_index, 0)
        self.assertEqual(vocabulary.unk_index, 1)

    def test_min_count_unknown_mapping_and_truncation(self) -> None:
        vocabulary = Vocabulary.build(
            ["keep keep drop"],
            min_count=2,
            pad_token="<pad>",
            unk_token="<unk>",
        )
        self.assertEqual(vocabulary.encode("KEEP missing keep", max_length=2), [2, 1])
        self.assertEqual(tokenize_statement("  One   TWO "), ["one", "two"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from deep_truth.data.loaders import (
    load_split_frames,
    make_loaders,
    prepare_bilstm_data,
    prepare_distilbert_data,
    prepare_tfidf_data,
)
from tests.helpers import FakeTokenizer, make_config


class DataInterfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.train = [
            ("train-1", "false", "shared alpha alpha extra tokens"),
            ("train-2", "true", "shared beta"),
            ("train-3", "half-true", "shared gamma"),
            ("train-4", "mostly-true", "shared delta"),
        ]
        self.validation = [("valid-1", "true", "validationonly shared")]
        self.test = [("test-1", "pants-fire", "testonly shared")]
        self.config = make_config(self.root, self.train, self.validation, self.test)
        glove_path = self.config.path(self.config.data["embeddings"]["glove"]["path"])
        glove_path.write_text(
            "shared 0.1 0.2 0.3\n"
            "alpha 0.2 0.3 0.4\n"
            "beta 0.3 0.4 0.5\n"
            "gamma 0.4 0.5 0.6\n"
            "delta 0.5 0.6 0.7\n"
            "extra 0.6 0.7 0.8\n"
            "tokens 0.7 0.8 0.9\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_test_split_is_opt_in(self) -> None:
        self.config.split_path("test").unlink()
        frames = load_split_frames(self.config)
        self.assertIsNone(frames.test)
        self.assertEqual(len(frames.train), 4)
        self.assertEqual(frames.validation.iloc[0]["length_bin"], "medium")

        with self.assertRaises(FileNotFoundError):
            load_split_frames(self.config, include_test=True)

    def test_tfidf_view_uses_raw_text_and_common_label_indexes(self) -> None:
        data = prepare_tfidf_data(self.config)
        self.assertEqual(data.train.statements[0], self.train[0][2])
        self.assertEqual(data.train.labels[0], self.config.label_to_index["false"])
        self.assertEqual(list(data.train.metadata.columns), [
            "id", "statement", "word_count", "length_bin", "label"
        ])

    def test_bilstm_pipeline_uses_training_vocabulary_and_configured_padding(self) -> None:
        data = prepare_bilstm_data(self.config, include_test=True)
        self.assertNotIn("validationonly", data.vocabulary.token_to_index)
        self.assertNotIn("testonly", data.vocabulary.token_to_index)
        validation_item = data.datasets.validation[0]
        self.assertEqual(tuple(validation_item["input_ids"].shape), (4,))
        self.assertEqual(validation_item["input_ids"][0].item(), data.vocabulary.unk_index)
        self.assertEqual(validation_item["length"].item(), 2)
        self.assertEqual(data.datasets.train[0]["length"].item(), 4)
        self.assertEqual(data.embeddings.weights.shape[1], 3)
        self.assertIsNotNone(data.datasets.test)

    def test_distilbert_pipeline_has_fixed_shapes_and_aligned_metadata(self) -> None:
        data = prepare_distilbert_data(self.config, tokenizer=FakeTokenizer())
        item = data.train[0]
        self.assertEqual(tuple(item["input_ids"].shape), (5,))
        self.assertEqual(tuple(item["attention_mask"].shape), (5,))
        self.assertEqual(item["labels"].item(), self.config.label_to_index["false"])
        self.assertEqual(data.train.metadata.iloc[0]["id"], "train-1")
        self.assertIsNone(data.test)

    def test_train_loader_shuffle_is_reproducible_and_evaluation_order_is_stable(self) -> None:
        data = prepare_bilstm_data(self.config).datasets
        first = make_loaders(data, batch_size=2, seed=42)
        second = make_loaders(data, batch_size=2, seed=42)
        self.assertTrue(
            torch.equal(
                next(iter(first.train))["example_index"],
                next(iter(second.train))["example_index"],
            )
        )
        self.assertEqual(next(iter(first.validation))["example_index"].tolist(), [0])


if __name__ == "__main__":
    unittest.main()

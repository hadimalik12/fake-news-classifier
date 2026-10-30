from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import joblib
import pandas as pd

from deep_truth.train_tfidf import selection_key, train
from tests.helpers import make_config


class TfidfTests(unittest.TestCase):
    def test_train_only_vocabulary_and_saved_pipeline_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root, [
                ("a", "false", "wrong claim"), ("b", "true", "correct claim"),
                ("c", "false", "wrong statement"), ("d", "true", "correct statement"),
            ], [("v", "true", "correct validationonly")])
            # Training must work even when the test file is absent.
            config.split_path("test").unlink()
            config.data["tfidf"]["grid"] = {
                "ngram_range": [[1, 1]], "min_df": [1], "max_features": [100], "C": [0.1, 1.0],
            }
            artifacts, results = root / "artifacts", root / "results"
            summary = train(config, artifacts, results)
            pipeline = joblib.load(artifacts / "model.joblib")
            self.assertNotIn("validationonly", pipeline["tfidf"].vocabulary_)
            predictions = pd.read_csv(results / "validation_predictions.csv", dtype=str)
            expected = config.labels[int(pipeline.predict(["correct validationonly"])[0])]
            self.assertEqual(predictions.iloc[0]["prediction"], expected)
            self.assertEqual(predictions.iloc[0]["id"], "v")
            self.assertEqual(summary["search_configurations"], 2)
            self.assertEqual(len(pd.read_csv(results / "search.csv")), 2)

    def test_selection_uses_accuracy_then_simpler_parameters_on_rounded_ties(self):
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory), [], [])
            record = {"parameters": {"max_features": 100, "ngram_range": (1, 1), "min_df": 2, "C": 1.0},
                      "validation": {"macro_f1": 0.30001, "accuracy": 0.4}}
            other = deepcopy(record)
            other["validation"].update(macro_f1=0.30002, accuracy=0.5)
            self.assertLess(selection_key(other, config), selection_key(record, config))
            other["validation"] = deepcopy(record["validation"])
            other["parameters"]["max_features"] = 500
            self.assertLess(selection_key(record, config), selection_key(other, config))

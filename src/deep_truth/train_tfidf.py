"""Entry point for the TF-IDF plus logistic regression run."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from deep_truth.config import Config, grid_size, iter_grid
from deep_truth.data.loaders import prepare_tfidf_data
from deep_truth.metrics import compute_metrics
from deep_truth.models.tfidf import build_tfidf_model
from deep_truth.runtime import prepare_run
from deep_truth.selection import selection_key


def train(config: Config, artifact_dir: Path, results_dir: Path) -> dict:
    """Fit only on train, select on validation, and never load the test split."""
    data = prepare_tfidf_data(config)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(config.labels)
    records = []
    best_record = None
    best_model = None
    grid = config.data["tfidf"]["grid"]
    for index, parameters in enumerate(iter_grid(grid), start=1):
        model = build_tfidf_model(config, parameters)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(data.train.statements, data.train.labels)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        for warning in caught:
            warnings.warn(str(warning.message), warning.category, stacklevel=2)
        predictions = labels[model.predict(data.validation.statements)]
        metrics = compute_metrics(data.validation.metadata["label"], predictions, config.labels)
        record = {"parameters": parameters, "validation": metrics, "converged": converged}
        records.append(record)
        # Persist completed cells so a failed run still leaves useful diagnostics.
        (results_dir / "search.json").write_text(json.dumps(records, indent=2) + "\n")
        print(
            f"[{index}/{grid_size(grid)}] {parameters} "
            f"validation macro-F1={metrics['macro_f1']:.4f} "
            f"accuracy={metrics['accuracy']:.4f} converged={converged}",
            flush=True,
        )
        if best_record is None or selection_key(record, config) < selection_key(best_record, config):
            best_record, best_model = record, model

    if best_model is None:
        raise ValueError("TF-IDF search grid is empty.")
    # Keep the train-only fit: do not refit on validation after selection.
    model_path = artifact_dir / "model.joblib"
    joblib.dump(best_model, model_path)
    summary = {
        "run": "tfidf_logreg",
        "seed": config.seed,
        "labels": config.labels,
        "selected": best_record,
        "search_configurations": len(records),
        "model_path": str(model_path),
        "config": config.data,
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    output = data.validation.metadata.copy()
    output["prediction"] = labels[best_model.predict(data.validation.statements)]
    output.to_csv(results_dir / "validation_predictions.csv", index=False)
    pd.DataFrame([
        {**r["parameters"], **{k: r["validation"][k] for k in (
            "accuracy", "macro_precision", "macro_recall", "macro_f1"
        )}, "converged": r["converged"]}
        for r in records
    ]).to_csv(results_dir / "search.csv", index=False)
    print(f"Best validation macro-F1: {best_record['validation']['macro_f1']:.4f}")
    print(f"Saved model: {model_path}")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Describe the grid without fitting.")
    args = parser.parse_args(argv)
    config = prepare_run("tfidf_logreg")
    if args.dry_run:
        print(f"tfidf_logreg: seed {config.seed}, grid size {grid_size(config.data['tfidf']['grid'])}; dry run.")
        return
    train(config, config.path("artifacts/tfidf_logreg"), config.path("results/tfidf_logreg"))


if __name__ == "__main__":
    main()

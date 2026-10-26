"""Entry point for the TF-IDF plus logistic regression run."""

from __future__ import annotations

from deep_truth.runtime import describe_run, prepare_run


def main() -> None:
    config = prepare_run("tfidf_logreg")
    print(describe_run("tfidf_logreg", config.data["tfidf"]["grid"]))


if __name__ == "__main__":
    main()

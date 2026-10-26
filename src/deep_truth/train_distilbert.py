"""Entry point for the frozen and fine-tuned DistilBERT runs."""

from __future__ import annotations

import argparse

from deep_truth.runtime import describe_run, prepare_run

RUNS = ("distilbert_frozen", "distilbert_finetuned")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Load the DistilBERT run config.")
    parser.add_argument("--run", required=True, choices=RUNS)
    args = parser.parse_args(argv)
    config = prepare_run(args.run)
    section = "frozen" if args.run == "distilbert_frozen" else "finetuned"
    print(describe_run(args.run, config.data["distilbert"][section]["grid"]))


if __name__ == "__main__":
    main()

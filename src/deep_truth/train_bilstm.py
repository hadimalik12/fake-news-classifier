"""Entry point for the BiLSTM run."""

from __future__ import annotations

from deep_truth.runtime import describe_run, prepare_run


def main() -> None:
    config = prepare_run("bilstm")
    print(describe_run("bilstm", config.data["bilstm"]["grid"]))


if __name__ == "__main__":
    main()

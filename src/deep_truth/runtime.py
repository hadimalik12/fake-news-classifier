"""Helpers shared by the training entry points."""

from __future__ import annotations

from deep_truth.config import Config, grid_size, load_config
from deep_truth.seed import set_seed


def prepare_run(run_name: str) -> Config:
    config = load_config()
    known = config.data["runs"]["all"]
    if run_name not in known:
        raise KeyError(f"Unknown run {run_name!r}. config.yaml lists {known}.")
    set_seed(config.seed)
    return config


def describe_run(run_name: str, search_grid: dict) -> str:
    return f"{run_name}: seed loaded, grid size {grid_size(search_grid)}. Fitting is a later step."

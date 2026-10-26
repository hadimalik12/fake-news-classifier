"""Load the shared experiment config.

Scripts import this module instead of restating seeds, labels, paths, or grids.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import yaml


def find_repo_root(start: Path | None = None) -> Path:
    """Return the repository root that contains config.yaml and the approach spec."""
    start = (start or Path.cwd()).resolve()
    candidates = [start, *start.parents]
    package_dir = Path(__file__).resolve().parent
    candidates.extend(package_dir.parents)
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "config.yaml").is_file() and (candidate / "docs" / "approach.md").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find config.yaml next to docs/approach.md. Run commands from the repository root."
    )


def parse_probability(value: Any) -> float:
    """Parse a YAML probability written as a float or a fraction string such as '1/3'."""
    if isinstance(value, str):
        if "/" not in value:
            return float(value)
        numerator, denominator = value.split("/", maxsplit=1)
        return float(numerator) / float(denominator)
    return float(value)


def iter_grid(grid: dict[str, list[Any]]) -> Iterator[dict[str, Any]]:
    """Yield each combination in a search grid, in config key order."""
    keys = list(grid)
    if not keys:
        yield {}
        return
    head, *rest = keys
    for value in grid[head]:
        normalized = tuple(value) if isinstance(value, list) else value
        for partial in iter_grid({key: grid[key] for key in rest}):
            yield {head: normalized, **partial}


def grid_size(grid: dict[str, list[Any]]) -> int:
    size = 1
    for values in grid.values():
        size *= len(values)
    return size


@dataclass(frozen=True)
class Config:
    """Resolved experiment config. Paths are absolute. Derived fields may be None."""

    root: Path
    data: dict[str, Any]

    @property
    def seed(self) -> int:
        return int(self.data["seed"])

    @property
    def labels(self) -> list[str]:
        return list(self.data["labels"])

    @property
    def label_to_index(self) -> dict[str, int]:
        return {label: index for index, label in enumerate(self.labels)}

    def path(self, relative: str) -> Path:
        path = Path(relative)
        if path.is_absolute():
            return path
        return self.root / path

    def split_path(self, split: str) -> Path:
        return self.path(self.data["splits"][split]["path"])

    def split_examples(self, split: str) -> int:
        return int(self.data["splits"][split]["examples"])

    @property
    def comparison_decimals(self) -> int:
        return int(self.data["metric"]["comparison_decimals"])

    @property
    def length_cutpoints(self) -> tuple[float, float] | None:
        bins = self.data["length_bins"]
        if bins["q1"] is None or bins["q2"] is None:
            return None
        return float(bins["q1"]), float(bins["q2"])

    def require_length_cutpoints(self) -> tuple[float, float]:
        cutpoints = self.length_cutpoints
        if cutpoints is None:
            raise RuntimeError(
                "Length cutpoints are empty in config.yaml. Data setup has to write q1 and q2 before binning."
            )
        return cutpoints


def load_config(root: Path | None = None) -> Config:
    repo_root = find_repo_root(root)
    with (repo_root / "config.yaml").open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("config.yaml must contain a mapping at the top level.")
    labels = data.get("labels")
    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        raise ValueError("config.yaml labels must be quoted strings. YAML otherwise reads true and false as booleans.")
    return Config(root=repo_root, data=data)

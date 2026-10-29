"""Download and extract the external files required by the experiment."""

from __future__ import annotations

import csv
import hashlib
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from deep_truth.config import Config, load_config

USER_AGENT = "deep-truth-data-setup"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Checksum mismatch for {path}: expected {expected}, found {actual}.")


def download_file(urls: list[str], destination: Path, expected_sha256: str) -> None:
    """Download to a temporary file and atomically move it into place."""
    if destination.is_file():
        verify_checksum(destination, expected_sha256)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for url in urls:
        print(f"Downloading {destination.name} from {url}")
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            verify_checksum(temporary, expected_sha256)
            temporary.replace(destination)
            return
        except (OSError, urllib.error.URLError, ValueError) as error:
            last_error = error
            temporary.unlink(missing_ok=True)
    raise RuntimeError(f"Could not download {destination.name} from any configured URL.") from last_error


def extract_members(archive_path: Path, members: dict[str, Path]) -> None:
    """Extract selected archive members without unpacking unrelated files."""
    missing_destinations = {name: path for name, path in members.items() if not path.is_file()}
    if not missing_destinations:
        return
    with zipfile.ZipFile(archive_path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise zipfile.BadZipFile(f"CRC check failed for {bad_member} in {archive_path}.")
        available = {Path(name).name: name for name in archive.namelist()}
        missing_members = sorted(set(missing_destinations) - set(available))
        if missing_members:
            raise FileNotFoundError(f"{archive_path} does not contain {missing_members}.")
        for member_name, destination in missing_destinations.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".part")
            with archive.open(available[member_name]) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target)
            temporary.replace(destination)


def ensure_liar_files(config: Config) -> list[Path]:
    dataset = config.data["dataset"]
    expected = {
        dataset["files"][split]: config.split_path(split)
        for split in ("train", "validation", "test")
    }
    archive_path = config.path(dataset["archive_path"])
    if not all(path.is_file() for path in expected.values()):
        download_file(list(dataset["urls"]), archive_path, dataset["archive_sha256"])
        extract_members(archive_path, expected)
    paths = list(expected.values())
    validate_liar_files(config, paths)
    return paths


def validate_liar_files(config: Config, paths: list[Path]) -> None:
    labels = set(config.labels)
    split_names = ("train", "validation", "test")
    for split, path in zip(split_names, paths):
        rows = 0
        unknown: set[str] = set()
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE):
                if len(row) <= int(config.data["columns"]["statement"]):
                    raise ValueError(f"{path} contains a malformed row at line {rows + 1}.")
                rows += 1
                if row[int(config.data["columns"]["label"])] not in labels:
                    unknown.add(row[int(config.data["columns"]["label"])])
        expected_rows = config.split_examples(split)
        if rows != expected_rows:
            raise ValueError(f"{path} has {rows} rows; expected {expected_rows}.")
        if unknown:
            raise ValueError(f"{path} contains unknown labels: {sorted(unknown)}.")


def ensure_glove_file(config: Config) -> Path:
    glove = config.data["embeddings"]["glove"]
    destination = config.path(glove["path"])
    if not destination.is_file():
        archive_path = config.path(glove["archive_path"])
        download_file(list(glove["urls"]), archive_path, glove["archive_sha256"])
        extract_members(archive_path, {glove["file"]: destination})
    validate_glove_file(destination, int(glove["dimensions"]), int(glove["vocabulary_size"]))
    return destination


def validate_glove_file(path: Path, dimensions: int, expected_rows: int) -> None:
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for rows, line in enumerate(handle, start=1):
            fields = line.rstrip().split()
            if len(fields) != dimensions + 1:
                raise ValueError(
                    f"{path} line {rows} has {len(fields) - 1} values; expected {dimensions}."
                )
    if rows != expected_rows:
        raise ValueError(f"{path} has {rows} vectors; expected {expected_rows}.")


def ensure_data_files(config: Config) -> list[Path]:
    """Ensure all external data files required by the planned runs exist."""
    return [*ensure_liar_files(config), ensure_glove_file(config)]


def main() -> None:
    config = load_config()
    paths = ensure_data_files(config)
    print("Data files are ready:")
    for path in paths:
        print(f"  {path.relative_to(config.root)}")


if __name__ == "__main__":
    main()

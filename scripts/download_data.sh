#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${DEEP_TRUTH_PYTHON:-${repo_root}/.venv/bin/python}"

if [[ ! -x "${python_bin}" ]]; then
  echo "Project environment not found. Run ./scripts/setup.sh first." >&2
  exit 1
fi

cd "${repo_root}"
exec "${python_bin}" -m deep_truth.data.download

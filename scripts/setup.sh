#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
venv_path="${DEEP_TRUTH_VENV:-${repo_root}/.venv}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="${PYTHON_BIN}"
else
  for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      python_bin="${candidate}"
      break
    fi
  done
fi

if [[ -z "${python_bin:-}" ]]; then
  echo "Python 3.10 through 3.13 is required." >&2
  exit 1
fi

if ! "${python_bin}" -c 'import sys; raise SystemExit(not ((3, 10) <= sys.version_info[:2] < (3, 14)))'; then
  echo "${python_bin} is not Python 3.10 through 3.13." >&2
  exit 1
fi

if [[ ! -x "${venv_path}/bin/python" ]]; then
  "${python_bin}" -m venv "${venv_path}"
fi

"${venv_path}/bin/python" -m pip install --upgrade pip
"${venv_path}/bin/python" -m pip install --editable "${repo_root}"

echo "Environment ready at ${venv_path}"
echo "Next: ./scripts/download_data.sh"

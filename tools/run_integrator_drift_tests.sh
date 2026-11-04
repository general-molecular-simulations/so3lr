#!/usr/bin/env bash

set -euo pipefail

if (( $# != 1 )); then
  echo "Usage: $0 /path/to/python" >&2
  exit 2
fi

python_bin=$1
if [[ ! -x ${python_bin} ]]; then
  echo "Python executable is missing or not executable: ${python_bin}" >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_dir=$(cd -- "${script_dir}/.." && pwd -P)
cd "${repo_dir}"

# Repeated JAX/XLA compilation of every ensemble in one interpreter can exhaust
# native compiler resources on large CPU nodes. Discover the parametrized
# nodeids first, then give each numerical threshold a clean Python process.
collection=$("${python_bin}" -m pytest -p no:cacheprovider --collect-only -q \
  --full tests/test_integrator_drift.py)
nodeids=()
while IFS= read -r nodeid; do
  if [[ ${nodeid} == tests/test_integrator_drift.py::* ]]; then
    nodeids+=("${nodeid}")
  fi
done <<< "${collection}"

if (( ${#nodeids[@]} == 0 )); then
  echo "No integrator drift tests were collected" >&2
  exit 1
fi

printf 'Running %d integrator drift cases in isolated processes\n' "${#nodeids[@]}"
for nodeid in "${nodeids[@]}"; do
  "${python_bin}" -m pytest -p no:cacheprovider -v --tb=short --full "${nodeid}"
done

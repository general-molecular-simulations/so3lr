#!/usr/bin/env bash
set -euo pipefail

demo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$demo_dir/../.." && pwd)"

if [[ -x "$project_dir/.venv/bin/so3lr" ]]; then
    so3lr_cli="$project_dir/.venv/bin/so3lr"
elif command -v so3lr >/dev/null 2>&1; then
    so3lr_cli="$(command -v so3lr)"
else
    printf '%s\n' "SO3LR is not installed. From the project root, run: uv sync --locked" >&2
    exit 1
fi

mkdir -p "$demo_dir/output"
run_dir="$(mktemp -d "$demo_dir/output/run-XXXXXXXX")"
cp "$demo_dir/ethanol.xyz" "$demo_dir/settings.yaml" "$run_dir/"
printf 'Demo outputs: %s\n' "$run_dir"
cd "$run_dir"
exec "$so3lr_cli" --settings settings.yaml "$@"

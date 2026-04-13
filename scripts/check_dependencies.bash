#!/usr/bin/env bash
set -euo pipefail

missing=()
issues=()

for command in docker pre-commit uv psql; do
  if ! command -v "$command" >/dev/null 2>&1; then
    missing+=("$command")
  fi
done

if ((${#missing[@]} > 0)); then
  printf 'Missing required commands: %s\n' "${missing[*]}" >&2
  printf 'See README.md for installation instructions.\n' >&2
  exit 1
fi

if [[ ! -f .pre-commit-config.yaml ]]; then
  issues+=("Missing .pre-commit-config.yaml")
fi

if [[ ! -f .secrets.baseline ]]; then
  issues+=("Missing .secrets.baseline")
fi

if ! grep -q 'id: detect-secrets' .pre-commit-config.yaml; then
  issues+=("detect-secrets hook is not configured in .pre-commit-config.yaml")
fi

hook_path="$(git rev-parse --git-path hooks/pre-commit)"

if [[ ! -f "$hook_path" ]]; then
  issues+=("Git pre-commit hook is not installed; run: pre-commit install")
elif ! grep -q 'pre-commit' "$hook_path"; then
  issues+=("Git pre-commit hook is not managed by pre-commit; run: pre-commit install --overwrite")
fi

if ((${#issues[@]} > 0)); then
  printf '%s\n' "${issues[@]}" >&2
  exit 1
fi
